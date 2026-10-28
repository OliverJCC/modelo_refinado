from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FILENAME_RE = re.compile(
    r"^capture_(?P<date>\d{8})_(?P<time>\d{6})_"
    r"(?P<label>[A-Za-z])_(?P<subject>S\d+)\.csv$"
)


@dataclass
class FileQuality:
    source_file: str
    subject_id: str = ""
    label: str = ""
    rows: int = 0
    candidate_windows_row_order: int = 0
    windows: int = 0
    status: str = "used"
    reason: str = ""
    missing_fraction: float = 0.0
    repaired_timestamps: int = 0
    discontinuities: int = 0
    segments: int = 0


def parse_capture_filename(name: str) -> dict[str, str]:
    match = FILENAME_RE.match(Path(name).name)
    if not match:
        raise ValueError(f"Nombre de captura no reconocido: {name}")
    result = match.groupdict()
    result["label"] = result["label"].upper()
    return result


def _repair_isolated_timestamps(ts_ns: np.ndarray, expected_dt: float) -> tuple[np.ndarray, int]:
    """Repair isolated timestamp spikes while preserving acquisition row order."""
    repaired = ts_ns.astype(np.float64, copy=True) / 1e9
    repairs = 0
    if repaired.size < 3:
        return repaired, repairs

    lower_neighbor_span = 0.5 * (2.0 * expected_dt)
    upper_neighbor_span = 2.5 * (2.0 * expected_dt)
    for index in range(1, repaired.size - 1):
        previous, current, following = repaired[index - 1 : index + 2]
        if not np.all(np.isfinite([previous, current, following])):
            continue
        neighbor_span = following - previous
        left_dt = current - previous
        right_dt = following - current
        neighbors_are_continuous = lower_neighbor_span <= neighbor_span <= upper_neighbor_span
        current_is_outlier = (
            left_dt <= 0
            or right_dt <= 0
            or left_dt > 5.0 * expected_dt
            or right_dt > 5.0 * expected_dt
        )
        if neighbors_are_continuous and current_is_outlier:
            repaired[index] = previous + neighbor_span / 2.0
            repairs += 1
    return repaired, repairs


def _continuous_segments(
    repaired_seconds: np.ndarray,
    gap_seconds: float,
) -> tuple[list[tuple[int, int]], int]:
    if repaired_seconds.size == 0:
        return [], 0
    diffs = np.diff(repaired_seconds)
    breaks = np.flatnonzero(
        ~np.isfinite(diffs) | (diffs <= 0.0) | (diffs > gap_seconds)
    )
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [repaired_seconds.size]))
    return list(zip(starts.tolist(), ends.tolist())), int(breaks.size)


def _impute_features(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    missing_mask = numeric.isna().to_numpy()
    numeric = numeric.interpolate(method="linear", axis=0, limit_direction="both")
    numeric = numeric.fillna(0.0)
    values = numeric.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Quedaron valores no finitos después de la imputación")
    return values, missing_mask


def build_dataset_from_zip(
    zip_path: str | Path,
    output_npz: str | Path,
    manifest_csv: str | Path,
    quality_csv: str | Path,
    summary_json: str | Path,
    features: list[str],
    sampling_rate_hz: float = 25.0,
    window_size: int = 128,
    step_size: int = 64,
    max_window_imputed_fraction: float = 0.2,
    timestamp_gap_seconds: float = 0.2,
) -> dict[str, object]:
    zip_path = Path(zip_path)
    output_npz = Path(output_npz)
    manifest_csv = Path(manifest_csv)
    quality_csv = Path(quality_csv)
    summary_json = Path(summary_json)

    if not zip_path.is_file():
        raise FileNotFoundError(f"No existe el ZIP: {zip_path}")
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size y step_size deben ser positivos")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    quality_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    windows: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []
    source_files: list[str] = []
    capture_dates: list[str] = []
    start_rows: list[int] = []
    end_rows: list[int] = []
    start_timestamps: list[float] = []
    end_timestamps: list[float] = []
    imputed_fractions: list[float] = []
    segment_ids: list[int] = []
    quality_rows: list[FileQuality] = []
    content_hashes: set[str] = set()

    expected_columns = {"ts_ns", "label", *features}
    expected_dt = 1.0 / sampling_rate_hz

    with zipfile.ZipFile(zip_path) as archive:
        csv_names = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
        if not csv_names:
            raise RuntimeError(f"El ZIP no contiene archivos CSV: {zip_path}")

        for name in csv_names:
            quality = FileQuality(source_file=name)
            try:
                parsed = parse_capture_filename(name)
                quality.subject_id = parsed["subject"]
                quality.label = parsed["label"]
                raw = archive.read(name)
                digest = hashlib.sha256(raw).hexdigest()
                if digest in content_hashes:
                    quality.status = "skipped"
                    quality.reason = "duplicate_content"
                    quality_rows.append(quality)
                    continue
                content_hashes.add(digest)

                frame = pd.read_csv(io.BytesIO(raw))
                quality.rows = int(len(frame))
                quality.candidate_windows_row_order = max(
                    0, 1 + (quality.rows - window_size) // step_size
                )
                missing_columns = sorted(expected_columns.difference(frame.columns))
                if missing_columns:
                    raise ValueError(f"missing_columns:{','.join(missing_columns)}")

                file_labels = set(frame["label"].dropna().astype(str).str.upper().unique())
                if file_labels != {parsed["label"]}:
                    raise ValueError(
                        f"label_mismatch:filename={parsed['label']},csv={sorted(file_labels)}"
                    )

                values, missing_mask = _impute_features(frame, features)
                quality.missing_fraction = float(missing_mask.mean())
                raw_ts = pd.to_numeric(frame["ts_ns"], errors="coerce").to_numpy(dtype=np.float64)
                repaired_ts, repaired_count = _repair_isolated_timestamps(raw_ts, expected_dt)
                segments, discontinuities = _continuous_segments(repaired_ts, timestamp_gap_seconds)
                quality.repaired_timestamps = repaired_count
                quality.discontinuities = discontinuities
                quality.segments = len(segments)

                for segment_number, (segment_start, segment_end) in enumerate(segments):
                    segment_length = segment_end - segment_start
                    if segment_length < window_size:
                        continue
                    last_start = segment_end - window_size
                    for start in range(segment_start, last_start + 1, step_size):
                        end = start + window_size
                        missing_fraction = float(missing_mask[start:end].mean())
                        if missing_fraction > max_window_imputed_fraction:
                            continue
                        windows.append(values[start:end].T.copy())
                        labels.append(parsed["label"])
                        subjects.append(parsed["subject"])
                        source_files.append(name)
                        capture_dates.append(parsed["date"])
                        start_rows.append(start)
                        end_rows.append(end)
                        start_timestamps.append(float(repaired_ts[start]))
                        end_timestamps.append(float(repaired_ts[end - 1]))
                        imputed_fractions.append(missing_fraction)
                        segment_ids.append(segment_number)
                        quality.windows += 1

                if quality.windows == 0:
                    quality.status = "skipped"
                    quality.reason = (
                        "too_short"
                        if quality.rows < window_size
                        else "temporal_discontinuities_or_quality"
                    )
            except Exception as exc:
                quality.status = "skipped"
                quality.reason = str(exc)
            quality_rows.append(quality)

    if not windows:
        raise RuntimeError("No se generaron ventanas válidas")

    X = np.stack(windows).astype(np.float32)
    label_names = np.asarray(sorted(set(labels)), dtype="U32")
    label_to_index = {label: index for index, label in enumerate(label_names.tolist())}
    y = np.asarray([label_to_index[label] for label in labels], dtype=np.int64)
    window_ids = np.arange(len(windows), dtype=np.int64)

    np.savez_compressed(
        output_npz,
        X=X,
        y=y,
        window_id=window_ids,
        subject_id=np.asarray(subjects, dtype="U16"),
        source_file=np.asarray(source_files, dtype="U128"),
        capture_date=np.asarray(capture_dates, dtype="U16"),
        start_row=np.asarray(start_rows, dtype=np.int64),
        end_row=np.asarray(end_rows, dtype=np.int64),
        start_timestamp_s=np.asarray(start_timestamps, dtype=np.float64),
        end_timestamp_s=np.asarray(end_timestamps, dtype=np.float64),
        imputed_fraction=np.asarray(imputed_fractions, dtype=np.float32),
        segment_id=np.asarray(segment_ids, dtype=np.int32),
        label_names=label_names,
        feature_names=np.asarray(features, dtype="U32"),
        sampling_rate_hz=np.asarray(sampling_rate_hz, dtype=np.float64),
        window_size=np.asarray(window_size, dtype=np.int64),
        step_size=np.asarray(step_size, dtype=np.int64),
    )

    manifest = pd.DataFrame(
        {
            "window_id": window_ids,
            "subject_id": subjects,
            "label": labels,
            "label_index": y,
            "source_file": source_files,
            "capture_date": capture_dates,
            "start_row": start_rows,
            "end_row": end_rows,
            "start_timestamp_s": start_timestamps,
            "end_timestamp_s": end_timestamps,
            "imputed_fraction": imputed_fractions,
            "segment_id": segment_ids,
        }
    )
    manifest.to_csv(manifest_csv, index=False)
    quality_frame = pd.DataFrame(asdict(row) for row in quality_rows)
    quality_frame.to_csv(quality_csv, index=False)

    class_counts = Counter(labels)
    subject_counts = Counter(subjects)
    summary: dict[str, object] = {
        "source_zip": str(zip_path.resolve()),
        "output_npz": str(output_npz.resolve()),
        "manifest_csv": str(manifest_csv.resolve()),
        "n_windows": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "window_size": int(X.shape[2]),
        "n_subjects": len(subject_counts),
        "subjects": sorted(subject_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "subject_window_counts": dict(sorted(subject_counts.items())),
        "files_total": len(quality_rows),
        "files_used": int((quality_frame["windows"] > 0).sum()),
        "files_without_windows": int((quality_frame["windows"] == 0).sum()),
        "candidate_windows_row_order": int(
            quality_frame["candidate_windows_row_order"].sum()
        ),
        "windows_removed_by_quality_control": int(
            quality_frame["candidate_windows_row_order"].sum() - X.shape[0]
        ),
        "timestamp_repairs": int(quality_frame["repaired_timestamps"].sum()),
        "remaining_discontinuities": int(quality_frame["discontinuities"].sum()),
        "feature_names": features,
        "label_to_index": label_to_index,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary

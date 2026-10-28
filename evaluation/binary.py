from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


BINARY_CLASS_NAMES = ("no_tallado", "tallado")
POSITIVE_CONTEXTS = frozenset(("A", "B", "C"))


def binary_targets(context_labels: Iterable[str]) -> np.ndarray:
    labels = np.asarray(list(context_labels), dtype=str)
    unknown = sorted(set(labels).difference((*POSITIVE_CONTEXTS, "D")))
    if unknown:
        raise ValueError(f"Contextos desconocidos: {unknown}")
    return np.isin(labels, list(POSITIVE_CONTEXTS)).astype(np.int64)


def binary_classification_metrics(
    y_true: Iterable[int],
    probability_tallado: Iterable[float],
    threshold: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probability_tallado, dtype=np.float64)
    if y_true.shape != probability.shape:
        raise ValueError("y_true y probability_tallado deben tener la misma forma")
    if not 0.0 < threshold < 1.0:
        raise ValueError("El umbral debe estar entre 0 y 1")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Las probabilidades deben estar dentro de [0, 1]")

    y_pred = (probability >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    has_both = set(np.unique(y_true)) == {0, 1}
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_pred)) if has_both else float("nan")
        ),
        "precision_tallado": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity_recall_tallado": float(sensitivity),
        "specificity_no_tallado": float(specificity),
        "f1_tallado": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if has_both else float("nan"),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)) if has_both else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, probability)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, probability)) if has_both else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, y_pred, matrix


def select_operating_threshold(
    y_true: Iterable[int],
    probability_tallado: Iterable[float],
    minimum_sensitivity: float = 0.95,
) -> tuple[float, pd.DataFrame]:
    """Select the most specific threshold meeting a sensitivity constraint."""
    if not 0.0 < minimum_sensitivity <= 1.0:
        raise ValueError("minimum_sensitivity debe estar en (0, 1]")
    rows = []
    for threshold in np.round(np.linspace(0.01, 0.99, 99), 2):
        metrics, _, _ = binary_classification_metrics(y_true, probability_tallado, threshold)
        rows.append(metrics)
    table = pd.DataFrame(rows)
    eligible = table[table["sensitivity_recall_tallado"] >= minimum_sensitivity]
    if eligible.empty:
        selected = table.sort_values(
            ["sensitivity_recall_tallado", "specificity_no_tallado", "balanced_accuracy"],
            ascending=False,
        ).iloc[0]
    else:
        selected = eligible.sort_values(
            ["specificity_no_tallado", "balanced_accuracy", "threshold"],
            ascending=False,
        ).iloc[0]
    table["selected"] = np.isclose(table["threshold"], float(selected["threshold"]))
    return float(selected["threshold"]), table


def binary_metrics_by_group(
    y_true: np.ndarray,
    probability_tallado: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    group_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in sorted(np.unique(groups).tolist()):
        mask = groups == group
        aggregate, _, _ = binary_classification_metrics(
            y_true[mask], probability_tallado[mask], threshold
        )
        rows.append({group_name: group, "n_windows": int(mask.sum()), **aggregate})
    return pd.DataFrame(rows)


def context_sensitivity_table(
    context_labels: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for context in ("A", "B", "C", "D"):
        mask = context_labels == context
        if not mask.any():
            continue
        predicted_positive_rate = float(y_pred[mask].mean())
        rows.append(
            {
                "context": context,
                "target": "tallado" if context in POSITIVE_CONTEXTS else "no_tallado",
                "n_windows": int(mask.sum()),
                "sensitivity": predicted_positive_rate if context in POSITIVE_CONTEXTS else float("nan"),
                "specificity": 1.0 - predicted_positive_rate if context == "D" else float("nan"),
                "predicted_tallado_rate": predicted_positive_rate,
            }
        )
    return pd.DataFrame(rows)


def _positive_runs(frame: pd.DataFrame, step_size: int, minimum_windows: int) -> list[pd.DataFrame]:
    ordered = frame.sort_values("start_row")
    runs: list[list[int]] = []
    current: list[int] = []
    previous_start: int | None = None
    for index, row in ordered.iterrows():
        start = int(row["start_row"])
        is_positive = int(row["y_pred"]) == 1
        is_contiguous = previous_start is None or start - previous_start <= int(step_size * 1.5)
        if is_positive and is_contiguous:
            current.append(index)
        else:
            if len(current) >= minimum_windows:
                runs.append(current)
            current = [index] if is_positive else []
        previous_start = start
    if len(current) >= minimum_windows:
        runs.append(current)
    return [ordered.loc[indexes] for indexes in runs]


def event_detection_metrics(
    predictions: pd.DataFrame,
    sampling_rate_hz: float,
    window_size: int,
    step_size: int,
    minimum_consecutive_windows: int = 2,
) -> tuple[dict[str, float], pd.DataFrame]:
    required = {
        "subject_id", "context", "source_file", "segment_id", "start_row",
        "start_timestamp_s", "end_timestamp_s", "y_pred",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Columnas ausentes para eventos: {missing}")

    rows: list[dict[str, object]] = []
    grouping = ["subject_id", "context", "source_file", "segment_id"]
    for keys, frame in predictions.groupby(grouping, sort=True):
        subject, context, source_file, segment_id = keys
        runs = _positive_runs(frame, step_size, minimum_consecutive_windows)
        duration_seconds = max(
            float(frame["end_timestamp_s"].max() - frame["start_timestamp_s"].min()),
            window_size / sampling_rate_hz,
        )
        true_event = context in POSITIVE_CONTEXTS
        evaluable = len(frame) >= minimum_consecutive_windows
        detected = bool(runs) if true_event and evaluable else float("nan")
        latency = (
            float(runs[0]["start_timestamp_s"].min() - frame["start_timestamp_s"].min())
            if detected is True or detected == 1.0
            else float("nan")
        )
        rows.append(
            {
                "subject_id": subject,
                "context": context,
                "source_file": source_file,
                "segment_id": int(segment_id),
                "n_windows": int(len(frame)),
                "duration_minutes": duration_seconds / 60.0,
                "true_event": true_event,
                "evaluable": evaluable,
                "detected": detected,
                "false_alarm_events": (
                    int(len(runs)) if not true_event and evaluable else float("nan")
                ),
                "detection_latency_seconds": latency,
            }
        )
    event_frame = pd.DataFrame(rows)
    positives_all = event_frame[event_frame["true_event"]]
    negatives_all = event_frame[~event_frame["true_event"]]
    positives = positives_all[positives_all["evaluable"]]
    negatives = negatives_all[negatives_all["evaluable"]]
    negative_minutes = float(negatives["duration_minutes"].sum())
    latencies = positives.loc[positives["detected"], "detection_latency_seconds"].dropna()
    aggregate = {
        "minimum_consecutive_windows": int(minimum_consecutive_windows),
        "positive_capture_segments_total": int(len(positives_all)),
        "positive_capture_segments_excluded_short": int((~positives_all["evaluable"]).sum()),
        "positive_capture_segments": int(len(positives)),
        "detected_capture_segments": int(positives["detected"].dropna().astype(bool).sum()),
        "event_sensitivity": float(positives["detected"].dropna().mean()) if len(positives) else float("nan"),
        "negative_capture_segments_total": int(len(negatives_all)),
        "negative_capture_segments_excluded_short": int((~negatives_all["evaluable"]).sum()),
        "negative_minutes": negative_minutes,
        "false_alarm_events": int(negatives["false_alarm_events"].sum()),
        "false_alarms_per_minute": (
            float(negatives["false_alarm_events"].sum() / negative_minutes)
            if negative_minutes > 0
            else float("nan")
        ),
        "median_detection_latency_seconds": float(latencies.median()) if len(latencies) else float("nan"),
        "p95_detection_latency_seconds": float(latencies.quantile(0.95)) if len(latencies) else float("nan"),
    }
    return aggregate, event_frame

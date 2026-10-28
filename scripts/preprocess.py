from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smartwatch_pipeline.config import load_config, project_path
from smartwatch_pipeline.preprocessing import build_dataset_from_zip
from smartwatch_pipeline.splits import create_group_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocesamiento leakage-safe de smartwatch")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--zip-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    split_config = config["split"]
    source_zip = args.zip_path or Path(data_config["source_zip"])

    dataset_path = project_path("data", "processed", "smartwatch_windows.npz")
    raw_manifest = project_path("data", "manifests", "windows_raw.csv")
    quality_csv = project_path("data", "manifests", "file_quality.csv")
    dataset_summary = project_path("data", "manifests", "dataset_summary.json")
    split_manifest = project_path("data", "manifests", "windows_with_splits.csv")
    subject_manifest = project_path("data", "manifests", "subjects_with_splits.csv")
    split_summary = project_path("data", "manifests", "split_summary.json")

    dataset = build_dataset_from_zip(
        zip_path=source_zip,
        output_npz=dataset_path,
        manifest_csv=raw_manifest,
        quality_csv=quality_csv,
        summary_json=dataset_summary,
        features=list(data_config["features"]),
        sampling_rate_hz=float(data_config["sampling_rate_hz"]),
        window_size=int(data_config["window_size"]),
        step_size=int(data_config["step_size"]),
        max_window_imputed_fraction=float(data_config["max_window_imputed_fraction"]),
        timestamp_gap_seconds=float(data_config["timestamp_gap_seconds"]),
    )
    splits = create_group_splits(
        manifest_csv=raw_manifest,
        output_windows_csv=split_manifest,
        output_subjects_csv=subject_manifest,
        output_summary_json=split_summary,
        test_subject_count=int(split_config["test_subjects"]),
        validation_subject_count=int(split_config["validation_subjects"]),
        cv_folds=int(split_config["cv_folds"]),
        seed=int(config["seed"]),
    )
    print(json.dumps({"dataset": dataset, "splits": splits}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

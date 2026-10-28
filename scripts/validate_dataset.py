from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validar dataset y ausencia de fuga por usuario")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "smartwatch_windows.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "windows_with_splits.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    with np.load(args.dataset, allow_pickle=False) as dataset:
        required = {
            "X", "y", "window_id", "subject_id", "source_file", "label_names",
            "feature_names", "sampling_rate_hz", "window_size", "step_size",
        }
        missing = required.difference(dataset.files)
        assert not missing, f"Faltan claves en NPZ: {sorted(missing)}"
        X, y = dataset["X"], dataset["y"]
        assert X.ndim == 3 and X.shape[0] == y.shape[0]
        assert len(manifest) == len(y)
        assert np.array_equal(manifest["window_id"].to_numpy(), dataset["window_id"])
        assert np.isfinite(X).all(), "X contiene NaN o infinito"
        assert set(np.unique(y)) == set(range(len(dataset["label_names"])))

    split_subjects = {
        split: set(frame["subject_id"].unique())
        for split, frame in manifest.groupby("holdout_split")
    }
    assert not (split_subjects["train"] & split_subjects["validation"])
    assert not (split_subjects["train"] & split_subjects["test"])
    assert not (split_subjects["validation"] & split_subjects["test"])
    assert manifest.loc[manifest["holdout_split"] == "test", "cv_validation_fold"].eq(-1).all()
    assert manifest.loc[manifest["holdout_split"] != "test", "cv_validation_fold"].ge(0).all()

    result = {
        "status": "OK",
        "windows": int(len(manifest)),
        "subjects": int(manifest["subject_id"].nunique()),
        "classes": sorted(manifest["label"].unique().tolist()),
        "windows_by_split": manifest["holdout_split"].value_counts().to_dict(),
        "subjects_by_split": {
            name: len(values) for name, values in split_subjects.items()
        },
        "windows_by_class": manifest["label"].value_counts().sort_index().to_dict(),
        "cv_validation_users": {
            str(int(fold)): int(frame["subject_id"].nunique())
            for fold, frame in manifest[manifest["cv_validation_fold"] >= 0].groupby("cv_validation_fold")
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from smartwatch_pipeline.splits import create_group_splits


class SplitTests(unittest.TestCase):
    def test_subjects_are_disjoint(self) -> None:
        rows = []
        window_id = 0
        for subject_index in range(10):
            subject = f"S{subject_index + 1:02d}"
            for label_index, label in enumerate("ABCD"):
                for _ in range(3 + ((subject_index + label_index) % 2)):
                    rows.append(
                        {
                            "window_id": window_id,
                            "subject_id": subject,
                            "label": label,
                            "label_index": label_index,
                        }
                    )
                    window_id += 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.csv"
            windows = root / "windows.csv"
            subjects = root / "subjects.csv"
            summary = root / "summary.json"
            pd.DataFrame(rows).to_csv(raw, index=False)
            create_group_splits(
                raw,
                windows,
                subjects,
                summary,
                test_subject_count=2,
                validation_subject_count=2,
                cv_folds=3,
                seed=42,
            )
            result = pd.read_csv(windows)
            groups = {
                name: set(frame["subject_id"])
                for name, frame in result.groupby("holdout_split")
            }
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["validation"] & groups["test"])
            self.assertTrue(result.loc[result["holdout_split"] == "test", "cv_validation_fold"].eq(-1).all())


if __name__ == "__main__":
    unittest.main()

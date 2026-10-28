from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from evaluation.binary import (
    binary_classification_metrics,
    binary_targets,
    event_detection_metrics,
    select_operating_threshold,
)


class BinaryEvaluationTests(unittest.TestCase):
    def test_context_mapping(self) -> None:
        mapped = binary_targets(["A", "B", "C", "D"])
        np.testing.assert_array_equal(mapped, [1, 1, 1, 0])

    def test_threshold_respects_sensitivity_constraint(self) -> None:
        y = np.asarray([0, 0, 0, 1, 1, 1])
        probability = np.asarray([0.1, 0.4, 0.8, 0.6, 0.7, 0.9])
        threshold, table = select_operating_threshold(y, probability, minimum_sensitivity=2 / 3)
        selected = table[table["selected"]].iloc[0]
        self.assertAlmostEqual(threshold, 0.6)
        self.assertGreaterEqual(selected["sensitivity_recall_tallado"], 2 / 3)

    def test_event_metrics_require_two_consecutive_windows(self) -> None:
        frame = pd.DataFrame(
            {
                "subject_id": ["S01"] * 6,
                "context": ["A"] * 3 + ["D"] * 3,
                "source_file": ["positive.csv"] * 3 + ["negative.csv"] * 3,
                "segment_id": [0] * 6,
                "start_row": [0, 64, 128, 0, 64, 128],
                "start_timestamp_s": [0.0, 2.56, 5.12, 0.0, 2.56, 5.12],
                "end_timestamp_s": [5.08, 7.64, 10.2, 5.08, 7.64, 10.2],
                "y_pred": [0, 1, 1, 1, 0, 1],
            }
        )
        aggregate, _ = event_detection_metrics(frame, 25.0, 128, 64, 2)
        self.assertEqual(aggregate["detected_capture_segments"], 1)
        self.assertEqual(aggregate["false_alarm_events"], 0)

    def test_binary_metrics_confusion(self) -> None:
        metrics, prediction, matrix = binary_classification_metrics(
            [0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9], 0.5
        )
        np.testing.assert_array_equal(prediction, [0, 1, 1, 1])
        np.testing.assert_array_equal(matrix, [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["sensitivity_recall_tallado"], 1.0)


if __name__ == "__main__":
    unittest.main()

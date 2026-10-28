from __future__ import annotations

import unittest

import numpy as np

from evaluation.metrics import classification_metrics, metrics_by_user


class MetricsTests(unittest.TestCase):
    def test_perfect_multiclass_metrics(self) -> None:
        y = np.asarray([0, 1, 2, 3, 0, 1, 2, 3])
        probabilities = np.eye(4)[y] * 0.97 + 0.03 / 4
        aggregate, per_class, matrix = classification_metrics(y, y, probabilities, list("ABCD"))
        self.assertAlmostEqual(aggregate["accuracy"], 1.0)
        self.assertAlmostEqual(aggregate["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(aggregate["f1_macro"], 1.0)
        self.assertTrue((per_class["specificity"] == 1.0).all())
        self.assertEqual(matrix.shape, (4, 4))

    def test_user_metrics_keep_missing_auc_as_nan(self) -> None:
        y = np.asarray([0, 0, 1, 1])
        pred = y.copy()
        probabilities = np.full((4, 4), 0.01)
        probabilities[np.arange(4), y] = 0.97
        users = np.asarray(["S01", "S01", "S02", "S02"])
        frame = metrics_by_user(y, pred, probabilities, users, list("ABCD"))
        self.assertEqual(len(frame), 2)
        self.assertTrue(frame["roc_auc_ovr_macro"].isna().all())


if __name__ == "__main__":
    unittest.main()

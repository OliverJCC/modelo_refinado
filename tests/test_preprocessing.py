from __future__ import annotations

import unittest

import numpy as np

from smartwatch_pipeline.preprocessing import (
    _continuous_segments,
    _repair_isolated_timestamps,
    parse_capture_filename,
)


class PreprocessingTests(unittest.TestCase):
    def test_parse_filename(self) -> None:
        parsed = parse_capture_filename("capture_20260618_132028_A_S24.csv")
        self.assertEqual(parsed["subject"], "S24")
        self.assertEqual(parsed["label"], "A")
        self.assertEqual(parsed["date"], "20260618")

    def test_isolated_timestamp_is_repaired(self) -> None:
        seconds = np.asarray([0.00, 0.04, 99.0, 0.12, 0.16])
        repaired, count = _repair_isolated_timestamps(seconds * 1e9, expected_dt=0.04)
        self.assertEqual(count, 1)
        self.assertAlmostEqual(repaired[2], 0.08, places=6)

    def test_discontinuity_creates_segments(self) -> None:
        seconds = np.asarray([0.00, 0.04, 0.08, 1.00, 1.04])
        segments, breaks = _continuous_segments(seconds, gap_seconds=0.2)
        self.assertEqual(segments, [(0, 3), (3, 5)])
        self.assertEqual(breaks, 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.binary import event_detection_metrics
from smartwatch_pipeline.config import load_config
from smartwatch_pipeline.data import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruir métricas binarias de eventos")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_config()
    arrays = load_dataset(PROJECT_ROOT / "data" / "processed" / "smartwatch_windows.npz")
    metric_dir = PROJECT_ROOT / "metrics" / args.run_id
    predictions = pd.read_csv(metric_dir / "predictions.csv")
    summary_rows = []
    detail_frames = []
    for (model, seed, fold), frame in predictions.groupby(["model", "seed", "fold"]):
        aggregate, details = event_detection_metrics(
            frame,
            arrays.sampling_rate_hz,
            int(arrays.X.shape[2]),
            int(config["data"]["step_size"]),
            minimum_consecutive_windows=2,
        )
        summary_rows.append({"model": model, "seed": seed, "fold": fold, **aggregate})
        details.insert(0, "model", model)
        details.insert(1, "seed", seed)
        details.insert(2, "fold", fold)
        detail_frames.append(details)
    pd.DataFrame(summary_rows).to_csv(metric_dir / "event_summary.csv", index=False)
    pd.concat(detail_frames, ignore_index=True).to_csv(metric_dir / "event_details.csv", index=False)
    print(f"Métricas de eventos reconstruidas: {metric_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = ("fold_metrics.csv", "class_metrics.csv", "user_metrics.csv", "predictions.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidar corridas K-Fold compatibles")
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--source-run-ids", nargs="+", required=True)
    args = parser.parse_args()

    output_metric_dir = PROJECT_ROOT / "metrics" / args.output_run_id
    output_result_dir = PROJECT_ROOT / "results" / args.output_run_id
    output_metric_dir.mkdir(parents=True, exist_ok=True)
    output_result_dir.mkdir(parents=True, exist_ok=True)

    for table_name in TABLES:
        frames = []
        for run_id in args.source_run_ids:
            path = PROJECT_ROOT / "metrics" / run_id / table_name
            if not path.exists():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path))
        pd.concat(frames, ignore_index=True).to_csv(output_metric_dir / table_name, index=False)

    for run_id in args.source_run_ids:
        source_dir = PROJECT_ROOT / "metrics" / run_id
        for history_path in source_dir.glob("history_*.csv"):
            destination = output_metric_dir / history_path.name
            if destination.exists():
                raise FileExistsError(f"Historia duplicada: {destination}")
            shutil.copy2(history_path, destination)

    (output_result_dir / "merged_runs.json").write_text(
        json.dumps({"source_run_ids": args.source_run_ids}, indent=2),
        encoding="utf-8",
    )
    print(f"Corridas consolidadas en: {args.output_run_id}")


if __name__ == "__main__":
    main()

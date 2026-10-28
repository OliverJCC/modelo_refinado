from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import aggregate_fold_metrics
from evaluation.plotting import plot_model_comparison
from evaluation.statistics import cluster_bootstrap_confidence_intervals, paired_user_wilcoxon


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparar modelos usando métricas K-Fold")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--primary-metric", default="f1_macro")
    args = parser.parse_args()
    metric_dir = PROJECT_ROOT / "metrics" / args.run_id
    plot_dir = PROJECT_ROOT / "plots" / args.run_id
    result_dir = PROJECT_ROOT / "results" / args.run_id
    fold_metrics = pd.read_csv(metric_dir / "fold_metrics.csv")
    user_metrics = pd.read_csv(metric_dir / "user_metrics.csv")
    predictions = pd.read_csv(metric_dir / "predictions.csv")
    if args.primary_metric not in fold_metrics.columns:
        raise ValueError(f"Métrica ausente: {args.primary_metric}")
    summary = aggregate_fold_metrics(fold_metrics)
    summary = summary.sort_values(f"{args.primary_metric}_mean", ascending=False)
    summary.to_csv(metric_dir / "model_comparison.csv", index=False)
    plot_model_comparison(
        fold_metrics,
        args.primary_metric,
        plot_dir / f"comparison_{args.primary_metric}.png",
        plot_dir / f"boxplot_{args.primary_metric}.png",
    )
    winner = summary.iloc[0]
    class_names = [
        column.removeprefix("probability_")
        for column in predictions.columns
        if column.startswith("probability_")
    ]
    confidence = cluster_bootstrap_confidence_intervals(
        predictions, class_names, n_bootstrap=2000, seed=42
    )
    confidence.to_csv(metric_dir / "cluster_bootstrap_confidence_intervals.csv", index=False)
    statistical_tests = paired_user_wilcoxon(
        user_metrics, args.primary_metric, baseline_model=str(winner["model"])
    )
    statistical_tests.to_csv(metric_dir / "paired_user_wilcoxon.csv", index=False)
    markdown = [
        "# Comparación de modelos",
        "",
        f"Métrica primaria: `{args.primary_metric}`.",
        "",
        f"Mejor modelo por validación cruzada: **{winner['model']}** ",
        f"({winner[f'{args.primary_metric}_mean']:.4f} de media).",
        "",
        "El conjunto test fijo todavía no se ha usado. La arquitectura mejorada debe ",
        "definirse a partir de este resultado antes de abrir el test.",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
    ]
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "comparison_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

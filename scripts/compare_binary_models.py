from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import aggregate_fold_metrics
from evaluation.plotting import plot_model_comparison
from evaluation.statistics import paired_user_wilcoxon


def _aggregate(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(groups, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = dict(zip(groups, keys))
        row["n_runs"] = int(len(group))
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _binary_bootstrap(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap complete users while retaining each fold's selected predictions."""
    rng = np.random.default_rng(seed)
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "sensitivity_recall_tallado",
        "specificity_no_tallado",
        "f1_tallado",
        "mcc",
        "roc_auc",
        "pr_auc",
    )
    rows: list[dict[str, object]] = []
    for (model, model_seed), frame in predictions.groupby(["model", "seed"], sort=True):
        subjects = np.asarray(sorted(frame["subject_id"].unique()), dtype=str)
        subject_data = {
            subject: (
                group["y_true"].to_numpy(np.int64),
                group["y_pred"].to_numpy(np.int64),
                group["probability_tallado"].to_numpy(float),
            )
            for subject, group in frame.groupby("subject_id", sort=False)
        }
        estimates = {metric: [] for metric in metric_names}
        for _ in range(n_bootstrap):
            sampled = rng.choice(subjects, size=len(subjects), replace=True)
            y_true = np.concatenate([subject_data[subject][0] for subject in sampled])
            y_pred = np.concatenate([subject_data[subject][1] for subject in sampled])
            probability = np.concatenate([subject_data[subject][2] for subject in sampled])
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            values = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "sensitivity_recall_tallado": tp / (tp + fn),
                "specificity_no_tallado": tn / (tn + fp),
                "f1_tallado": float(f1_score(y_true, y_pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "roc_auc": float(roc_auc_score(y_true, probability)),
                "pr_auc": float(average_precision_score(y_true, probability)),
            }
            for metric, value in values.items():
                estimates[metric].append(value)
        for metric, values in estimates.items():
            array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "model": model,
                    "seed": int(model_seed),
                    "metric": metric,
                    "bootstrap_unit": "subject",
                    "n_subjects": int(len(subjects)),
                    "n_bootstrap": int(n_bootstrap),
                    "estimate": float(np.nanmedian(array)),
                    "ci_2.5": float(np.nanpercentile(array, 2.5)),
                    "ci_97.5": float(np.nanpercentile(array, 97.5)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparar los modelos binarios por usuario")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--primary-metric", default="balanced_accuracy")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metric_dir = PROJECT_ROOT / "metrics" / args.run_id
    plot_dir = PROJECT_ROOT / "plots" / args.run_id
    result_dir = PROJECT_ROOT / "results" / args.run_id
    plot_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = pd.read_csv(metric_dir / "fold_metrics.csv")
    user_metrics = pd.read_csv(metric_dir / "user_metrics.csv")
    predictions = pd.read_csv(metric_dir / "predictions.csv")
    context_metrics = pd.read_csv(metric_dir / "context_metrics.csv")
    event_summary = pd.read_csv(metric_dir / "event_summary.csv")

    if args.primary_metric not in fold_metrics:
        raise ValueError(f"Métrica primaria ausente: {args.primary_metric}")
    summary = aggregate_fold_metrics(fold_metrics).sort_values(
        f"{args.primary_metric}_mean", ascending=False
    )
    summary.to_csv(metric_dir / "model_comparison.csv", index=False)
    winner = str(summary.iloc[0]["model"])

    context_summary = _aggregate(
        context_metrics,
        ["model", "context", "target"],
        ["n_windows", "sensitivity", "specificity", "predicted_tallado_rate"],
    )
    context_summary.to_csv(metric_dir / "context_comparison.csv", index=False)
    event_metrics = [
        "positive_capture_segments_total",
        "positive_capture_segments_excluded_short",
        "positive_capture_segments",
        "detected_capture_segments",
        "event_sensitivity",
        "negative_capture_segments_total",
        "negative_capture_segments_excluded_short",
        "negative_minutes",
        "false_alarm_events",
        "false_alarms_per_minute",
        "median_detection_latency_seconds",
        "p95_detection_latency_seconds",
    ]
    event_comparison = _aggregate(event_summary, ["model"], event_metrics)
    event_comparison.to_csv(metric_dir / "event_comparison.csv", index=False)

    confidence = _binary_bootstrap(predictions, args.bootstrap, args.seed)
    confidence.to_csv(metric_dir / "cluster_bootstrap_confidence_intervals.csv", index=False)
    tests = paired_user_wilcoxon(user_metrics, args.primary_metric, winner)
    tests.to_csv(metric_dir / "paired_user_wilcoxon.csv", index=False)

    for metric in (
        args.primary_metric,
        "sensitivity_recall_tallado",
        "specificity_no_tallado",
        "f1_tallado",
    ):
        plot_model_comparison(
            fold_metrics,
            metric,
            plot_dir / f"comparison_{metric}.png",
            plot_dir / f"boxplot_{metric}.png",
        )

    main_columns = [
        "model",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "sensitivity_recall_tallado_mean",
        "specificity_no_tallado_mean",
        "f1_tallado_mean",
        "roc_auc_mean",
        "pr_auc_mean",
        "selected_threshold_mean",
    ]
    display = summary[[column for column in main_columns if column in summary]].copy()
    markdown = [
        "# Comparación binaria por validación cruzada",
        "",
        "Objetivo: detectar tallado de ojos (A/B/C) frente a cualquier otra acción (D).",
        "Los folds están separados por usuario y el umbral de cada fold se eligió únicamente",
        "con usuarios de validación internos, exigiendo sensibilidad mínima de 95 %.",
        "",
        f"Modelo congelado por `{args.primary_metric}`: **{winner}**.",
        "",
        "```text",
        display.to_string(index=False),
        "```",
        "",
        "Los intervalos de confianza usan bootstrap por usuario, no por ventanas.",
        "Los eventos exigen dos ventanas positivas consecutivas; los segmentos con menos",
        "de dos ventanas se excluyen explícitamente del cálculo de sensibilidad de eventos.",
    ]
    (result_dir / "comparison_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(display.to_string(index=False))
    print(f"\nModelo seleccionado: {winner}")


if __name__ == "__main__":
    main()

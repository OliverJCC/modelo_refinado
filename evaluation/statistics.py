from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef


def cluster_bootstrap_confidence_intervals(
    predictions: pd.DataFrame,
    class_names: list[str],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap subjects, not correlated windows, to obtain 95% intervals."""
    metrics_to_report = ["accuracy", "balanced_accuracy", "f1_macro", "mcc"]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    grouping_columns = [column for column in ("model", "seed") if column in predictions]
    for keys, frame in predictions.groupby(grouping_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(grouping_columns, keys))
        subjects = np.asarray(sorted(frame["subject_id"].unique()), dtype=str)
        estimates: dict[str, list[float]] = {metric: [] for metric in metrics_to_report}
        for _ in range(n_bootstrap):
            sampled = rng.choice(subjects, size=len(subjects), replace=True)
            pieces = [frame[frame["subject_id"] == subject] for subject in sampled]
            bootstrap = pd.concat(pieces, ignore_index=True)
            y_true = bootstrap["y_true"].to_numpy()
            y_pred = bootstrap["y_pred"].to_numpy()
            aggregate = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
            }
            for metric in metrics_to_report:
                estimates[metric].append(aggregate[metric])
        for metric, values in estimates.items():
            values_array = np.asarray(values, dtype=float)
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "bootstrap_unit": "subject",
                    "n_subjects": len(subjects),
                    "n_bootstrap": n_bootstrap,
                    "ci_2.5": float(np.nanpercentile(values_array, 2.5)),
                    "ci_97.5": float(np.nanpercentile(values_array, 97.5)),
                }
            )
    return pd.DataFrame(rows)


def paired_user_wilcoxon(
    user_metrics: pd.DataFrame,
    metric: str,
    baseline_model: str,
) -> pd.DataFrame:
    """Compare models over matched users and apply Holm correction."""
    collapsed = user_metrics.groupby(["subject_id", "model"], as_index=False)[metric].mean()
    pivot = collapsed.pivot(index="subject_id", columns="model", values=metric)
    rows = []
    for model in pivot.columns:
        if model == baseline_model:
            continue
        aligned = pivot[[baseline_model, model]].dropna()
        if len(aligned) < 2 or np.allclose(aligned[baseline_model], aligned[model]):
            statistic, p_value = float("nan"), float("nan")
        else:
            statistic, p_value = wilcoxon(
                aligned[model], aligned[baseline_model], alternative="two-sided"
            )
        rows.append(
            {
                "model": model,
                "baseline_model": baseline_model,
                "metric": metric,
                "n_paired_subjects": len(aligned),
                "mean_difference": float((aligned[model] - aligned[baseline_model]).mean()),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    valid = result["p_value"].notna()
    ordered = result.loc[valid, "p_value"].sort_values().index.tolist()
    corrected = pd.Series(np.nan, index=result.index, dtype=float)
    running_max = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, float(result.loc[index, "p_value"]) * (total - rank))
        running_max = max(running_max, adjusted)
        corrected.loc[index] = running_max
    result["p_value_holm"] = corrected
    return result

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def _safe_multiclass_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    result = {
        "roc_auc_ovr_macro": float("nan"),
        "roc_auc_ovr_weighted": float("nan"),
        "pr_auc_macro": float("nan"),
        "pr_auc_weighted": float("nan"),
    }
    if set(np.unique(y_true)) != set(labels.tolist()):
        return result
    binary = label_binarize(y_true, classes=labels)
    try:
        result["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="macro")
        )
        result["roc_auc_ovr_weighted"] = float(
            roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="weighted")
        )
        result["pr_auc_macro"] = float(average_precision_score(binary, probabilities, average="macro"))
        result["pr_auc_weighted"] = float(
            average_precision_score(binary, probabilities, average="weighted")
        )
    except ValueError:
        pass
    return result


def classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    probabilities: np.ndarray,
    class_names: Iterable[str],
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    class_names = np.asarray(list(class_names), dtype=str)
    labels = np.arange(len(class_names), dtype=np.int64)
    if probabilities.shape != (len(y_true), len(labels)):
        raise ValueError("La matriz de probabilidades tiene dimensiones incorrectas")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Las probabilidades deben ser finitas y no negativas")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Cada fila de probabilidades debe tener suma positiva")
    # CSV round-tripping can move a softmax row a few ulps away from 1.0.
    # Re-normalize before sklearn's strict probabilistic metrics.
    probabilities = probabilities / row_sums

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    has_multiple_true_classes = len(np.unique(y_true)) > 1
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_pred))
            if has_multiple_true_classes
            else float("nan")
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if has_multiple_true_classes else float("nan"),
        "cohen_kappa": (
            float(cohen_kappa_score(y_true, y_pred))
            if has_multiple_true_classes
            else float("nan")
        ),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
    }
    metrics.update(_safe_multiclass_auc(y_true, probabilities, labels))

    per_class_rows = []
    for class_index, class_name in enumerate(class_names):
        tp = int(cm[class_index, class_index])
        fn = int(cm[class_index, :].sum() - tp)
        fp = int(cm[:, class_index].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        sensitivity = tp / (tp + fn) if tp + fn else float("nan")
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = (
            2.0 * precision * sensitivity / (precision + sensitivity)
            if np.isfinite(sensitivity) and precision + sensitivity > 0
            else 0.0
        )
        per_class_rows.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "support": int(cm[class_index, :].sum()),
                "sensitivity_recall": sensitivity,
                "specificity": specificity,
                "precision": precision,
                "f1": f1,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            }
        )
    return metrics, pd.DataFrame(per_class_rows), cm


def metrics_by_user(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    subject_ids: np.ndarray,
    class_names: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject in sorted(np.unique(subject_ids).tolist()):
        mask = subject_ids == subject
        aggregate, _, _ = classification_metrics(
            y_true[mask], y_pred[mask], probabilities[mask], class_names
        )
        rows.append({"subject_id": subject, "n_windows": int(mask.sum()), **aggregate})
    return pd.DataFrame(rows)


def aggregate_fold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"fold", "seed", "n_windows", "best_epoch"}
    metric_columns = [column for column in numeric if column not in excluded]
    rows = []
    for model, group in frame.groupby("model"):
        row: dict[str, object] = {"model": model, "n_runs": len(group)}
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def _prepare(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_history(history: pd.DataFrame, output_path: str | Path) -> None:
    output_path = _prepare(output_path)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["validation_accuracy"], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[2].plot(history["epoch"], history["validation_f1_macro"], label="validation")
    axes[2].set(title="Macro F1", xlabel="Epoch", ylabel="F1")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    normalize: bool = False,
) -> None:
    output_path = _prepare(output_path)
    display_matrix = matrix.astype(float) if normalize else matrix.astype(int)
    if normalize:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            display_matrix,
            row_sums,
            out=np.zeros_like(display_matrix),
            where=row_sums != 0,
        )
    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(display_matrix, display_labels=class_names).plot(
        ax=axis, cmap="Blues", values_format=".2f" if normalize else "d", colorbar=False
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_roc_pr_curves(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    roc_path: str | Path,
    pr_path: str | Path,
) -> None:
    labels = np.arange(len(class_names))
    roc_figure, roc_axis = plt.subplots(figsize=(7, 6))
    pr_figure, pr_axis = plt.subplots(figsize=(7, 6))
    for index, name in enumerate(class_names):
        binary = (y_true == labels[index]).astype(int)
        if len(np.unique(binary)) < 2:
            continue
        RocCurveDisplay.from_predictions(binary, probabilities[:, index], name=name, ax=roc_axis)
        PrecisionRecallDisplay.from_predictions(binary, probabilities[:, index], name=name, ax=pr_axis)
    roc_axis.plot([0, 1], [0, 1], "--", color="gray")
    roc_axis.set_title("ROC one-vs-rest")
    pr_axis.set_title("Precision-Recall one-vs-rest")
    roc_figure.tight_layout()
    pr_figure.tight_layout()
    roc_figure.savefig(_prepare(roc_path), dpi=200)
    pr_figure.savefig(_prepare(pr_path), dpi=200)
    plt.close(roc_figure)
    plt.close(pr_figure)


def plot_model_comparison(
    fold_metrics: pd.DataFrame,
    metric: str,
    bar_path: str | Path,
    box_path: str | Path,
) -> None:
    order = (
        fold_metrics.groupby("model")[metric].mean().sort_values(ascending=False).index.tolist()
    )
    values = [fold_metrics.loc[fold_metrics["model"] == name, metric].dropna() for name in order]
    means = [float(value.mean()) for value in values]
    stds = [float(value.std(ddof=1)) if len(value) > 1 else 0.0 for value in values]

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(order, means, yerr=stds, capsize=4)
    axis.set(title=f"Comparación por {metric}", ylabel=metric)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(_prepare(bar_path), dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.boxplot(values, tick_labels=order, showmeans=True)
    axis.set(title=f"Variabilidad entre folds: {metric}", ylabel=metric)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(_prepare(box_path), dpi=200)
    plt.close(figure)

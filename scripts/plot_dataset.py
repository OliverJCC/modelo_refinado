from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _save(figure: plt.Figure, name: str) -> None:
    output = PROJECT_ROOT / "plots" / "dataset" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    windows = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "windows_with_splits.csv")
    quality = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "file_quality.csv")

    class_counts = windows["label"].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(class_counts.index, class_counts.values)
    axis.set(title="Ventanas válidas por clase", xlabel="Clase", ylabel="Ventanas")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, "windows_by_class.png")

    subject_counts = windows.groupby(["subject_id", "holdout_split"]).size().reset_index(name="windows")
    colors = {"train": "#4C78A8", "validation": "#F2CF5B", "test": "#E45756"}
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(
        subject_counts["subject_id"],
        subject_counts["windows"],
        color=[colors[value] for value in subject_counts["holdout_split"]],
    )
    axis.set(title="Ventanas por usuario y partición", xlabel="Usuario", ylabel="Ventanas")
    axis.tick_params(axis="x", rotation=45)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values()]
    axis.legend(handles, colors.keys(), title="Partición")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, "windows_by_subject_split.png")

    candidate = int(quality["candidate_windows_row_order"].sum())
    retained = int(quality["windows"].sum())
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["Candidatas", "Retenidas"], [candidate, retained], color=["#BAB0AC", "#59A14F"])
    axis.set(title="Efecto del control temporal", ylabel="Ventanas")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, "quality_control_windows.png")
    print(f"Gráficas de dataset guardadas en: {PROJECT_ROOT / 'plots' / 'dataset'}")


if __name__ == "__main__":
    main()

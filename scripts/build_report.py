from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir informe académico de una corrida")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    metric_dir = PROJECT_ROOT / "metrics" / args.run_id
    result_dir = PROJECT_ROOT / "results" / args.run_id
    comparison_path = metric_dir / "model_comparison.csv"
    fold_path = metric_dir / "fold_metrics.csv"
    if not comparison_path.exists() or not fold_path.exists():
        raise FileNotFoundError("Ejecuta compare_models.py antes de build_report.py")

    comparison = pd.read_csv(comparison_path)
    folds = pd.read_csv(fold_path)
    best_f1 = comparison.sort_values("f1_macro_mean", ascending=False).iloc[0]
    best_balanced = comparison.sort_values("balanced_accuracy_mean", ascending=False).iloc[0]
    most_stable = comparison.sort_values("f1_macro_std", ascending=True).iloc[0]

    generalization_gaps = []
    for history_path in metric_dir.glob("history_*.csv"):
        history = pd.read_csv(history_path)
        best_row = history.loc[history["validation_f1_macro"].idxmax()]
        generalization_gaps.append(
            {
                "model": best_row.get("model", history_path.stem.split("_")[1]),
                "accuracy_gap": float(best_row["train_accuracy"] - best_row["validation_accuracy"]),
                "loss_gap": float(best_row["validation_loss"] - best_row["train_loss"]),
            }
        )
    gap_frame = pd.DataFrame(generalization_gaps)
    gap_summary = (
        gap_frame.groupby("model", as_index=False).mean()
        if not gap_frame.empty
        else pd.DataFrame(columns=["model", "accuracy_gap", "loss_gap"])
    )
    gap_summary.to_csv(metric_dir / "overfitting_indicators.csv", index=False)

    report = [
        "# Informe comparativo",
        "",
        f"Mejor modelo por macro-F1: **{best_f1['model']}** "
        f"({best_f1['f1_macro_mean']:.4f} ± {best_f1['f1_macro_std']:.4f}).",
        "",
        f"Mejor modelo por balanced accuracy: **{best_balanced['model']}** "
        f"({best_balanced['balanced_accuracy_mean']:.4f} ± "
        f"{best_balanced['balanced_accuracy_std']:.4f}).",
        "",
        f"Menor variabilidad de macro-F1: **{most_stable['model']}** "
        f"(desviación estándar {most_stable['f1_macro_std']:.4f}).",
        "",
        "## Confiabilidad",
        "",
        "Todos los folds separan usuarios completos. El test fijo permanece fuera de K-Fold, "
        "la normalización se ajusta dentro de cada entrenamiento y los intervalos de confianza "
        "se calculan agrupando por usuario. Esto elimina la fuga por ventanas presente en el "
        "pipeline anterior, aunque el número reducido y desigual de usuarios sigue limitando "
        "la precisión de las estimaciones.",
        "",
        "## Sobreajuste",
        "",
        "Los archivos `overfitting_indicators.csv` y las curvas por fold cuantifican las "
        "diferencias train-validation en el mejor epoch. Una brecha alta y consistente, junto "
        "con pérdida de validación creciente, se interpreta como evidencia de sobreajuste.",
        "",
        "## Tabla completa",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
    ]
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "academic_summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Informe guardado en: {result_dir / 'academic_summary.md'}")


if __name__ == "__main__":
    main()

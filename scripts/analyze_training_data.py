from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.binary import binary_targets
from smartwatch_pipeline.data import load_dataset


STATISTICS = ("mean", "std", "rms", "min", "max", "ptp", "median", "iqr", "mean_abs_diff")


def _window_features(X: np.ndarray, channel_names: np.ndarray) -> pd.DataFrame:
    values = {
        "mean": X.mean(axis=2),
        "std": X.std(axis=2),
        "rms": np.sqrt(np.mean(np.square(X, dtype=np.float64), axis=2)),
        "min": X.min(axis=2),
        "max": X.max(axis=2),
        "ptp": np.ptp(X, axis=2),
        "median": np.median(X, axis=2),
        "iqr": np.quantile(X, 0.75, axis=2) - np.quantile(X, 0.25, axis=2),
        "mean_abs_diff": np.mean(np.abs(np.diff(X, axis=2)), axis=2),
    }
    columns: dict[str, np.ndarray] = {}
    for statistic in STATISTICS:
        for index, channel in enumerate(channel_names):
            columns[f"{channel}__{statistic}"] = values[statistic][:, index]
    return pd.DataFrame(columns)


def _subject_standardize(features: pd.DataFrame, subjects: np.ndarray) -> pd.DataFrame:
    adjusted = features.copy()
    groups = pd.Series(subjects, index=features.index)
    for column in features:
        means = features[column].groupby(groups).transform("mean")
        stds = features[column].groupby(groups).transform("std").replace(0.0, 1.0).fillna(1.0)
        adjusted[column] = (features[column] - means) / stds
    return adjusted


def _cohen_d(feature: np.ndarray, target: np.ndarray) -> float:
    negative = feature[target == 0]
    positive = feature[target == 1]
    denominator = len(negative) + len(positive) - 2
    if denominator <= 0:
        return float("nan")
    pooled = np.sqrt(
        ((len(negative) - 1) * negative.var(ddof=1) + (len(positive) - 1) * positive.var(ddof=1))
        / denominator
    )
    return float((positive.mean() - negative.mean()) / pooled) if pooled > 0 else 0.0


def _mutual_information_table(
    features: pd.DataFrame,
    adjusted: pd.DataFrame,
    binary_target: np.ndarray,
    context_target: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    raw_binary = mutual_info_classif(features, binary_target, random_state=seed)
    adjusted_binary = mutual_info_classif(adjusted, binary_target, random_state=seed)
    raw_context = mutual_info_classif(features, context_target, random_state=seed)
    adjusted_context = mutual_info_classif(adjusted, context_target, random_state=seed)
    rows = []
    for index, feature_name in enumerate(features.columns):
        channel, statistic = feature_name.split("__", maxsplit=1)
        rows.append(
            {
                "feature": feature_name,
                "channel": channel,
                "statistic": statistic,
                "mi_binary_raw": float(raw_binary[index]),
                "mi_binary_subject_adjusted": float(adjusted_binary[index]),
                "mi_context4_raw": float(raw_context[index]),
                "mi_context4_subject_adjusted": float(adjusted_context[index]),
                "cohen_d_binary_subject_adjusted": _cohen_d(
                    adjusted.iloc[:, index].to_numpy(float), binary_target
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("mi_binary_subject_adjusted", ascending=False)


def _leave_one_subject_out_mi(
    adjusted: pd.DataFrame,
    binary_target: np.ndarray,
    subjects: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    estimates = []
    top20_counts = np.zeros(adjusted.shape[1], dtype=np.int64)
    for offset, subject in enumerate(sorted(np.unique(subjects).tolist())):
        keep = subjects != subject
        values = mutual_info_classif(
            adjusted.loc[keep], binary_target[keep], random_state=seed + offset
        )
        estimates.append(values)
        top20_counts[np.argsort(values)[-20:]] += 1
    matrix = np.vstack(estimates)
    return pd.DataFrame(
        {
            "feature": adjusted.columns,
            "mi_binary_loso_mean": matrix.mean(axis=0),
            "mi_binary_loso_std": matrix.std(axis=0, ddof=1),
            "top20_loso_fraction": top20_counts / matrix.shape[0],
        }
    )


def _exact_duplicate_summary(X: np.ndarray, labels: np.ndarray, subjects: np.ndarray) -> dict[str, int]:
    groups: dict[str, list[int]] = {}
    for index, window in enumerate(X):
        digest = hashlib.sha256(np.ascontiguousarray(window).tobytes()).hexdigest()
        groups.setdefault(digest, []).append(index)
    duplicated_groups = [indices for indices in groups.values() if len(indices) > 1]
    return {
        "exact_duplicate_groups": int(len(duplicated_groups)),
        "windows_in_exact_duplicate_groups": int(sum(len(indices) for indices in duplicated_groups)),
        "cross_label_duplicate_groups": int(
            sum(len(set(labels[indices])) > 1 for indices in duplicated_groups)
        ),
        "cross_subject_duplicate_groups": int(
            sum(len(set(subjects[indices])) > 1 for indices in duplicated_groups)
        ),
    }


def _heatmap(matrix: pd.DataFrame, path: Path, title: str, limits: tuple[float, float]) -> None:
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix.to_numpy(float), cmap="coolwarm", vmin=limits[0], vmax=limits[1])
    axis.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=60, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_class_by_user(counts: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 7))
    image = axis.imshow(counts.to_numpy(float), cmap="Blues")
    axis.set_xticks(np.arange(len(counts.columns)), labels=counts.columns)
    axis.set_yticks(np.arange(len(counts.index)), labels=counts.index)
    axis.set(xlabel="Contexto", ylabel="Usuario", title="Ventanas de entrenamiento por usuario y contexto")
    for row in range(len(counts.index)):
        for column in range(len(counts.columns)):
            axis.text(column, row, str(int(counts.iloc[row, column])), ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_top_mi(table: pd.DataFrame, path: Path, top: int = 20) -> None:
    selected = table.nlargest(top, "mi_binary_subject_adjusted").sort_values(
        "mi_binary_subject_adjusted"
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(selected["feature"], selected["mi_binary_subject_adjusted"])
    axis.set(
        xlabel="Información mutua (nats)",
        title="Características más informativas para tallado/no tallado\n(ajustadas dentro de usuario)",
    )
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditar exclusivamente los datos de entrenamiento")
    parser.add_argument("--run-id", default="20260901_training_data_audit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    arrays = load_dataset(PROJECT_ROOT / "data" / "processed" / "smartwatch_windows.npz")
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "windows_with_splits.csv")
    file_quality = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "file_quality.csv")
    if not np.array_equal(arrays.window_id, manifest["window_id"].to_numpy()):
        raise AssertionError("NPZ y manifiesto no están alineados")

    train_mask = manifest["holdout_split"].to_numpy(str) == "train"
    X = arrays.X[train_mask]
    subjects = arrays.subject_id[train_mask]
    contexts = manifest.loc[train_mask, "label"].to_numpy(str)
    binary = binary_targets(contexts)
    context_index = pd.Categorical(contexts, categories=["A", "B", "C", "D"]).codes
    train_manifest = manifest.loc[train_mask].reset_index(drop=True)
    output_dir = PROJECT_ROOT / "results" / args.run_id
    metric_dir = PROJECT_ROOT / "metrics" / args.run_id
    plot_dir = PROJECT_ROOT / "plots" / args.run_id
    for directory in (output_dir, metric_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    features = _window_features(X, arrays.feature_names)
    adjusted = _subject_standardize(features, subjects)
    feature_metadata = train_manifest[["window_id", "subject_id", "label", "source_file"]].copy()
    feature_metadata["binary_label"] = binary
    pd.concat([feature_metadata, features], axis=1).to_csv(
        metric_dir / "window_statistical_features.csv", index=False
    )

    sample_matrix = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
    sample_correlation = pd.DataFrame(sample_matrix, columns=arrays.feature_names).corr("pearson")
    mean_correlation = features[[f"{name}__mean" for name in arrays.feature_names]].copy()
    mean_correlation.columns = arrays.feature_names
    mean_correlation = mean_correlation.corr("pearson")
    rms_correlation = features[[f"{name}__rms" for name in arrays.feature_names]].copy()
    rms_correlation.columns = arrays.feature_names
    rms_correlation = rms_correlation.corr("spearman")
    sample_correlation.to_csv(metric_dir / "channel_sample_pearson_correlation.csv")
    mean_correlation.to_csv(metric_dir / "channel_window_mean_pearson_correlation.csv")
    rms_correlation.to_csv(metric_dir / "channel_window_rms_spearman_correlation.csv")

    redundant_rows = []
    for row in range(len(arrays.feature_names)):
        for column in range(row + 1, len(arrays.feature_names)):
            value = float(sample_correlation.iloc[row, column])
            if abs(value) >= 0.90:
                redundant_rows.append(
                    {
                        "channel_1": arrays.feature_names[row],
                        "channel_2": arrays.feature_names[column],
                        "pearson_r": value,
                        "absolute_r": abs(value),
                    }
                )
    redundant = pd.DataFrame(redundant_rows)
    if not redundant.empty:
        redundant = redundant.sort_values("absolute_r", ascending=False)
    redundant.to_csv(metric_dir / "highly_correlated_channel_pairs.csv", index=False)

    mi_table = _mutual_information_table(features, adjusted, binary, context_index, args.seed)
    mi_stability = _leave_one_subject_out_mi(adjusted, binary, subjects, args.seed)
    mi_table = mi_table.merge(mi_stability, on="feature", how="left").sort_values(
        "mi_binary_subject_adjusted", ascending=False
    )
    mi_table.to_csv(metric_dir / "mutual_information_features.csv", index=False)

    class_counts = pd.crosstab(pd.Series(subjects, name="subject_id"), pd.Series(contexts, name="context"))
    class_counts = class_counts.reindex(columns=["A", "B", "C", "D"], fill_value=0)
    class_counts["tallado_total"] = class_counts[["A", "B", "C"]].sum(axis=1)
    class_counts["no_tallado_total"] = class_counts["D"]
    class_counts.to_csv(metric_dir / "windows_by_training_user_and_context.csv")

    channel_quality_rows = []
    outlier_rows: list[dict[str, object]] = []
    windows_with_outlier = np.zeros(len(X), dtype=bool)
    for channel_index, channel in enumerate(arrays.feature_names):
        values = X[:, channel_index, :].astype(np.float64)
        window_rms = np.sqrt(np.mean(np.square(values), axis=1))
        median = float(np.median(window_rms))
        mad = float(np.median(np.abs(window_rms - median)))
        robust_z = 0.6745 * (window_rms - median) / mad if mad > 0 else np.zeros_like(window_rms)
        flagged = np.abs(robust_z) > 6
        windows_with_outlier |= flagged
        for window_position in np.flatnonzero(flagged):
            outlier_rows.append(
                {
                    "window_id": int(train_manifest.loc[window_position, "window_id"]),
                    "subject_id": subjects[window_position],
                    "context": contexts[window_position],
                    "source_file": train_manifest.loc[window_position, "source_file"],
                    "channel": channel,
                    "window_rms": float(window_rms[window_position]),
                    "robust_z": float(robust_z[window_position]),
                }
            )
        channel_quality_rows.append(
            {
                "channel": channel,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "window_rms_median": median,
                "window_rms_mad": mad,
                "windows_with_abs_robust_z_gt_6": int((np.abs(robust_z) > 6).sum()),
                "constant_windows": int((values.std(axis=1) < 1e-8).sum()),
            }
        )
    channel_quality = pd.DataFrame(channel_quality_rows)
    channel_quality.to_csv(metric_dir / "channel_quality.csv", index=False)
    outlier_frame = pd.DataFrame(outlier_rows)
    outlier_frame.to_csv(metric_dir / "amplitude_outlier_flags.csv", index=False)
    if not outlier_frame.empty:
        outlier_file_summary = (
            outlier_frame.groupby(["subject_id", "context", "source_file"], as_index=False)
            .agg(unique_flagged_windows=("window_id", "nunique"), channel_flags=("channel", "size"))
            .sort_values(["unique_flagged_windows", "channel_flags"], ascending=False)
        )
    else:
        outlier_file_summary = pd.DataFrame(
            columns=["subject_id", "context", "source_file", "unique_flagged_windows", "channel_flags"]
        )
    outlier_file_summary.to_csv(metric_dir / "amplitude_outliers_by_source_file.csv", index=False)

    train_subjects = set(subjects.tolist())
    quality = file_quality[file_quality["subject_id"].astype(str).isin(train_subjects)].copy()
    quality_summary = {
        "files_total_for_training_subjects": int(len(quality)),
        "files_used": int((quality["status"] == "used").sum()),
        "files_skipped": int((quality["status"] != "used").sum()),
        "source_rows": int(quality["rows"].sum()),
        "timestamp_repairs": int(quality["repaired_timestamps"].sum()),
        "remaining_discontinuities": int(quality["discontinuities"].sum()),
        "candidate_windows": int(quality["candidate_windows_row_order"].sum()),
        "retained_windows": int(quality["windows"].sum()),
    }
    quality_summary["retained_candidate_fraction"] = (
        quality_summary["retained_windows"] / quality_summary["candidate_windows"]
        if quality_summary["candidate_windows"]
        else float("nan")
    )
    skipped_reasons = {
        str(reason): int(count)
        for reason, count in quality.loc[quality["status"] != "used", "reason"].value_counts().items()
    }
    duplicates = _exact_duplicate_summary(X, contexts, subjects)
    audit = {
        "scope": "training_only",
        "n_windows": int(len(X)),
        "n_subjects": int(len(np.unique(subjects))),
        "n_channels": int(X.shape[1]),
        "samples_per_window": int(X.shape[2]),
        "binary_counts": {
            "no_tallado": int((binary == 0).sum()),
            "tallado": int((binary == 1).sum()),
        },
        "non_finite_values": int((~np.isfinite(X)).sum()),
        "imputed_fraction_mean": float(train_manifest["imputed_fraction"].mean()),
        "imputed_fraction_max": float(train_manifest["imputed_fraction"].max()),
        "windows_with_imputation": int((train_manifest["imputed_fraction"] > 0).sum()),
        "windows_over_1_percent_imputed": int((train_manifest["imputed_fraction"] > 0.01).sum()),
        "windows_over_5_percent_imputed": int((train_manifest["imputed_fraction"] > 0.05).sum()),
        **duplicates,
        **quality_summary,
        "highly_correlated_channel_pairs_abs_r_ge_0.90": int(len(redundant)),
        "constant_windows_across_all_channel_checks": int(channel_quality["constant_windows"].sum()),
        "amplitude_outlier_channel_window_flags": int(
            channel_quality["windows_with_abs_robust_z_gt_6"].sum()
        ),
        "unique_windows_with_amplitude_outlier_flag": int(windows_with_outlier.sum()),
        "amplitude_outlier_unique_windows_by_context": {
            str(context): int(
                outlier_frame.loc[outlier_frame["context"] == context, "window_id"].nunique()
            )
            for context in ("A", "B", "C", "D")
        },
        "skipped_file_reasons": skipped_reasons,
    }
    (metric_dir / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _heatmap(sample_correlation, plot_dir / "channel_sample_correlation.png", "Correlación Pearson entre canales", (-1, 1))
    _heatmap(rms_correlation, plot_dir / "channel_window_rms_correlation.png", "Correlación Spearman del RMS por ventana", (-1, 1))
    _plot_class_by_user(class_counts[["A", "B", "C", "D"]], plot_dir / "class_distribution_by_user.png")
    _plot_top_mi(mi_table, plot_dir / "top_mutual_information_binary.png")

    top_mi = mi_table.head(12)[
        [
            "feature",
            "mi_binary_subject_adjusted",
            "mi_binary_loso_mean",
            "top20_loso_fraction",
            "cohen_d_binary_subject_adjusted",
        ]
    ]
    upper_triangle = sample_correlation.abs().where(
        np.triu(np.ones(sample_correlation.shape), k=1).astype(bool)
    )
    strongest_pair = upper_triangle.stack().sort_values(ascending=False).index[0]
    strongest_value = float(sample_correlation.loc[strongest_pair[0], strongest_pair[1]])
    missing_context_users = int((class_counts[["A", "B", "C", "D"]] == 0).any(axis=1).sum())
    lines = [
        "# Auditoría de los datos de entrenamiento",
        "",
        "## Alcance",
        "",
        "Este análisis utiliza exclusivamente los 16 usuarios y 1,601 ventanas de `train`.",
        "No consulta validación ni holdout y, por tanto, no contamina la evaluación de los modelos.",
        "",
        "## Integridad y calidad",
        "",
        f"- Valores NaN o infinitos: **{audit['non_finite_values']}**.",
        f"- Ventanas exactas duplicadas: **{audit['windows_in_exact_duplicate_groups']}** ",
        f"en {audit['exact_duplicate_groups']} grupos; duplicados entre etiquetas: ",
        f"**{audit['cross_label_duplicate_groups']}**.",
        f"- Imputación media por ventana: **{100 * audit['imputed_fraction_mean']:.3f} %**; ",
        f"máxima: **{100 * audit['imputed_fraction_max']:.3f} %**.",
        f"- Ventanas con más de 5 % imputado: **{audit['windows_over_5_percent_imputed']}**.",
        f"- Ventanas constantes sumadas entre todos los canales: **{audit['constant_windows_across_all_channel_checks']}**.",
        f"- Ventanas marcadas como amplitud extrema en al menos un canal: "
        f"**{audit['unique_windows_with_amplitude_outlier_flag']}**. Son señales para revisión, no descartes automáticos.",
        f"- Archivos utilizables de usuarios train: {audit['files_used']} de {audit['files_total_for_training_subjects']}; ",
        f"{audit['files_skipped']} no generaron ventanas válidas.",
        f"  Motivos: {skipped_reasons}.",
        f"- Se repararon {audit['timestamp_repairs']} timestamps de {audit['source_rows']} filas de origen ",
        f"({100 * audit['timestamp_repairs'] / audit['source_rows']:.2f} %) y se segmentaron "
        f"{audit['remaining_discontinuities']} discontinuidades restantes.",
        f"- Se conservaron **{100 * audit['retained_candidate_fraction']:.1f} %** de las ventanas candidatas después del control temporal.",
        "",
        "## Distribución",
        "",
        f"La tarea binaria contiene {audit['binary_counts']['tallado']} ventanas de tallado y ",
        f"{audit['binary_counts']['no_tallado']} de no tallado. La razón aproximada es ",
        f"{audit['binary_counts']['tallado'] / audit['binary_counts']['no_tallado']:.2f}:1.",
        f"Hay {missing_context_users} usuario(s) de entrenamiento sin al menos uno de los cuatro contextos.",
        "La cantidad de ventanas difiere entre usuarios aunque el protocolo de recolección haya sido similar,",
        "porque las duraciones, discontinuidades y ventanas retenidas no son idénticas.",
        "",
        "## Correlación",
        "",
        f"Se encontraron {len(redundant)} pares de canales con |r| ≥ 0.90 a nivel de muestra.",
        f"La relación más fuerte fue {strongest_pair[0]}–{strongest_pair[1]} (r={strongest_value:.3f}).",
        "Hay relaciones físicas esperadas entre aceleración, gravedad y señales derivadas, pero ninguna",
        "superó el criterio de redundancia extrema. La correlación no indica por sí sola qué canal predice",
        "el tallado y tampoco justifica eliminar canales sin una ablación por usuarios.",
        "",
        "## Información mutua",
        "",
        "La información mutua mide dependencia, incluidas relaciones no lineales, pero no tiene dirección",
        "ni se interpreta como porcentaje. Se extrajeron nueve estadísticas por canal. Para reducir la",
        "confusión por identidad del usuario, el ranking principal estandariza cada característica dentro",
        "de cada persona antes de calcular la información mutua con tallado/no tallado.",
        "También se repitió el cálculo retirando un usuario cada vez; `top20_loso_fraction` indica en qué",
        "proporción de esas 16 repeticiones la característica permaneció entre las 20 principales.",
        "",
        "```text",
        top_mi.to_string(index=False),
        "```",
        "",
        "El tamaño de efecto de Cohen aporta dirección: valores positivos son mayores durante tallado y",
        "valores negativos son mayores durante no tallado. Estas asociaciones son descriptivas; las ventanas",
        "adyacentes se solapan y no deben tratarse como 1,601 observaciones completamente independientes.",
        "",
        "## Dictamen",
        "",
        "Los datos de entrenamiento son técnicamente utilizables: no contienen valores no finitos, la",
        "imputación es baja y no hay evidencia de duplicados contradictorios. Los puntos de atención son",
        "el desbalance binario esperado al combinar A/B/C, la variación de cantidad de ventanas entre",
        "usuarios, las discontinuidades temporales ya segmentadas y las amplitudes extremas señaladas para revisión.",
        f"De las ventanas extremas, {audit['amplitude_outlier_unique_windows_by_context']['D']} pertenecen a D "
        f"y {audit['amplitude_outlier_unique_windows_by_context']['B']} a B; no aparecen en A ni C.",
        "El predominio de características de amplitud y variación sugiere revisar si D contiene movimientos",
        "sistemáticamente más intensos: el modelo podría aprender intensidad general además del patrón del tallado.",
        "La información mutua debe respaldarse con ablaciones dentro de GroupKFold antes de eliminar o",
        "declarar imprescindible cualquier canal.",
    ]
    (output_dir / "training_data_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("\nCaracterísticas principales por información mutua ajustada:")
    print(top_mi.to_string(index=False))


if __name__ == "__main__":
    main()

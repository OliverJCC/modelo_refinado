from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def _choose_balanced_subjects(
    manifest: pd.DataFrame,
    candidates: list[str],
    count: int,
    target_window_fraction: float,
) -> list[str]:
    if count <= 0 or count >= len(candidates):
        raise ValueError("El número de grupos reservado debe estar entre 1 y n_subjects-1")

    labels = sorted(manifest["label"].unique())
    global_counts = manifest["label"].value_counts().reindex(labels, fill_value=0).to_numpy(float)
    global_props = global_counts / global_counts.sum()
    total_windows = len(manifest)
    grouped = manifest.groupby(["subject_id", "label"]).size().unstack(fill_value=0)
    grouped = grouped.reindex(index=candidates, columns=labels, fill_value=0)

    best_score = float("inf")
    best_combo: tuple[str, ...] | None = None
    for combo in itertools.combinations(sorted(candidates), count):
        counts = grouped.loc[list(combo)].sum(axis=0).to_numpy(float)
        if np.any(counts == 0):
            continue
        sample_fraction = counts.sum() / total_windows
        class_props = counts / counts.sum()
        score = 5.0 * abs(sample_fraction - target_window_fraction)
        score += float(np.mean(np.abs(class_props - global_props)))
        if score < best_score - 1e-12:
            best_score = score
            best_combo = combo
    if best_combo is None:
        raise RuntimeError("No se encontró una combinación de usuarios que contenga todas las clases")
    return list(best_combo)


def create_group_splits(
    manifest_csv: str | Path,
    output_windows_csv: str | Path,
    output_subjects_csv: str | Path,
    output_summary_json: str | Path,
    test_subject_count: int = 4,
    validation_subject_count: int = 3,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    manifest = pd.read_csv(manifest_csv)
    subjects = sorted(manifest["subject_id"].unique().tolist())
    if test_subject_count + validation_subject_count >= len(subjects):
        raise ValueError("No quedarían usuarios para entrenamiento")

    test_subjects = _choose_balanced_subjects(
        manifest,
        subjects,
        test_subject_count,
        target_window_fraction=test_subject_count / len(subjects),
    )
    development_subjects = sorted(set(subjects).difference(test_subjects))
    development_manifest = manifest[manifest["subject_id"].isin(development_subjects)]
    validation_subjects = _choose_balanced_subjects(
        development_manifest,
        development_subjects,
        validation_subject_count,
        target_window_fraction=validation_subject_count / len(development_subjects),
    )
    train_subjects = sorted(set(development_subjects).difference(validation_subjects))

    split_map = {subject: "train" for subject in train_subjects}
    split_map.update({subject: "validation" for subject in validation_subjects})
    split_map.update({subject: "test" for subject in test_subjects})
    manifest["holdout_split"] = manifest["subject_id"].map(split_map)

    development_mask = manifest["holdout_split"] != "test"
    development = manifest.loc[development_mask]
    placeholder = np.zeros(len(development), dtype=np.float32)
    development_y = development["label_index"].to_numpy()
    development_groups = development["subject_id"].to_numpy()
    label_count = int(development["label_index"].nunique())
    target_subjects_per_fold = len(development_subjects) / cv_folds
    minimum_subjects_per_fold = len(development_subjects) // cv_folds
    maximum_subjects_per_fold = int(np.ceil(target_subjects_per_fold))
    target_windows_per_fold = len(development) / cv_folds
    global_class_props = (
        development["label_index"].value_counts(normalize=True).reindex(range(label_count), fill_value=0).to_numpy()
    )
    best_score = float("inf")
    best_fold = None
    selected_cv_seed = seed
    for candidate_seed in range(seed, seed + 500):
        candidate = np.full(len(development), -1, dtype=np.int32)
        splitter = GroupKFold(
            n_splits=cv_folds, shuffle=True, random_state=candidate_seed
        )
        valid = True
        score = 0.0
        for fold, (_, validation_indices) in enumerate(
            splitter.split(placeholder, development_y, groups=development_groups)
        ):
            fold_groups = np.unique(development_groups[validation_indices])
            fold_counts = np.bincount(development_y[validation_indices], minlength=label_count)
            if (
                np.any(fold_counts == 0)
                or len(fold_groups) < minimum_subjects_per_fold
                or len(fold_groups) > maximum_subjects_per_fold
            ):
                valid = False
                break
            fold_props = fold_counts / fold_counts.sum()
            score += 3.0 * abs(len(fold_groups) - target_subjects_per_fold)
            score += abs(len(validation_indices) - target_windows_per_fold) / target_windows_per_fold
            score += float(np.mean(np.abs(fold_props - global_class_props)))
            candidate[validation_indices] = fold
        if valid and np.all(candidate >= 0) and score < best_score:
            best_score = score
            best_fold = candidate
            selected_cv_seed = candidate_seed
    if best_fold is None:
        raise RuntimeError("GroupKFold no pudo construir folds válidos")
    cv_fold = np.full(len(manifest), -1, dtype=np.int32)
    cv_fold[development.index.to_numpy()] = best_fold
    manifest["cv_validation_fold"] = cv_fold

    if manifest.loc[development_mask, "cv_validation_fold"].lt(0).any():
        raise AssertionError("Alguna ventana de desarrollo no recibió fold")
    if manifest.loc[~development_mask, "cv_validation_fold"].ne(-1).any():
        raise AssertionError("El test fijo apareció dentro de K-Fold")

    output_windows_csv = Path(output_windows_csv)
    output_subjects_csv = Path(output_subjects_csv)
    output_summary_json = Path(output_summary_json)
    output_windows_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_windows_csv, index=False)

    subject_frame = (
        manifest.groupby(["subject_id", "holdout_split", "label"], as_index=False)
        .size()
        .pivot_table(
            index=["subject_id", "holdout_split"],
            columns="label",
            values="size",
            fill_value=0,
        )
        .reset_index()
    )
    subject_frame.columns.name = None
    subject_frame.to_csv(output_subjects_csv, index=False)

    train_set, val_set, test_set = map(set, (train_subjects, validation_subjects, test_subjects))
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise AssertionError("Fuga de usuarios detectada en holdout")

    summary = {
        "seed": seed,
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "test_subjects": test_subjects,
        "counts_by_split": manifest["holdout_split"].value_counts().to_dict(),
        "subjects_by_split": subject_frame["holdout_split"].value_counts().to_dict(),
        "cv_folds": cv_folds,
        "cv_random_state_selected": selected_cv_seed,
        "cv_validation_subjects": {
            str(fold): sorted(
                manifest.loc[manifest["cv_validation_fold"] == fold, "subject_id"].unique().tolist()
            )
            for fold in range(cv_folds)
        },
    }
    with output_summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary

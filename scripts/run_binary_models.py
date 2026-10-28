from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.binary import (
    BINARY_CLASS_NAMES,
    binary_classification_metrics,
    binary_metrics_by_group,
    binary_targets,
    context_sensitivity_table,
    event_detection_metrics,
    select_operating_threshold,
)
from evaluation.plotting import (
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_training_history,
)
from models import MODEL_NAMES, build_model
from smartwatch_pipeline.config import load_config
from smartwatch_pipeline.data import ChannelStandardizer, load_dataset
from training.engine import class_weights, make_loader, predict, seed_everything, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K-Fold binario A/B/C=tallado frente a D=no tallado; no usa holdout test."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--models", default="all")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--minimum-sensitivity", type=float, default=0.95)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero no está disponible")
    return torch.device(name)


def _inner_split(indices: np.ndarray, subjects: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    validation_groups = max(2, round(0.2 * len(np.unique(subjects[indices]))))
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_groups, random_state=seed)
    fit_relative, validation_relative = next(splitter.split(indices, groups=subjects[indices]))
    return indices[fit_relative], indices[validation_relative]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_config = dict(config["training"])
    if args.max_epochs is not None:
        training_config["max_epochs"] = args.max_epochs
    if args.learning_rate is not None:
        training_config["learning_rate"] = args.learning_rate
    models = list(MODEL_NAMES) if args.models == "all" else [x.strip() for x in args.models.split(",")]
    unknown = sorted(set(models).difference(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Modelos desconocidos: {unknown}")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    device = _device(args.device)

    arrays = load_dataset(PROJECT_ROOT / "data" / "processed" / "smartwatch_windows.npz")
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "windows_with_splits.csv")
    if not np.array_equal(arrays.window_id, manifest["window_id"].to_numpy()):
        raise AssertionError("NPZ y manifiesto no están alineados")
    contexts = manifest["label"].astype(str).to_numpy()
    y_binary = binary_targets(contexts)

    cv_folds = sorted(
        manifest.loc[manifest["cv_validation_fold"] >= 0, "cv_validation_fold"].unique()
    )
    if args.dry_run:
        report: dict[str, object] = {
            "status": "OK",
            "task": "binary_tallado_vs_no_tallado",
            "mapping": {"A": 1, "B": 1, "C": 1, "D": 0},
            "device": str(device),
            "models": {},
            "folds": {},
        }
        for model_name in models:
            seed_everything(seeds[0])
            model = build_model(
                model_name, arrays.X.shape[1], arrays.X.shape[2], 2, arrays.sampling_rate_hz
            ).to(device)
            model.eval()
            with torch.no_grad():
                output = model(torch.zeros(2, *arrays.X.shape[1:], device=device))
            report["models"][model_name] = {
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "output_shape": list(output.shape),
            }
        for fold_value in cv_folds:
            fold = int(fold_value)
            evaluation = manifest["cv_validation_fold"].to_numpy() == fold
            training = (manifest["holdout_split"].to_numpy() != "test") & ~evaluation
            fit, early = _inner_split(np.flatnonzero(training), arrays.subject_id, seeds[0] + fold)
            report["folds"][str(fold)] = {
                "fit_subjects": len(np.unique(arrays.subject_id[fit])),
                "early_subjects": len(np.unique(arrays.subject_id[early])),
                "evaluation_subjects": len(np.unique(arrays.subject_id[evaluation])),
                "fit_binary_counts": np.bincount(y_binary[fit], minlength=2).tolist(),
            }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_binary_cv")
    result_dir = PROJECT_ROOT / "results" / run_id
    metric_dir = PROJECT_ROOT / "metrics" / run_id
    plot_dir = PROJECT_ROOT / "plots" / run_id
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / run_id
    for directory in (result_dir, metric_dir, plot_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    effective_config = dict(config)
    effective_config["training"] = training_config
    (result_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "task": "binary_tallado_vs_no_tallado",
                "mapping": {"A": 1, "B": 1, "C": 1, "D": 0},
                "models": models,
                "seeds": seeds,
                "device": str(device),
                "minimum_sensitivity": args.minimum_sensitivity,
                "event_minimum_consecutive_windows": 2,
                **effective_config,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    fold_rows: list[dict[str, object]] = []
    user_frames: list[pd.DataFrame] = []
    context_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    event_summary_rows: list[dict[str, object]] = []
    event_frames: list[pd.DataFrame] = []
    id_to_position = {int(identifier): position for position, identifier in enumerate(arrays.window_id)}

    for model_name in models:
        for seed in seeds:
            for fold_value in cv_folds:
                fold = int(fold_value)
                outer_validation = manifest["cv_validation_fold"].to_numpy() == fold
                outer_training = (
                    (manifest["holdout_split"].to_numpy() != "test") & ~outer_validation
                )
                outer_train_indices = np.flatnonzero(outer_training)
                evaluation_indices = np.flatnonzero(outer_validation)
                fit_indices, early_indices = _inner_split(
                    outer_train_indices, arrays.subject_id, seed + fold
                )
                subject_sets = [
                    set(arrays.subject_id[index])
                    for index in (fit_indices, early_indices, evaluation_indices)
                ]
                if subject_sets[0] & subject_sets[1] or subject_sets[0] & subject_sets[2] or subject_sets[1] & subject_sets[2]:
                    raise AssertionError("Fuga de usuarios dentro del fold binario")

                scaler = ChannelStandardizer.fit(arrays.X[fit_indices])
                X_fit = scaler.transform(arrays.X[fit_indices])
                X_early = scaler.transform(arrays.X[early_indices])
                X_evaluation = scaler.transform(arrays.X[evaluation_indices])
                seed_everything(seed)
                model = build_model(
                    model_name,
                    arrays.X.shape[1],
                    arrays.X.shape[2],
                    2,
                    arrays.sampling_rate_hz,
                )
                stem = f"{model_name}_seed{seed}_fold{fold}"
                checkpoint_path = checkpoint_dir / f"{stem}.pth"
                scaler.save(checkpoint_dir / f"scaler_{stem}.npz")
                weights = class_weights(y_binary[fit_indices], 2)
                print(
                    f"\n[START-BINARY] model={model_name} seed={seed} fold={fold} "
                    f"fit={len(fit_indices)} early={len(early_indices)} "
                    f"evaluation={len(evaluation_indices)} weights={weights.tolist()} device={device}",
                    flush=True,
                )
                history, best_epoch, best_inner_macro_f1 = train_model(
                    model=model,
                    X_train=X_fit,
                    y_train=y_binary[fit_indices],
                    train_ids=arrays.window_id[fit_indices],
                    X_validation=X_early,
                    y_validation=y_binary[early_indices],
                    validation_ids=arrays.window_id[early_indices],
                    device=device,
                    checkpoint_path=checkpoint_path,
                    seed=seed,
                    batch_size=int(training_config["batch_size"]),
                    max_epochs=int(training_config["max_epochs"]),
                    learning_rate=float(training_config["learning_rate"]),
                    weight_decay=float(training_config["weight_decay"]),
                    early_stopping_patience=int(training_config["early_stopping_patience"]),
                    scheduler_patience=int(training_config["scheduler_patience"]),
                    scheduler_factor=float(training_config["scheduler_factor"]),
                    weighted_loss=True,
                    n_classes=2,
                )
                history.insert(0, "model", model_name)
                history.insert(1, "seed", seed)
                history.insert(2, "fold", fold)
                history.to_csv(metric_dir / f"history_{stem}.csv", index=False)
                plot_training_history(history, plot_dir / f"history_{stem}.png")

                criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
                early_loader = make_loader(
                    X_early,
                    y_binary[early_indices],
                    arrays.window_id[early_indices],
                    int(training_config["batch_size"]),
                    False,
                    seed,
                )
                inner_bundle = predict(model.to(device), early_loader, criterion, device)
                threshold, threshold_table = select_operating_threshold(
                    inner_bundle.y_true,
                    inner_bundle.probabilities[:, 1],
                    args.minimum_sensitivity,
                )
                threshold_table.to_csv(metric_dir / f"threshold_search_{stem}.csv", index=False)
                selected_inner = threshold_table[threshold_table["selected"]].iloc[0]

                evaluation_loader = make_loader(
                    X_evaluation,
                    y_binary[evaluation_indices],
                    arrays.window_id[evaluation_indices],
                    int(training_config["batch_size"]),
                    False,
                    seed,
                )
                bundle = predict(model.to(device), evaluation_loader, criterion, device)
                aggregate, y_pred, matrix = binary_classification_metrics(
                    bundle.y_true, bundle.probabilities[:, 1], threshold
                )
                fold_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "n_windows": len(bundle.y_true),
                        "n_evaluation_subjects": len(subject_sets[2]),
                        "best_epoch": best_epoch,
                        "best_inner_f1_macro_at_0.5": best_inner_macro_f1,
                        "selected_threshold": threshold,
                        "inner_sensitivity_at_selected_threshold": selected_inner["sensitivity_recall_tallado"],
                        "inner_specificity_at_selected_threshold": selected_inner["specificity_no_tallado"],
                        "evaluation_loss": bundle.loss,
                        "weight_no_tallado": float(weights[0]),
                        "weight_tallado": float(weights[1]),
                        **aggregate,
                    }
                )

                positions = np.asarray([id_to_position[int(identifier)] for identifier in bundle.window_id])
                probabilities = bundle.probabilities[:, 1]
                user_frame = binary_metrics_by_group(
                    bundle.y_true,
                    probabilities,
                    arrays.subject_id[positions],
                    threshold,
                    "subject_id",
                )
                user_frame.insert(0, "model", model_name)
                user_frame.insert(1, "seed", seed)
                user_frame.insert(2, "fold", fold)
                user_frames.append(user_frame)

                context_frame = context_sensitivity_table(contexts[positions], y_pred)
                context_frame.insert(0, "model", model_name)
                context_frame.insert(1, "seed", seed)
                context_frame.insert(2, "fold", fold)
                context_frames.append(context_frame)

                selected_manifest = manifest.iloc[positions].reset_index(drop=True)
                prediction = pd.DataFrame(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "window_id": bundle.window_id,
                        "subject_id": arrays.subject_id[positions],
                        "context": contexts[positions],
                        "source_file": selected_manifest["source_file"],
                        "segment_id": selected_manifest["segment_id"],
                        "start_row": selected_manifest["start_row"],
                        "end_row": selected_manifest["end_row"],
                        "start_timestamp_s": selected_manifest["start_timestamp_s"],
                        "end_timestamp_s": selected_manifest["end_timestamp_s"],
                        "y_true": bundle.y_true,
                        "y_pred": y_pred,
                        "probability_no_tallado": bundle.probabilities[:, 0],
                        "probability_tallado": probabilities,
                        "selected_threshold": threshold,
                    }
                )
                prediction_frames.append(prediction)
                event_aggregate, event_frame = event_detection_metrics(
                    prediction,
                    arrays.sampling_rate_hz,
                    int(arrays.X.shape[2]),
                    int(config["data"]["step_size"]),
                    minimum_consecutive_windows=2,
                )
                event_summary_rows.append(
                    {"model": model_name, "seed": seed, "fold": fold, **event_aggregate}
                )
                event_frame.insert(0, "model", model_name)
                event_frame.insert(1, "seed", seed)
                event_frame.insert(2, "fold", fold)
                event_frames.append(event_frame)

                np.savetxt(metric_dir / f"confusion_{stem}.csv", matrix, delimiter=",", fmt="%d")
                plot_confusion_matrix(matrix, list(BINARY_CLASS_NAMES), plot_dir / f"confusion_{stem}.png")
                plot_confusion_matrix(
                    matrix,
                    list(BINARY_CLASS_NAMES),
                    plot_dir / f"confusion_{stem}_normalized.png",
                    normalize=True,
                )
                plot_roc_pr_curves(
                    bundle.y_true,
                    bundle.probabilities,
                    list(BINARY_CLASS_NAMES),
                    plot_dir / f"roc_{stem}.png",
                    plot_dir / f"pr_{stem}.png",
                )
                print(
                    f"[DONE-BINARY] model={model_name} seed={seed} fold={fold} "
                    f"threshold={threshold:.2f} sensitivity={aggregate['sensitivity_recall_tallado']:.4f} "
                    f"specificity={aggregate['specificity_no_tallado']:.4f} "
                    f"balanced_accuracy={aggregate['balanced_accuracy']:.4f}",
                    flush=True,
                )

    pd.DataFrame(fold_rows).to_csv(metric_dir / "fold_metrics.csv", index=False)
    pd.concat(user_frames, ignore_index=True).to_csv(metric_dir / "user_metrics.csv", index=False)
    pd.concat(context_frames, ignore_index=True).to_csv(metric_dir / "context_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(metric_dir / "predictions.csv", index=False)
    pd.DataFrame(event_summary_rows).to_csv(metric_dir / "event_summary.csv", index=False)
    pd.concat(event_frames, ignore_index=True).to_csv(metric_dir / "event_details.csv", index=False)
    print(f"K-Fold binario completado. Resultados: {result_dir}")


if __name__ == "__main__":
    main()

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

from evaluation.metrics import classification_metrics, metrics_by_user
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
        description="K-Fold de modelos base. No evalúa el test fijo."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--models", default="all")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida folds y forward pass sin entrenar ni escribir resultados.",
    )
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero torch.cuda.is_available() es False")
    return torch.device(name)


def _inner_split(indices: np.ndarray, subjects: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_subjects = np.unique(subjects[indices])
    validation_groups = max(2, round(0.2 * len(unique_subjects)))
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_groups, random_state=seed)
    fit_relative, validation_relative = next(
        splitter.split(indices, groups=subjects[indices])
    )
    return indices[fit_relative], indices[validation_relative]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_config = dict(config["training"])
    if args.max_epochs is not None:
        training_config["max_epochs"] = args.max_epochs
    if args.learning_rate is not None:
        training_config["learning_rate"] = args.learning_rate
    models = list(MODEL_NAMES) if args.models == "all" else args.models.split(",")
    unknown = sorted(set(models).difference(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Modelos desconocidos: {unknown}")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    device = _device(args.device)

    dataset_path = PROJECT_ROOT / "data" / "processed" / "smartwatch_windows.npz"
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "windows_with_splits.csv"
    arrays = load_dataset(dataset_path)
    manifest = pd.read_csv(manifest_path)
    if not np.array_equal(arrays.window_id, manifest["window_id"].to_numpy()):
        raise AssertionError("NPZ y manifiesto no están alineados")

    if args.dry_run:
        report: dict[str, object] = {
            "status": "OK",
            "device": str(device),
            "models": {},
            "folds": {},
        }
        for model_name in models:
            seed_everything(seeds[0])
            model = build_model(
                model_name,
                arrays.X.shape[1],
                arrays.X.shape[2],
                len(arrays.label_names),
                arrays.sampling_rate_hz,
            ).to(device)
            model.eval()
            with torch.no_grad():
                output = model(torch.zeros(2, *arrays.X.shape[1:], device=device))
            report["models"][model_name] = {
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "output_shape": list(output.shape),
            }
        for fold_value in sorted(
            manifest.loc[manifest["cv_validation_fold"] >= 0, "cv_validation_fold"].unique()
        ):
            fold = int(fold_value)
            outer_validation = manifest["cv_validation_fold"].to_numpy() == fold
            outer_training = (
                (manifest["holdout_split"].to_numpy() != "test") & ~outer_validation
            )
            outer_train_indices = np.flatnonzero(outer_training)
            evaluation_indices = np.flatnonzero(outer_validation)
            fit_indices, early_indices = _inner_split(
                outer_train_indices, arrays.subject_id, seeds[0] + fold
            )
            sets = [
                set(arrays.subject_id[fit_indices]),
                set(arrays.subject_id[early_indices]),
                set(arrays.subject_id[evaluation_indices]),
            ]
            if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
                raise AssertionError("Fuga detectada durante dry-run")
            report["folds"][str(fold)] = {
                "fit_users": len(sets[0]),
                "early_stopping_users": len(sets[1]),
                "evaluation_users": len(sets[2]),
                "fit_windows": len(fit_indices),
                "early_stopping_windows": len(early_indices),
                "evaluation_windows": len(evaluation_indices),
            }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_baselines")
    result_dir = PROJECT_ROOT / "results" / run_id
    metric_dir = PROJECT_ROOT / "metrics" / run_id
    plot_dir = PROJECT_ROOT / "plots" / run_id
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / run_id
    for directory in (result_dir, metric_dir, plot_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    effective_config = dict(config)
    effective_config["training"] = training_config
    with (result_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": run_id,
                "models": models,
                "seeds": seeds,
                "device": str(device),
                **effective_config,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    fold_rows: list[dict[str, object]] = []
    class_frames: list[pd.DataFrame] = []
    user_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    cv_folds = sorted(manifest.loc[manifest["cv_validation_fold"] >= 0, "cv_validation_fold"].unique())

    for model_name in models:
        for seed in seeds:
            for fold_value in cv_folds:
                fold = int(fold_value)
                outer_validation = manifest["cv_validation_fold"].to_numpy() == fold
                outer_training = (
                    (manifest["holdout_split"].to_numpy() != "test")
                    & ~outer_validation
                )
                outer_train_indices = np.flatnonzero(outer_training)
                evaluation_indices = np.flatnonzero(outer_validation)
                fit_indices, early_stop_indices = _inner_split(
                    outer_train_indices, arrays.subject_id, seed + fold
                )
                fit_subjects = set(arrays.subject_id[fit_indices])
                early_subjects = set(arrays.subject_id[early_stop_indices])
                evaluation_subjects = set(arrays.subject_id[evaluation_indices])
                if fit_subjects & early_subjects or fit_subjects & evaluation_subjects or early_subjects & evaluation_subjects:
                    raise AssertionError("Fuga de usuarios dentro del fold")

                scaler = ChannelStandardizer.fit(arrays.X[fit_indices])
                X_fit = scaler.transform(arrays.X[fit_indices])
                X_early = scaler.transform(arrays.X[early_stop_indices])
                X_evaluation = scaler.transform(arrays.X[evaluation_indices])
                seed_everything(seed)
                model = build_model(
                    model_name,
                    n_channels=arrays.X.shape[1],
                    n_times=arrays.X.shape[2],
                    n_classes=len(arrays.label_names),
                    sampling_rate_hz=arrays.sampling_rate_hz,
                )
                stem = f"{model_name}_seed{seed}_fold{fold}"
                checkpoint_path = checkpoint_dir / f"{stem}.pth"
                scaler.save(checkpoint_dir / f"scaler_{stem}.npz")
                print(
                    f"\n[START] model={model_name} seed={seed} fold={fold} "
                    f"fit={len(fit_indices)} early={len(early_stop_indices)} "
                    f"evaluation={len(evaluation_indices)} device={device}",
                    flush=True,
                )
                history, best_epoch, best_inner_f1 = train_model(
                    model=model,
                    X_train=X_fit,
                    y_train=arrays.y[fit_indices],
                    train_ids=arrays.window_id[fit_indices],
                    X_validation=X_early,
                    y_validation=arrays.y[early_stop_indices],
                    validation_ids=arrays.window_id[early_stop_indices],
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
                    weighted_loss=bool(training_config["class_weighted_loss"]),
                    n_classes=len(arrays.label_names),
                )
                history.insert(0, "model", model_name)
                history.insert(1, "seed", seed)
                history.insert(2, "fold", fold)
                history.to_csv(metric_dir / f"history_{stem}.csv", index=False)
                plot_training_history(history, plot_dir / f"history_{stem}.png")

                criterion = torch.nn.CrossEntropyLoss(
                    weight=class_weights(arrays.y[fit_indices], len(arrays.label_names)).to(device)
                    if training_config["class_weighted_loss"]
                    else None
                )
                evaluation_loader = make_loader(
                    X_evaluation,
                    arrays.y[evaluation_indices],
                    arrays.window_id[evaluation_indices],
                    int(training_config["batch_size"]),
                    False,
                    seed,
                )
                bundle = predict(model.to(device), evaluation_loader, criterion, device)
                aggregate, class_frame, matrix = classification_metrics(
                    bundle.y_true, bundle.y_pred, bundle.probabilities, arrays.label_names
                )
                fold_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "n_windows": len(bundle.y_true),
                        "n_evaluation_subjects": len(evaluation_subjects),
                        "best_epoch": best_epoch,
                        "best_inner_f1_macro": best_inner_f1,
                        "evaluation_loss": bundle.loss,
                        **aggregate,
                    }
                )
                class_frame.insert(0, "model", model_name)
                class_frame.insert(1, "seed", seed)
                class_frame.insert(2, "fold", fold)
                class_frames.append(class_frame)
                id_to_position = {int(identifier): position for position, identifier in enumerate(arrays.window_id)}
                positions = np.asarray([id_to_position[int(identifier)] for identifier in bundle.window_id])
                user_frame = metrics_by_user(
                    bundle.y_true,
                    bundle.y_pred,
                    bundle.probabilities,
                    arrays.subject_id[positions],
                    arrays.label_names,
                )
                user_frame.insert(0, "model", model_name)
                user_frame.insert(1, "seed", seed)
                user_frame.insert(2, "fold", fold)
                user_frames.append(user_frame)
                prediction = pd.DataFrame(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "window_id": bundle.window_id,
                        "subject_id": arrays.subject_id[positions],
                        "y_true": bundle.y_true,
                        "y_pred": bundle.y_pred,
                    }
                )
                for class_index, class_name in enumerate(arrays.label_names):
                    prediction[f"probability_{class_name}"] = bundle.probabilities[:, class_index]
                prediction_frames.append(prediction)
                np.savetxt(metric_dir / f"confusion_{stem}.csv", matrix, delimiter=",", fmt="%d")
                plot_confusion_matrix(matrix, arrays.label_names.tolist(), plot_dir / f"confusion_{stem}.png")
                plot_confusion_matrix(
                    matrix,
                    arrays.label_names.tolist(),
                    plot_dir / f"confusion_{stem}_normalized.png",
                    normalize=True,
                )
                plot_roc_pr_curves(
                    bundle.y_true,
                    bundle.probabilities,
                    arrays.label_names.tolist(),
                    plot_dir / f"roc_{stem}.png",
                    plot_dir / f"pr_{stem}.png",
                )
                print(
                    f"[DONE] model={model_name} seed={seed} fold={fold} "
                    f"best_epoch={best_epoch} eval_f1_macro={aggregate['f1_macro']:.4f} "
                    f"eval_balanced_accuracy={aggregate['balanced_accuracy']:.4f}",
                    flush=True,
                )

    pd.DataFrame(fold_rows).to_csv(metric_dir / "fold_metrics.csv", index=False)
    pd.concat(class_frames, ignore_index=True).to_csv(metric_dir / "class_metrics.csv", index=False)
    pd.concat(user_frames, ignore_index=True).to_csv(metric_dir / "user_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(metric_dir / "predictions.csv", index=False)
    print(f"K-Fold completado. Resultados: {result_dir}")


if __name__ == "__main__":
    main()

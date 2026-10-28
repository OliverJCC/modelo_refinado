from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import classification_metrics, metrics_by_user
from evaluation.plotting import plot_confusion_matrix, plot_roc_pr_curves, plot_training_history
from models import MODEL_NAMES, build_model
from smartwatch_pipeline.config import load_config
from smartwatch_pipeline.data import ChannelStandardizer, load_dataset
from training.engine import class_weights, make_loader, predict, seed_everything, train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluación final holdout. Abre el test fijo y exige confirmación explícita."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--models", default="all")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--confirm-test-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.confirm_test_evaluation:
        raise RuntimeError(
            "Test bloqueado. Añade --confirm-test-evaluation solo después de congelar modelos."
        )

    config = load_config(args.config)
    training_config = dict(config["training"])
    if args.max_epochs is not None:
        training_config["max_epochs"] = args.max_epochs
    if args.learning_rate is not None:
        training_config["learning_rate"] = args.learning_rate
    models = list(MODEL_NAMES) if args.models == "all" else args.models.split(",")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if set(models).difference(MODEL_NAMES):
        raise ValueError(f"Modelos desconocidos: {sorted(set(models).difference(MODEL_NAMES))}")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible en el entorno actual")

    arrays = load_dataset(PROJECT_ROOT / "data" / "processed" / "smartwatch_windows.npz")
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "windows_with_splits.csv")
    train_indices = np.flatnonzero(manifest["holdout_split"].to_numpy() == "train")
    validation_indices = np.flatnonzero(manifest["holdout_split"].to_numpy() == "validation")
    test_indices = np.flatnonzero(manifest["holdout_split"].to_numpy() == "test")
    split_subjects = [set(arrays.subject_id[index]) for index in (train_indices, validation_indices, test_indices)]
    if split_subjects[0] & split_subjects[1] or split_subjects[0] & split_subjects[2] or split_subjects[1] & split_subjects[2]:
        raise AssertionError("Fuga de usuarios en holdout")

    scaler = ChannelStandardizer.fit(arrays.X[train_indices])
    X_train = scaler.transform(arrays.X[train_indices])
    X_validation = scaler.transform(arrays.X[validation_indices])
    X_test = scaler.transform(arrays.X[test_indices])
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_final")
    metric_dir = PROJECT_ROOT / "metrics" / run_id
    plot_dir = PROJECT_ROOT / "plots" / run_id
    result_dir = PROJECT_ROOT / "results" / run_id
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / run_id
    for directory in (metric_dir, plot_dir, result_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    effective_config = dict(config)
    effective_config["training"] = training_config
    (result_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "models": models,
                "seeds": seeds,
                "device": str(device),
                **effective_config,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    scaler.save(result_dir / "holdout_scaler.npz")
    (result_dir / "test_opened.json").write_text(
        json.dumps({"timestamp": datetime.now().isoformat(), "test_subjects": sorted(split_subjects[2])}, indent=2),
        encoding="utf-8",
    )

    metric_rows = []
    class_frames = []
    user_frames = []
    prediction_frames = []
    for model_name in models:
        for seed in seeds:
            seed_everything(seed)
            model = build_model(
                model_name,
                arrays.X.shape[1],
                arrays.X.shape[2],
                len(arrays.label_names),
                arrays.sampling_rate_hz,
            )
            stem = f"{model_name}_seed{seed}"
            history, best_epoch, best_validation_f1 = train_model(
                model,
                X_train,
                arrays.y[train_indices],
                arrays.window_id[train_indices],
                X_validation,
                arrays.y[validation_indices],
                arrays.window_id[validation_indices],
                device,
                checkpoint_dir / f"{stem}.pth",
                seed,
                int(training_config["batch_size"]),
                int(training_config["max_epochs"]),
                float(training_config["learning_rate"]),
                float(training_config["weight_decay"]),
                int(training_config["early_stopping_patience"]),
                int(training_config["scheduler_patience"]),
                float(training_config["scheduler_factor"]),
                bool(training_config["class_weighted_loss"]),
                len(arrays.label_names),
            )
            history.to_csv(metric_dir / f"history_{stem}.csv", index=False)
            plot_training_history(history, plot_dir / f"history_{stem}.png")
            criterion = torch.nn.CrossEntropyLoss(
                weight=class_weights(arrays.y[train_indices], len(arrays.label_names)).to(device)
                if training_config["class_weighted_loss"]
                else None
            )
            test_loader = make_loader(
                X_test,
                arrays.y[test_indices],
                arrays.window_id[test_indices],
                int(training_config["batch_size"]),
                False,
                seed,
            )
            bundle = predict(model.to(device), test_loader, criterion, device)
            aggregate, per_class, matrix = classification_metrics(
                bundle.y_true, bundle.y_pred, bundle.probabilities, arrays.label_names
            )
            metric_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation_f1_macro": best_validation_f1,
                    "test_loss": bundle.loss,
                    **aggregate,
                }
            )
            per_class.insert(0, "model", model_name)
            per_class.insert(1, "seed", seed)
            class_frames.append(per_class)
            id_to_position = {int(identifier): position for position, identifier in enumerate(arrays.window_id)}
            positions = np.asarray([id_to_position[int(identifier)] for identifier in bundle.window_id])
            per_user = metrics_by_user(
                bundle.y_true,
                bundle.y_pred,
                bundle.probabilities,
                arrays.subject_id[positions],
                arrays.label_names,
            )
            per_user.insert(0, "model", model_name)
            per_user.insert(1, "seed", seed)
            user_frames.append(per_user)
            predictions = pd.DataFrame(
                {
                    "model": model_name,
                    "seed": seed,
                    "window_id": bundle.window_id,
                    "subject_id": arrays.subject_id[positions],
                    "y_true": bundle.y_true,
                    "y_pred": bundle.y_pred,
                }
            )
            for class_index, class_name in enumerate(arrays.label_names):
                predictions[f"probability_{class_name}"] = bundle.probabilities[:, class_index]
            prediction_frames.append(predictions)
            np.savetxt(metric_dir / f"confusion_{stem}.csv", matrix, delimiter=",", fmt="%d")
            plot_confusion_matrix(matrix, arrays.label_names.tolist(), plot_dir / f"confusion_{stem}.png")
            plot_roc_pr_curves(
                bundle.y_true,
                bundle.probabilities,
                arrays.label_names.tolist(),
                plot_dir / f"roc_{stem}.png",
                plot_dir / f"pr_{stem}.png",
            )
    pd.DataFrame(metric_rows).to_csv(metric_dir / "final_test_metrics.csv", index=False)
    pd.concat(class_frames, ignore_index=True).to_csv(metric_dir / "final_class_metrics.csv", index=False)
    pd.concat(user_frames, ignore_index=True).to_csv(metric_dir / "final_user_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(metric_dir / "final_predictions.csv", index=False)
    print(f"Evaluación final guardada en: {result_dir}")


if __name__ == "__main__":
    main()

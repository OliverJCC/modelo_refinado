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

from evaluation.binary import (
    BINARY_CLASS_NAMES,
    binary_classification_metrics,
    binary_metrics_by_group,
    binary_targets,
    context_sensitivity_table,
    event_detection_metrics,
    select_operating_threshold,
)
from evaluation.plotting import plot_confusion_matrix, plot_roc_pr_curves, plot_training_history
from models import MODEL_NAMES, build_model
from smartwatch_pipeline.config import load_config
from smartwatch_pipeline.data import ChannelStandardizer, load_dataset
from training.engine import class_weights, make_loader, predict, seed_everything, train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluación holdout exploratoria binaria A/B/C frente a D."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--models", default="all")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--minimum-sensitivity", type=float, default=0.95)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--confirm-exploratory-holdout", action="store_true")
    args = parser.parse_args()
    if not args.confirm_exploratory_holdout:
        raise RuntimeError(
            "Holdout bloqueado. Añade --confirm-exploratory-holdout para reconocer que ya no es confirmatorio."
        )

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
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible")

    arrays = load_dataset(PROJECT_ROOT / "data" / "processed" / "smartwatch_windows.npz")
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "windows_with_splits.csv")
    if not np.array_equal(arrays.window_id, manifest["window_id"].to_numpy()):
        raise AssertionError("NPZ y manifiesto no están alineados")
    contexts = manifest["label"].astype(str).to_numpy()
    y_binary = binary_targets(contexts)
    split = manifest["holdout_split"].astype(str).to_numpy()
    train_indices = np.flatnonzero(split == "train")
    validation_indices = np.flatnonzero(split == "validation")
    test_indices = np.flatnonzero(split == "test")
    subject_sets = [
        set(arrays.subject_id[index])
        for index in (train_indices, validation_indices, test_indices)
    ]
    if subject_sets[0] & subject_sets[1] or subject_sets[0] & subject_sets[2] or subject_sets[1] & subject_sets[2]:
        raise AssertionError("Fuga de usuarios en el holdout")

    scaler = ChannelStandardizer.fit(arrays.X[train_indices])
    X_train = scaler.transform(arrays.X[train_indices])
    X_validation = scaler.transform(arrays.X[validation_indices])
    X_test = scaler.transform(arrays.X[test_indices])
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_binary_holdout_exploratory")
    metric_dir = PROJECT_ROOT / "metrics" / run_id
    plot_dir = PROJECT_ROOT / "plots" / run_id
    result_dir = PROJECT_ROOT / "results" / run_id
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / run_id
    for directory in (metric_dir, plot_dir, result_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    scaler.save(result_dir / "holdout_scaler.npz")
    (result_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "task": "binary_tallado_vs_no_tallado",
                "evaluation_status": "exploratory_not_confirmatory",
                "mapping": {"A": 1, "B": 1, "C": 1, "D": 0},
                "models": models,
                "seeds": seeds,
                "device": str(device),
                "minimum_sensitivity": args.minimum_sensitivity,
                "train_subjects": sorted(subject_sets[0]),
                "validation_subjects": sorted(subject_sets[1]),
                "test_subjects": sorted(subject_sets[2]),
                "training": training_config,
                "data": config["data"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    metric_rows: list[dict[str, object]] = []
    user_frames: list[pd.DataFrame] = []
    context_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    event_rows: list[dict[str, object]] = []
    event_frames: list[pd.DataFrame] = []
    id_to_position = {int(identifier): position for position, identifier in enumerate(arrays.window_id)}

    for model_name in models:
        for seed in seeds:
            seed_everything(seed)
            model = build_model(
                model_name, arrays.X.shape[1], arrays.X.shape[2], 2, arrays.sampling_rate_hz
            )
            stem = f"{model_name}_seed{seed}"
            checkpoint_path = checkpoint_dir / f"{stem}.pth"
            weights = class_weights(y_binary[train_indices], 2)
            print(
                f"\n[START-HOLDOUT-BINARY] model={model_name} seed={seed} "
                f"train={len(train_indices)} validation={len(validation_indices)} "
                f"test={len(test_indices)} device={device}",
                flush=True,
            )
            history, best_epoch, best_validation_f1 = train_model(
                model=model,
                X_train=X_train,
                y_train=y_binary[train_indices],
                train_ids=arrays.window_id[train_indices],
                X_validation=X_validation,
                y_validation=y_binary[validation_indices],
                validation_ids=arrays.window_id[validation_indices],
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
            history.to_csv(metric_dir / f"history_{stem}.csv", index=False)
            plot_training_history(history, plot_dir / f"history_{stem}.png")

            criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
            validation_loader = make_loader(
                X_validation,
                y_binary[validation_indices],
                arrays.window_id[validation_indices],
                int(training_config["batch_size"]),
                False,
                seed,
            )
            validation_bundle = predict(model.to(device), validation_loader, criterion, device)
            threshold, threshold_table = select_operating_threshold(
                validation_bundle.y_true,
                validation_bundle.probabilities[:, 1],
                args.minimum_sensitivity,
            )
            threshold_table.to_csv(metric_dir / f"threshold_search_{stem}.csv", index=False)
            selected_validation = threshold_table[threshold_table["selected"]].iloc[0]

            test_loader = make_loader(
                X_test,
                y_binary[test_indices],
                arrays.window_id[test_indices],
                int(training_config["batch_size"]),
                False,
                seed,
            )
            bundle = predict(model.to(device), test_loader, criterion, device)
            aggregate, y_pred, matrix = binary_classification_metrics(
                bundle.y_true, bundle.probabilities[:, 1], threshold
            )
            metric_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "n_windows": int(len(bundle.y_true)),
                    "n_test_subjects": int(len(subject_sets[2])),
                    "best_epoch": best_epoch,
                    "best_validation_f1_macro_at_0.5": best_validation_f1,
                    "selected_threshold": threshold,
                    "validation_sensitivity_at_selected_threshold": selected_validation["sensitivity_recall_tallado"],
                    "validation_specificity_at_selected_threshold": selected_validation["specificity_no_tallado"],
                    "test_loss": bundle.loss,
                    "weight_no_tallado": float(weights[0]),
                    "weight_tallado": float(weights[1]),
                    **aggregate,
                }
            )
            positions = np.asarray([id_to_position[int(identifier)] for identifier in bundle.window_id])
            probability = bundle.probabilities[:, 1]
            user_frame = binary_metrics_by_group(
                bundle.y_true, probability, arrays.subject_id[positions], threshold, "subject_id"
            )
            user_frame.insert(0, "model", model_name)
            user_frame.insert(1, "seed", seed)
            user_frames.append(user_frame)
            context_frame = context_sensitivity_table(contexts[positions], y_pred)
            context_frame.insert(0, "model", model_name)
            context_frame.insert(1, "seed", seed)
            context_frames.append(context_frame)

            selected_manifest = manifest.iloc[positions].reset_index(drop=True)
            prediction = pd.DataFrame(
                {
                    "model": model_name,
                    "seed": seed,
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
                    "probability_tallado": probability,
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
            event_rows.append({"model": model_name, "seed": seed, **event_aggregate})
            event_frame.insert(0, "model", model_name)
            event_frame.insert(1, "seed", seed)
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
                f"[DONE-HOLDOUT-BINARY] model={model_name} threshold={threshold:.2f} "
                f"sensitivity={aggregate['sensitivity_recall_tallado']:.4f} "
                f"specificity={aggregate['specificity_no_tallado']:.4f} "
                f"balanced_accuracy={aggregate['balanced_accuracy']:.4f}",
                flush=True,
            )

    pd.DataFrame(metric_rows).to_csv(metric_dir / "exploratory_holdout_metrics.csv", index=False)
    pd.concat(user_frames, ignore_index=True).to_csv(metric_dir / "exploratory_user_metrics.csv", index=False)
    pd.concat(context_frames, ignore_index=True).to_csv(metric_dir / "exploratory_context_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(metric_dir / "exploratory_predictions.csv", index=False)
    pd.DataFrame(event_rows).to_csv(metric_dir / "exploratory_event_summary.csv", index=False)
    pd.concat(event_frames, ignore_index=True).to_csv(metric_dir / "exploratory_event_details.csv", index=False)
    print(f"Evaluación binaria exploratoria completada: {result_dir}")


if __name__ == "__main__":
    main()

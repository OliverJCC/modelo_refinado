from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int, deterministic: bool = True) -> None:
    if deterministic:
        # Required by CUDA/cuBLAS for deterministic matrix multiplications.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    window_ids: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
        torch.from_numpy(window_ids).long(),
    )
    generator = torch.Generator().manual_seed(seed)
    # PaperLSTM uses BatchNorm and cannot train on a batch containing one item.
    drop_last = bool(shuffle and len(dataset) > batch_size and len(dataset) % batch_size == 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        worker_init_fn=_seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.divide(
        len(y),
        n_classes * counts,
        out=np.zeros_like(counts),
        where=counts > 0,
    )
    return torch.tensor(weights, dtype=torch.float32)


@dataclass
class PredictionBundle:
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    window_id: np.ndarray
    loss: float


def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> PredictionBundle:
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch, ids_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            probabilities = torch.softmax(logits, dim=1)
            total_loss += float(loss.item()) * len(y_batch)
            total += len(y_batch)
            all_true.append(y_batch.cpu().numpy())
            all_pred.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_ids.append(ids_batch.numpy())
    return PredictionBundle(
        y_true=np.concatenate(all_true),
        y_pred=np.concatenate(all_pred),
        probabilities=np.concatenate(all_probabilities),
        window_id=np.concatenate(all_ids),
        loss=total_loss / max(total, 1),
    )


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    validation_ids: np.ndarray,
    device: torch.device,
    checkpoint_path: str | Path,
    seed: int,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    scheduler_patience: int,
    scheduler_factor: float,
    weighted_loss: bool,
    n_classes: int,
) -> tuple[pd.DataFrame, int, float]:
    seed_everything(seed)
    model = model.to(device)
    weights = class_weights(y_train, n_classes).to(device) if weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )
    train_loader = make_loader(X_train, y_train, train_ids, batch_size, True, seed)
    validation_loader = make_loader(
        X_validation, y_validation, validation_ids, batch_size, False, seed
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_rows: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        train_loss_total = 0.0
        train_total = 0
        for X_batch, y_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item()) * len(y_batch)
            train_total += len(y_batch)
            train_true.append(y_batch.detach().cpu().numpy())
            train_pred.append(torch.argmax(logits, dim=1).detach().cpu().numpy())

        train_true_array = np.concatenate(train_true)
        train_pred_array = np.concatenate(train_pred)
        validation = predict(model, validation_loader, criterion, device)
        validation_f1 = float(
            f1_score(validation.y_true, validation.y_pred, average="macro", zero_division=0)
        )
        scheduler.step(validation_f1)
        history_rows.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss_total / max(train_total, 1),
                "train_accuracy": accuracy_score(train_true_array, train_pred_array),
                "validation_loss": validation.loss,
                "validation_accuracy": accuracy_score(validation.y_true, validation.y_pred),
                "validation_f1_macro": validation_f1,
            }
        )
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={history_rows[-1]['train_loss']:.4f} "
            f"train_acc={history_rows[-1]['train_accuracy']:.4f} "
            f"val_loss={validation.loss:.4f} "
            f"val_f1_macro={validation_f1:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        if validation_f1 > best_f1 + 1e-6:
            best_f1 = validation_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "best_epoch": best_epoch,
                    "best_validation_f1_macro": best_f1,
                    "seed": seed,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"[early-stop] epoch={epoch} best_epoch={best_epoch} "
                    f"best_val_f1_macro={best_f1:.4f}",
                    flush=True,
                )
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return pd.DataFrame(history_rows), best_epoch, best_f1

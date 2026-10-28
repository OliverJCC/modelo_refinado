from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DatasetArrays:
    X: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray
    window_id: np.ndarray
    label_names: np.ndarray
    feature_names: np.ndarray
    sampling_rate_hz: float


def load_dataset(path: str | Path) -> DatasetArrays:
    with np.load(path, allow_pickle=False) as data:
        return DatasetArrays(
            X=data["X"].astype(np.float32, copy=False),
            y=data["y"].astype(np.int64, copy=False),
            subject_id=data["subject_id"].astype(str),
            window_id=data["window_id"].astype(np.int64, copy=False),
            label_names=data["label_names"].astype(str),
            feature_names=data["feature_names"].astype(str),
            sampling_rate_hz=float(data["sampling_rate_hz"]),
        )


@dataclass
class ChannelStandardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "ChannelStandardizer":
        if X.ndim != 3 or len(X) == 0:
            raise ValueError("X debe ser (N, C, T) y no estar vacío")
        mean = X.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
        std = X.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean[None, :, None]) / self.std[None, :, None]).astype(
            np.float32,
            copy=False,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

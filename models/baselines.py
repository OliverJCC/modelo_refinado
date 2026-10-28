from __future__ import annotations

import torch
import torch.nn as nn
from braindecode.models import ATCNet, CTNet, EEGConformer, MSVTNet

from .improved_lstm import AttentionBiLSTM, CompactAttentionBiLSTM


BASELINE_MODEL_NAMES = ("ctnet", "eegconformer", "atcnet", "msvtnet", "lstm")
MODEL_NAMES = (*BASELINE_MODEL_NAMES, "lstm_attention", "lstm_attention_compact")


class PaperLSTM(nn.Module):
    """LSTM baseline migrated from the previous smartwatch script."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dense_dim: int = 32,
        num_classes: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.bn2 = nn.BatchNorm1d(dense_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dense_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x.transpose(1, 2))
        hidden = self.bn1(sequence[:, -1, :])
        hidden = self.bn2(self.fc1(hidden))
        return self.classifier(self.dropout(hidden))


def build_model(
    name: str,
    n_channels: int,
    n_times: int,
    n_classes: int,
    sampling_rate_hz: float,
) -> nn.Module:
    name = name.lower()
    if name == "ctnet":
        return CTNet(
            n_outputs=n_classes,
            n_chans=n_channels,
            sfreq=sampling_rate_hz,
            n_times=n_times,
            depth=3,
            heads=4,
            n_filters_time=16,
            depth_multiplier=2,
            emb_size=32,
            kernel_size=32,
            pool_size_1=4,
            pool_size_2=4,
            drop_prob_cnn=0.3,
            drop_prob_posi=0.1,
            drop_prob_final=0.5,
        )
    if name == "eegconformer":
        return EEGConformer(
            n_outputs=n_classes,
            n_chans=n_channels,
            n_times=n_times,
            sfreq=sampling_rate_hz,
            n_filters_time=32,
            filter_time_length=25,
            pool_time_length=32,
            pool_time_stride=16,
            drop_prob=0.3,
            att_depth=3,
            att_heads=4,
            att_drop_prob=0.3,
            final_fc_length="auto",
        )
    if name == "atcnet":
        return ATCNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            input_window_seconds=n_times / sampling_rate_hz,
            sfreq=sampling_rate_hz,
            conv_block_n_filters=16,
            conv_block_kernel_length_1=64,
            conv_block_kernel_length_2=16,
            n_windows=5,
            att_head_dim=8,
            att_num_heads=2,
            tcn_depth=2,
            tcn_kernel_size=4,
        )
    if name == "msvtnet":
        return MSVTNet(
            n_chans=n_channels,
            n_times=n_times,
            n_outputs=n_classes,
            sfreq=sampling_rate_hz,
        )
    if name == "lstm":
        return PaperLSTM(input_dim=n_channels, num_classes=n_classes)
    if name == "lstm_attention":
        return AttentionBiLSTM(input_dim=n_channels, num_classes=n_classes)
    if name == "lstm_attention_compact":
        return CompactAttentionBiLSTM(input_dim=n_channels, num_classes=n_classes)
    raise ValueError(f"Modelo desconocido: {name}. Opciones: {MODEL_NAMES}")

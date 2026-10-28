from __future__ import annotations

import torch
import torch.nn as nn


class AttentionBiLSTM(nn.Module):
    """Regularized bidirectional LSTM with learned temporal pooling.

    This is intentionally kept separate from ``PaperLSTM`` so the selected
    baseline remains unchanged and the comparison is reproducible.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        projection_dim: int = 48,
        hidden_dim: int = 48,
        num_layers: int = 2,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.recurrent = nn.LSTM(
            projection_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        representation_dim = hidden_dim * 2
        self.sequence_norm = nn.LayerNorm(representation_dim)
        self.attention = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(representation_dim * 2, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        sequence = self.input_projection(self.input_norm(sequence))
        sequence, _ = self.recurrent(sequence)
        sequence = self.sequence_norm(sequence)
        attention_weights = torch.softmax(self.attention(sequence), dim=1)
        attended = torch.sum(attention_weights * sequence, dim=1)
        mean_pooled = sequence.mean(dim=1)
        return self.classifier(torch.cat((attended, mean_pooled), dim=1))


class CompactAttentionBiLSTM(nn.Module):
    """Lower-capacity revision intended to reduce subject-level overfitting."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_dim: int = 32,
        dropout: float = 0.55,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(dropout * 0.5)
        self.recurrent = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        representation_dim = hidden_dim * 2
        self.sequence_norm = nn.LayerNorm(representation_dim)
        self.attention = nn.Sequential(
            nn.Linear(representation_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(representation_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.input_dropout(self.input_norm(x.transpose(1, 2)))
        sequence, _ = self.recurrent(sequence)
        sequence = self.sequence_norm(sequence)
        attention_weights = torch.softmax(self.attention(sequence), dim=1)
        attended = torch.sum(attention_weights * sequence, dim=1)
        return self.classifier(attended)

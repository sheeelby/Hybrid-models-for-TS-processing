"""Reference implementation of the Helformer architecture (Kehinde et al., 2025)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class HoltWintersDecomposition(nn.Module):
    """Holt–Winters decomposition layer with trainable smoothing coefficients."""

    def __init__(self) -> None:
        super().__init__()
        self.logit_alpha = nn.Parameter(torch.tensor(0.0))
        self.logit_gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        _ = batch
        alpha = torch.sigmoid(self.logit_alpha)
        gamma = torch.sigmoid(self.logit_gamma)

        level_prev = x[:, 0, :].clone()
        season_prev = torch.ones_like(level_prev)

        levels = [level_prev.unsqueeze(1)]
        seasons = [season_prev.unsqueeze(1)]
        y_values = []

        y_t = x[:, 0, :] / (level_prev * season_prev + 1e-8)
        y_values.append(y_t.unsqueeze(1))

        for t in range(1, seq_len):
            x_t = x[:, t, :]
            level_t = alpha * (x_t / (season_prev + 1e-8)) + (1 - alpha) * level_prev
            season_t = gamma * (x_t / (level_t + 1e-8)) + (1 - gamma) * season_prev
            y_t = x_t / (level_t * season_t + 1e-8)

            levels.append(level_t.unsqueeze(1))
            seasons.append(season_t.unsqueeze(1))
            y_values.append(y_t.unsqueeze(1))
            level_prev = level_t
            season_prev = season_t

        level = torch.cat(levels, dim=1)
        season = torch.cat(seasons, dim=1)
        y = torch.cat(y_values, dim=1)
        return level, season, y


class Helformer(nn.Module):
    """Single-step Helformer block (HW decomposition + attention + LSTM)."""

    def __init__(
        self,
        input_dim: int = 1,
        num_heads: int = 4,
        head_dim: int = 48,
        lstm_units: int = 20,
        dropout: float = 0.02,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = num_heads * head_dim
        self.decomposition = HoltWintersDecomposition()
        self.embed = nn.Linear(input_dim, self.d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(lstm_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        level, season, y = self.decomposition(x)
        h = self.embed(y)
        attn_out, _ = self.attention(h, h, h)
        h = self.attn_norm(h + self.attn_dropout(attn_out))
        lstm_out, _ = self.lstm(h)
        lstm_out = self.lstm_dropout(lstm_out)
        last_hidden = lstm_out[:, -1, :]
        forecast = self.out(last_hidden)
        last_level = level[:, -1, :]
        last_season = season[:, -1, :]
        return forecast * last_level * last_season


class HelformerAutoRegressor(nn.Module):
    """Autoregressive multi-step wrapper around the single-step Helformer."""

    def __init__(
        self,
        horizon: int,
        *,
        input_dim: int = 1,
        num_heads: int = 4,
        head_dim: int = 48,
        lstm_units: int = 20,
        dropout: float = 0.02,
        teacher_forcing: float = 0.5,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.input_dim = input_dim
        self.teacher_forcing = float(teacher_forcing)
        self.base = Helformer(
            input_dim=input_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            lstm_units=lstm_units,
            dropout=dropout,
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.transpose(1, 2)
        if x.dim() != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Helformer expects input shape (batch, seq, {self.input_dim}), got {tuple(x.shape)}")
        return x

    def _rollout(self, seq: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        state = seq
        preds = []
        teacher = None
        tf_prob = self.teacher_forcing if self.training else 0.0
        if target is not None and tf_prob > 0.0:
            teacher = target.unsqueeze(-1).to(seq.dtype)
        for step in range(self.horizon):
            next_pred = self.base(state)
            preds.append(next_pred.squeeze(-1))
            next_val = next_pred.unsqueeze(1)
            if teacher is not None:
                mask = torch.rand(state.size(0), 1, 1, device=state.device)
                use_teacher = mask < tf_prob
                teacher_val = teacher[:, step : step + 1, :]
                next_val = torch.where(use_teacher, teacher_val, next_val)
            state = torch.cat([state[:, 1:, :], next_val], dim=1)
        return torch.stack(preds, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._prepare_input(x)
        return self._rollout(seq)

    def forward_with_target(self, x: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
        seq = self._prepare_input(x)
        return self._rollout(seq, target=target)


__all__ = ["Helformer", "HelformerAutoRegressor", "HoltWintersDecomposition"]

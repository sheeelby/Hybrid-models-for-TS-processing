"""Helformer-style attention + LSTM forecaster tuned for this project.

The model is implemented as a direct multi-step forecaster: given an input
window of shape ``(batch, seq_len, 1)`` it predicts the next ``horizon``
values in one shot, similar to TimesNet and N-BEATS. This avoids the
autoregressive roll-out issues (e.g. first step reasonable, further steps
collapsing) and makes Helformer more competitive on M3 and synthetic data.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class HoltWintersDecomposition(nn.Module):
    """Holt–Winters-style decomposition with trainable smoothing coefficients.

    This block is kept for experimentation and can be enabled via the
    ``use_decomposition`` flag in :class:`Helformer`, but is disabled by
    default in the pipelines.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logit_alpha = nn.Parameter(torch.tensor(0.0))
        self.logit_gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
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
            level_t = alpha * (x_t / (season_prev + 1e-8)) + (1.0 - alpha) * level_prev
            season_t = gamma * (x_t / (level_t + 1e-8)) + (1.0 - gamma) * season_prev
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
    """Multi-step attention + LSTM forecaster.

    The architecture:
    - (optional) Holt–Winters decomposition to obtain deseasonalised series;
    - per-window normalisation to zero mean / unit variance;
    - linear embedding to ``d_model``;
    - single multi-head self-attention block with residual and LayerNorm;
    - single-layer LSTM;
    - final linear layer projecting the last hidden state to ``horizon``
      future values.
    """

    def __init__(
        self,
        horizon: int,
        *,
        input_dim: int = 1,
        num_heads: int = 4,
        head_dim: int = 48,
        lstm_units: int = 32,
        dropout: float = 0.1,
        use_decomposition: bool = False,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.input_dim = input_dim
        self.d_model = num_heads * head_dim
        self.use_decomposition = use_decomposition
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
        self.out = nn.Linear(lstm_units, self.horizon)

    @staticmethod
    def _normalise_core(core: torch.Tensor) -> torch.Tensor:
        mean = core.mean(dim=1, keepdim=True)
        std = core.std(dim=1, keepdim=True) + 1e-6
        return (core - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, input_dim)``.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(batch, horizon)``.
        """
        if self.use_decomposition:
            _, _, core = self.decomposition(x)
        else:
            core = x

        core_norm = self._normalise_core(core)
        h = self.embed(core_norm)
        attn_out, _ = self.attention(h, h, h)
        h = self.attn_norm(h + self.attn_dropout(attn_out))
        lstm_out, _ = self.lstm(h)
        lstm_out = self.lstm_dropout(lstm_out)
        last_hidden = lstm_out[:, -1, :]
        forecast = self.out(last_hidden)
        return forecast


class HelformerAutoRegressor(nn.Module):
    """Thin wrapper exposing the same interface as other models.

    The wrapper simply calls the underlying :class:`Helformer` which already
    predicts the full horizon in one shot.  ``forward_with_target`` ignores the
    target and delegates to :meth:`forward`, which is compatible with the
    generic training loop.
    """

    def __init__(
        self,
        horizon: int,
        *,
        input_dim: int = 1,
        num_heads: int = 4,
        head_dim: int = 48,
        lstm_units: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.base = Helformer(
            horizon=horizon,
            input_dim=input_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            lstm_units=lstm_units,
            dropout=dropout,
            use_decomposition=False,
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() != 3:
            raise ValueError(f"Helformer expects 3D input, got shape={tuple(x.shape)}")
        if x.shape[-1] == self.input_dim:
            return x
        if x.shape[1] == self.input_dim:
            return x.transpose(1, 2)
        if x.shape[1] == 1:
            return x.transpose(1, 2)
        raise ValueError(f"Cannot infer channel dimension for input shape={tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._prepare_input(x)
        return self.base(seq)

    def forward_with_target(self, x: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
        # Target is unused; the training loop compares the model output directly
        # to the provided horizon-length targets.
        return self.forward(x)


__all__ = ["HoltWintersDecomposition", "Helformer", "HelformerAutoRegressor"]


"""
Implementation of the Helformer architecture described in the Journal of Big
Data paper “Helformer: an attention‑based deep learning model for cryptocurrency
price forecasting” (Kehinde et al., 2025)【925666294844882†L599-L616】.  The
original work combines Holt‑Winters exponential smoothing with a transformer
encoder and an LSTM block.  The decomposition block uses trainable smoothing
coefficients α (for the level) and γ (for the seasonal component) to
iteratively compute level (Lₜ) and seasonality (Sₜ) for each time step.  A
window of deseasonalised observations Yₜ = Xₜ/(Lₜ·Sₜ) is then processed by a
multi‑head attention layer, followed by residual connections, layer
normalisation and an LSTM layer.  A final dense layer produces the
forecast【925666294844882†L599-L616】.  The architecture reduces the number of
transformer blocks to one and replaces the standard feed‑forward network with
an LSTM【925666294844882†L599-L616】, which helps capture long‑range temporal
dependencies.

This implementation is written in PyTorch and is intended as a starting point
for experimentation.  Hyperparameters such as the number of attention heads,
head dimension, number of LSTM units and dropout rates can be tuned via
Bayesian optimisation as described in the paper【925666294844882†L1288-L1303】.
The default values below follow the optimal configuration reported in Table 5
of the paper (4 heads with head size 48, 20 LSTM units, dropout≈0.02)【925666294844882†L1288-L1303】.

References
----------
Kehinde, T. O., Adedokun, O. J., Joseph, A., Kabirat, K. M., Akano, H. A., &
Olanrewaju, O. A. (2025). *Helformer: an attention‑based deep learning model
for cryptocurrency price forecasting*. Journal of Big Data, 12(81).  DOI:
10.1186/s40537‑025‑01135‑4【925666294844882†L599-L616】.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HoltWintersDecomposition(nn.Module):
    """Holt–Winters decomposition layer with trainable smoothing coefficients.

    Given a batch of univariate time series X of shape (batch, seq_len, 1),
    this layer computes per‑time‑step level and seasonal components using the
    multiplicative Holt–Winters equations described in the Helformer paper
    【925666294844882†L620-L636】.  The smoothing coefficients α and γ are
    represented in logit space and constrained to (0, 1) via a sigmoid.

    The deseasonalised observations Yₜ = Xₜ / (Lₜ·Sₜ) are returned along with
    the running level and seasonality sequences.  To keep the implementation
    simple and differentiable, the seasonal component is updated using a
    recursive formula similar to equation (2) in the paper【925666294844882†L640-L646】.
    """

    def __init__(self) -> None:
        super().__init__()
        # Parameters are initialised in logit space.  They are scalars shared
        # across the batch.  A sigmoid is applied during forward to ensure
        # values lie in (0, 1).
        self.logit_alpha = nn.Parameter(torch.tensor(0.0))
        self.logit_gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute level, seasonality and deseasonalised observations.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, 1).

        Returns
        -------
        level : torch.Tensor
            Tensor of shape (batch, seq_len, 1) containing the level sequence.

        season : torch.Tensor
            Tensor of shape (batch, seq_len, 1) containing the seasonal sequence.

        y : torch.Tensor
            Deseasonalised observations of shape (batch, seq_len, 1).
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        # Constrain α and γ to the unit interval using sigmoid
        alpha = torch.sigmoid(self.logit_alpha)
        gamma = torch.sigmoid(self.logit_gamma)

        # Initialise level and season; start with the first observation and unit
        # seasonality.  Broadcasting ensures shape (batch, 1).
        level_prev = x[:, 0, :].clone()
        season_prev = torch.ones_like(level_prev)

        levels = [level_prev.unsqueeze(1)]
        seasons = [season_prev.unsqueeze(1)]
        y_values = []

        # Deseasonalise the first point
        y_t = x[:, 0, :] / (level_prev * season_prev + 1e-8)
        y_values.append(y_t.unsqueeze(1))

        # Iterate through the sequence updating level and seasonality
        for t in range(1, seq_len):
            x_t = x[:, t, :]
            # Update level according to Eq. 1 in the paper【925666294844882†L638-L644】
            level_t = alpha * (x_t / (season_prev + 1e-8)) + (1 - alpha) * level_prev
            # Update seasonality according to a simplified version of Eq. 2【925666294844882†L640-L646】
            # This approximation uses the current observation and updated level.
            season_t = gamma * (x_t / (level_t + 1e-8)) + (1 - gamma) * season_prev
            # Deseasonalise
            y_t = x_t / (level_t * season_t + 1e-8)

            # Store sequences and prepare for next iteration
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
    """PyTorch implementation of the Helformer model.

    The architecture closely follows the description in the Helformer paper: a
    Holt–Winters decomposition layer is applied to the input, the
    deseasonalised series is embedded and processed by a single encoder block
    consisting of multi‑head self‑attention, residual connections and layer
    normalisation, and the resulting sequence is passed through an LSTM
    followed by a fully connected output layer【925666294844882†L599-L616】.  The
    predicted output is re‑seasonalised by multiplying the forecast with the
    last estimated level and seasonality values.
    """

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
        # Linear embedding to project deseasonalised observations to d_model
        self.embed = nn.Linear(input_dim, self.d_model)
        # Multi‑head attention layer (single encoder block)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.d_model)
        self.attn_dropout = nn.Dropout(dropout)
        # LSTM layer replacing the feed‑forward network
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(dropout)
        self.out = nn.Linear(lstm_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Helformer.

        Parameters
        ----------
        x : torch.Tensor
            Input time series of shape (batch, seq_len, input_dim).  The final
            prediction assumes that the target is the next time step after the
            input window.

        Returns
        -------
        torch.Tensor
            Predicted next value of shape (batch, 1).
        """
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
    """Обертка, которая превращает одношаговый Helformer в многошаговый авторегрессор.

    Архитектура самой модели не меняется: мы просто повторно подаем на вход
    скользящее окно, подставляя либо предсказанный шаг (инференс), либо
    правильный target (teacher forcing на обучении).
    """

    def __init__(
        self,
        horizon: int,
        *,
        input_dim: int = 1,
        num_heads: int = 4,
        head_dim: int = 48,
        lstm_units: int = 20,
        dropout: float = 0.02,
    ) -> None:
        super().__init__()
        self.horizon = max(1, int(horizon))
        self.base = Helformer(
            input_dim=input_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            lstm_units=lstm_units,
            dropout=dropout,
        )
        self.input_dim = input_dim

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() != 3:
            raise ValueError(f"Helformer ожидает 3D-тензор, получено shape={tuple(x.shape)}")
        if x.shape[-1] == self.input_dim:
            return x
        if x.shape[1] == self.input_dim:
            return x.transpose(1, 2)
        if x.shape[1] == 1:
            return x.transpose(1, 2)
        raise ValueError(f"Не удалось привести вход к виду (batch, seq, {self.input_dim})")

    @staticmethod
    def _ensure_column(pred: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        return pred

    def _rollout(self, seq: torch.Tensor, teacher: torch.Tensor | None = None) -> torch.Tensor:
        preds = []
        cur = seq
        teacher_seq = teacher
        if teacher_seq is not None:
            teacher_seq = teacher_seq.to(seq.device)
            if teacher_seq.dim() == 1:
                teacher_seq = teacher_seq.unsqueeze(1)
            if teacher_seq.dim() == 3:
                teacher_seq = teacher_seq.squeeze(-1)
        for step in range(self.horizon):
            pred = self._ensure_column(self.base(cur))
            preds.append(pred)
            if teacher_seq is not None and step < teacher_seq.shape[1]:
                next_val = teacher_seq[:, step : step + 1]
            else:
                next_val = pred
            next_input = next_val.unsqueeze(-1)
            cur = torch.cat([cur[:, 1:, :], next_input], dim=1)
        return torch.cat(preds, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._prepare_input(x)
        return self._rollout(seq)

    def forward_with_target(self, x: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
        seq = self._prepare_input(x)
        if target is None:
            return self._rollout(seq)
        return self._rollout(seq, teacher=target)


if __name__ == "__main__":
    # Example usage: generate a batch of synthetic data and run the model.
    # This block is not executed when the module is imported.  It is provided
    # for illustration and basic sanity checking.
    batch_size = 8
    seq_len = 30
    # Create a synthetic sine wave with noise to mimic a seasonal time series
    t = torch.linspace(0, 4 * torch.pi, seq_len)
    x = (torch.sin(t)[None, :, None] + 0.1 * torch.randn(batch_size, seq_len, 1)) + 2.0
    horizon = 5
    model = Helformer()
    wrapper = HelformerAutoRegressor(horizon=horizon)
    with torch.no_grad():
        single = model(x)
        multi = wrapper(x)
    print("Single-step:", single.shape, "Multi-step:", multi.shape)

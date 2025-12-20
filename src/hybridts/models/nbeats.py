from __future__ import annotations

import math

import torch
import torch.nn as nn


class NBEATBlock(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        width: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
        block_type: str = "generic",
        polynomial_degree: int = 3,
        n_harmonics: int | None = None,
    ):

        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.block_type = block_type

        layers = []
        in_features = lookback
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(in_features, width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = width
        self.fc = nn.Sequential(*layers)

        if block_type == "generic":
            self.theta_b = nn.Linear(width, lookback)
            self.theta_f = nn.Linear(width, horizon)
            self.register_buffer("backcast_basis", None, persistent=False)
            self.register_buffer("forecast_basis", None, persistent=False)
            self.n_theta_backcast = lookback
            self.n_theta_forecast = horizon
        else:
            if block_type == "trend":
                degree = max(1, int(polynomial_degree))
                n_theta_backcast = degree + 1
                n_theta_forecast = degree + 1

                scale = float(max(1, lookback))
                t_b = torch.arange(-lookback + 1, 1, dtype=torch.float32) / scale
                t_f = torch.arange(1, horizon + 1, dtype=torch.float32) / scale
                backcast_basis = torch.stack([t_b ** i for i in range(n_theta_backcast)], dim=0)
                forecast_basis = torch.stack([t_f ** i for i in range(n_theta_forecast)], dim=0)
            elif block_type == "seasonality":
                max_harm = lookback // 2
                if max_harm <= 0:
                    max_harm = 1
                K = int(n_harmonics) if n_harmonics is not None else min(10, max_harm)
                K = max(1, K)

                scale = float(max(1, lookback))
                t_b = torch.arange(-lookback + 1, 1, dtype=torch.float32) / scale
                t_f = torch.arange(1, horizon + 1, dtype=torch.float32) / scale
                freqs = torch.arange(1, K + 1, dtype=torch.float32)

                ang_b = 2 * math.pi * freqs.unsqueeze(1) * t_b.unsqueeze(0)
                ang_f = 2 * math.pi * freqs.unsqueeze(1) * t_f.unsqueeze(0)
                cos_b = torch.cos(ang_b)
                sin_b = torch.sin(ang_b)
                cos_f = torch.cos(ang_f)
                sin_f = torch.sin(ang_f)

                backcast_basis = torch.cat([cos_b, sin_b], dim=0)
                forecast_basis = torch.cat([cos_f, sin_f], dim=0)
                n_theta_backcast = backcast_basis.shape[0]
                n_theta_forecast = forecast_basis.shape[0]
            else:
                raise ValueError(f"Unsupported N-BEATS block_type: {block_type}")

            self.n_theta_backcast = n_theta_backcast
            self.n_theta_forecast = n_theta_forecast
            self.theta = nn.Linear(width, n_theta_backcast + n_theta_forecast)
            self.register_buffer("backcast_basis", backcast_basis, persistent=False)
            self.register_buffer("forecast_basis", forecast_basis, persistent=False)

    def forward(self, x: torch.Tensor):
        hidden = self.fc(x)
        if self.block_type == "generic":
            backcast = self.theta_b(hidden)
            forecast = self.theta_f(hidden)
            return backcast, forecast

        theta = self.theta(hidden)
        theta_b, theta_f = torch.split(
            theta, [self.n_theta_backcast, self.n_theta_forecast], dim=-1
        )
        backcast = torch.matmul(theta_b, self.backcast_basis)
        forecast = torch.matmul(theta_f, self.forecast_basis)
        return backcast, forecast


class NBEATSStack(nn.Module):
    """Doubly residual stacking with optional weight sharing inside a stack."""

    def __init__(
        self,
        lookback: int,
        horizon: int,
        *,
        width: int,
        depth: int,
        nblocks: int,
        dropout: float,
        block_type: str,
        polynomial_degree: int,
        n_harmonics: int | None,
        share_weights: bool = True,
    ) -> None:
        super().__init__()
        self.nblocks = int(max(1, nblocks))
        self.share_weights = bool(share_weights)

        if self.share_weights:
            self.block = NBEATBlock(
                lookback,
                horizon,
                width=width,
                depth=depth,
                dropout=dropout,
                block_type=block_type,
                polynomial_degree=polynomial_degree,
                n_harmonics=n_harmonics,
            )
            self.blocks = None
        else:
            self.block = None
            self.blocks = nn.ModuleList(
                [
                    NBEATBlock(
                        lookback,
                        horizon,
                        width=width,
                        depth=depth,
                        dropout=dropout,
                        block_type=block_type,
                        polynomial_degree=polynomial_degree,
                        n_harmonics=n_harmonics,
                    )
                    for _ in range(self.nblocks)
                ]
            )

    def forward(self, backcast: torch.Tensor, forecast: torch.Tensor):
        if self.share_weights and self.block is not None:
            for _ in range(self.nblocks):
                b, f = self.block(backcast)
                backcast = backcast - b
                forecast = forecast + f
            return backcast, forecast

        if self.blocks is None:
            return backcast, forecast
        for blk in self.blocks:
            b, f = blk(backcast)
            backcast = backcast - b
            forecast = forecast + f
        return backcast, forecast


class NBEATSV2(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        width: int = 256,
        depth: int = 4,
        nblocks: int = 4,
        dropout: float = 0.1,
        use_anchor: bool = False,
        share_weights: bool = True,
        use_trend: bool = True,
        use_seasonality: bool = True,
        use_generic: bool = True,
        diff_loss_weight: float = 0.08,
    ):

        super().__init__()
        self.use_anchor = bool(use_anchor)
        self.horizon = int(horizon)
        self.diff_loss_weight = float(diff_loss_weight)
        stack_kinds = []
        if use_trend:
            stack_kinds.append("trend")
        if use_seasonality:
            stack_kinds.append("seasonality")
        if use_generic:
            stack_kinds.append("generic")
        if not stack_kinds:
            stack_kinds = ["generic"]
        stack_kinds = tuple(stack_kinds)
        nblocks = int(max(1, nblocks))
        base = nblocks // len(stack_kinds)
        extra = nblocks % len(stack_kinds)
        counts = [base + (i < extra) for i in range(len(stack_kinds))]
        poly_deg = 3 if self.horizon < 18 else 4
        n_harm = None
        self.stacks = nn.ModuleList(
            [
                NBEATSStack(
                    lookback,
                    self.horizon,
                    width=width,
                    depth=depth,
                    nblocks=counts[i],
                    dropout=dropout,
                    block_type=kind,
                    polynomial_degree=poly_deg,
                    n_harmonics=n_harm,
                    share_weights=share_weights,
                )
                for i, kind in enumerate(stack_kinds)
                if counts[i] > 0
            ]
        )

    def forward(self, x: torch.Tensor):
        batch, _, lookback = x.shape
        backcast = x.view(batch, lookback)
        forecast = torch.zeros(batch, self.horizon, device=x.device, dtype=x.dtype)
        for stack in self.stacks:
            backcast, forecast = stack(backcast, forecast)
        if (not self.training) and forecast.size(1) >= 3:
            diffs = torch.abs(forecast[:, 1:] - forecast[:, :-1])  # (B, H-1)
            rest = diffs[:, 1:] if diffs.size(1) >= 3 else diffs
            scale = torch.quantile(rest, 0.75, dim=1, keepdim=True) + 1e-6
            d01 = torch.abs(forecast[:, 0] - forecast[:, 1]).unsqueeze(1)
            d12 = torch.abs(forecast[:, 1] - forecast[:, 2]).unsqueeze(1)
            mask = (d01 > 6.0 * scale) & (d12 < 2.5 * scale)
            if torch.any(mask):
                forecast = forecast.clone()
                forecast[:, 0] = torch.where(mask.squeeze(1), forecast[:, 1], forecast[:, 0])
        if not self.use_anchor:
            return forecast
        anchor = x[:, :, -1].squeeze(1).unsqueeze(1)
        return forecast + anchor


__all__ = ["NBEATSV2"]
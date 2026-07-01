from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class ResidualMLP(nn.Module):
    """Small MLP that predicts the residual energy over a physical baseline."""

    def __init__(self, input_dim: int, hidden_layers: Sequence[int] = (64, 64, 32), dropout: float = 0.0, zero_last: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        final = nn.Linear(prev, 1)
        if zero_last:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def get_parameters(model: nn.Module) -> list:
    return [val.cpu().detach().numpy().copy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list) -> None:
    state_dict = model.state_dict()
    new_state = {}
    for key, value in zip(state_dict.keys(), parameters):
        new_state[key] = torch.tensor(value, dtype=state_dict[key].dtype)
    model.load_state_dict(new_state, strict=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

"""Numeric MLP with the paper's Linear -> ReLU -> Dropout hidden blocks.

The development configuration's default ``dropout=0`` preserves the previously
verified no-dropout architecture and its state_dict keys/parameter count.
For new tuned baselines, ``dropout`` is an independent model-specific variable.
Reference: Gorishniy et al. (2021), official bin/mlp.py.
"""
from __future__ import annotations

from torch import nn


class MLP(nn.Module):
    def __init__(
        self, input_features: int, widths: list[int], n_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_features < 1 or n_classes < 2 or not widths or any(w < 1 for w in widths):
            raise ValueError('Invalid MLP dimensions')
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError('MLP dropout must be in [0, 1)')
        components = []
        previous = input_features
        for width in widths:
            components.extend((nn.Linear(previous, width), nn.ReLU()))
            if dropout:
                components.append(nn.Dropout(float(dropout)))
            previous = width
        components.append(nn.Linear(previous, n_classes))
        self.layers = nn.Sequential(*components)
        self.dropout = float(dropout)

    def forward(self, x):
        return self.layers(x)

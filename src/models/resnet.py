"""Numeric-only adaptation of the ORIGINAL Gorishniy et al. (2021) ResNet.

Reference: https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/resnet.py
Original class: ResNet (input projection, pre-norm residual blocks, normalized head).

Jannis contains exclusively numerical features, hence the original categorical
embedding path is deliberately absent. The original block ordering, parameter
shapes, activation choices and forward operations are retained. This is NOT
our earlier custom ResidualMLP.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _activate(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == 'relu':
        return F.relu(x)
    if name == 'gelu':
        return F.gelu(x)
    if name == 'reglu':
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)
    if name == 'geglu':
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)
    raise ValueError(f'Unsupported original ResNet activation: {name}')


class ResNet(nn.Module):
    """Paper-faithful ResNet for the numeric-only Jannis data path.

    Architecture: Linear -> [Norm -> Linear -> Activation -> Dropout ->
    Linear -> Dropout -> ResidualAdd] * n_layers -> Norm -> Activation -> Linear.

    Original paper config fields: d, d_hidden_factor, n_layers, activation,
    normalization, hidden_dropout, residual_dropout.
    """

    def __init__(
        self,
        input_features: int,
        n_classes: int,
        *,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        activation: str,
        normalization: str,
        hidden_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()
        if input_features <= 0 or n_classes < 2 or d < 1 or n_layers < 1 or d_hidden_factor <= 0:
            raise ValueError('Invalid paper ResNet dimensions')
        if activation not in ('relu', 'reglu', 'geglu'):
            raise ValueError(f'Unsupported activation {activation!r}')
        if normalization not in ('batchnorm', 'layernorm'):
            raise ValueError(f'Unsupported normalization {normalization!r}')
        if not 0 <= hidden_dropout < 1 or not 0 <= residual_dropout < 1:
            raise ValueError('Dropout must be in [0, 1)')
        d_hidden = int(d * d_hidden_factor)
        if d_hidden < 1:
            raise ValueError('d_hidden_factor produces zero hidden units')
        self.activation_name = activation
        self.hidden_dropout = float(hidden_dropout)
        self.residual_dropout = float(residual_dropout)
        self.d_hidden = d_hidden

        def make_normalization() -> nn.Module:
            return nn.BatchNorm1d(d) if normalization == 'batchnorm' else nn.LayerNorm(d)

        self.first_layer = nn.Linear(input_features, d)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': make_normalization(),
                'linear0': nn.Linear(d, d_hidden * (2 if activation.endswith('glu') else 1)),
                'linear1': nn.Linear(d_hidden, d),
            })
            for _ in range(n_layers)
        ])
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        for layer in self.layers:
            z = layer['norm'](x)
            z = layer['linear0'](z)
            z = _activate(z, self.activation_name)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = _activate(x, 'relu' if self.activation_name == 'reglu' else 'gelu'
                      if self.activation_name == 'geglu' else self.activation_name)
        return self.head(x)

"""Experimental configurable Denser Perceptron Network (DPN).

Additive prototype. Does NOT replace src.models.dpn.DPN or existing tickets.
Native PyTorch autograd handles optional LayerNorm, masking, dropout and stage
transitions. All choices are explicit, auditable configuration fields.

Stage boundaries use zero-based hidden block positions. At a boundary, the
entire accumulated representation is projected to the given transition width;
optionally, unmodified original features bypass the projection. Output blocks
are always linear logits, and later output blocks may receive preceding logits
(just as the original DPNAutoGrad implementation does).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

NORMALIZATIONS = ('none', 'pre_activation_ln', 'pre_linear_ln')
CHECKPOINT_FORMAT = 'dpn_flexible_development_v1'


@dataclass(frozen=True)
class FlexibleDPNConfig:
    input_features: int
    hidden_widths: tuple[int, ...]
    output_widths: tuple[int, ...]
    stage_boundaries: tuple[int, ...] = ()
    transition_widths: tuple[int, ...] = ()
    raw_input_bypass: bool = False
    normalization: str = 'none'
    layernorm_affine: bool = False
    layernorm_eps: float = 1e-5
    dropout: float = 0.0
    allow_singleton_pre_activation_ln: bool = False
    transition_activation: str = 'identity'
    initialization: str = 'he_normal_uniform_bias_v1'

    def validate(self) -> None:
        if not isinstance(self.input_features, int) or self.input_features < 1:
            raise ValueError('input_features must be a positive integer')
        if not self.hidden_widths or not self.output_widths:
            raise ValueError('Both hidden_widths and output_widths must be nonempty')
        if any(type(w) is not int or w < 1 for w in (*self.hidden_widths, *self.output_widths)):
            raise ValueError('Every block must have positive integer width')
        if self.normalization not in NORMALIZATIONS:
            raise ValueError('Unsupported normalization mode')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must lie in [0,1)')
        if self.layernorm_eps <= 0:
            raise ValueError('LayerNorm epsilon must be positive')
        if self.normalization == 'pre_linear_ln' and self.input_features == 1:
            raise ValueError('Pre-linear LayerNorm on a one-feature input is degenerate')
        if self.normalization == 'pre_activation_ln' and any(w == 1 for w in self.hidden_widths):
            if not self.allow_singleton_pre_activation_ln:
                raise ValueError('Per-block LayerNorm on singleton hidden blocks collapses output; explicit opt-in required')
        if self.transition_activation != 'identity' or self.initialization != 'he_normal_uniform_bias_v1':
            raise ValueError('Changed transition or initialization requires a new protocol')
        if len(self.stage_boundaries) != len(self.transition_widths):
            raise ValueError('A compression width is required per stage boundary')
        if any(type(i) is not int for i in self.stage_boundaries):
            raise ValueError('Invalid stage boundary index')
        if (tuple(sorted(set(self.stage_boundaries))) != self.stage_boundaries or
                any(i <= 0 or i >= len(self.hidden_widths) for i in self.stage_boundaries)):
            raise ValueError('Stage boundaries must be unique, sorted and inside hidden stack')
        if any(type(w) is not int or w <= 0 for w in self.transition_widths):
            raise ValueError('Transition width must be positive integer')
        if self.raw_input_bypass and not self.stage_boundaries:
            raise ValueError('Raw-input bypass is meaningful only with stage transitions')

    def sha256(self) -> str:
        self.validate()
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                                         separators=(',', ':')).encode()).hexdigest()


class FlexibleDPN(nn.Module):
    """Native-autograd DPN with zero-overhead default path and explicit options."""

    def __init__(self, config: FlexibleDPNConfig, init_seed: int = 42):
        super().__init__()
        config.validate()
        if type(init_seed) is not int or init_seed < 0:
            raise ValueError('Initialization seed must be a nonnegative integer')
        self.config = config
        self.init_seed = init_seed
        gen = torch.Generator(device='cpu').manual_seed(init_seed)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.pre_linear_norms = nn.ModuleList()
        self.pre_activation_norms = nn.ModuleList()
        self.transitions = nn.ModuleDict()
        self.transition_boundaries = {idx: width for idx, width in
                                       zip(config.stage_boundaries, config.transition_widths)}
        self._input_widths: list[int] = []
        self._transition_input_widths: dict[int, int] = {}

        x_width = config.input_features
        n_hidden = len(config.hidden_widths)
        for block_idx, width in enumerate((*config.hidden_widths, *config.output_widths)):
            if block_idx in self.transition_boundaries:
                compressed_width = self.transition_boundaries[block_idx]
                self._transition_input_widths[block_idx] = x_width
                transition = nn.Linear(x_width, compressed_width)
                with torch.no_grad():
                    transition.weight.copy_(torch.randn(
                        (compressed_width, x_width), generator=gen) * (2.0 / x_width) ** 0.5)
                    transition.bias.copy_(torch.rand((compressed_width,), generator=gen))
                self.transitions[str(block_idx)] = transition
                x_width = compressed_width + (config.input_features if config.raw_input_bypass else 0)
            self._input_widths.append(x_width)
            self.weights.append(nn.Parameter(
                torch.randn((width, x_width), generator=gen) * (2.0 / x_width) ** 0.5))
            self.biases.append(nn.Parameter(torch.rand((width,), generator=gen)))
            self.register_buffer(f'mask_{block_idx}', torch.ones((width, x_width)))
            if config.normalization == 'pre_linear_ln':
                self.pre_linear_norms.append(nn.LayerNorm(
                    x_width, eps=config.layernorm_eps, elementwise_affine=config.layernorm_affine))
            else:
                self.pre_linear_norms.append(nn.Identity())
            if config.normalization == 'pre_activation_ln' and block_idx < n_hidden:
                self.pre_activation_norms.append(nn.LayerNorm(
                    width, eps=config.layernorm_eps, elementwise_affine=config.layernorm_affine))
            else:
                self.pre_activation_norms.append(nn.Identity())
            x_width += width
        self.output_features = sum(config.output_widths)
        self._n_hidden = n_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.config.input_features:
            raise ValueError('Input must be [batch, input_features]')
        raw = x
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            if i in self.transition_boundaries:
                x = self.transitions[str(i)](x)
                if self.config.raw_input_bypass:
                    x = torch.cat((raw, x), dim=-1)
            normalized = self.pre_linear_norms[i](x)
            output = F.linear(normalized, weight * getattr(self, f'mask_{i}'), bias)
            if i < self._n_hidden:
                output = self.pre_activation_norms[i](output)
                output = F.relu(output)
                if self.config.dropout:
                    output = F.dropout(output, p=self.config.dropout, training=self.training)
            x = torch.cat((x, output), dim=-1)
        return x[:, -self.output_features:]

    def accounting(self) -> dict[str, int]:
        allocated = sum(p.numel() for p in self.parameters())
        masked = sum(int((getattr(self, f'mask_{i}') == 0).sum().item())
                     for i in range(len(self.weights)))
        return {'parameters_allocated': allocated,
                'parameters_active': allocated - masked,
                'masked_weights': masked,
                'normalization_parameters': sum(p.numel() for m in
                    (*self.pre_linear_norms, *self.pre_activation_norms) for p in m.parameters())}

    def prune_mask(self, block_idx: int, new_mask: torch.Tensor) -> None:
        """Monotonically remove weight connections; never silently unprune."""
        if not 0 <= block_idx < len(self.weights):
            raise IndexError('Invalid DPN block index')
        old = getattr(self, f'mask_{block_idx}')
        if new_mask.shape != old.shape or not torch.all((new_mask == 0) | (new_mask == 1)):
            raise ValueError('Mask shape or binary content invalid')
        if torch.any((old == 0) & (new_mask.to(old.device) != 0)):
            raise ValueError('Unpruning requires a separately audited protocol')
        with torch.no_grad():
            old.copy_(new_mask.to(old.device, dtype=old.dtype))

    def state_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(name.encode()); digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
        return digest.hexdigest()

    def save_design_checkpoint(self, file: Path) -> None:
        file = Path(file)
        if file.exists():
            raise FileExistsError(f'Refusing to overwrite {file}')
        payload: dict[str, Any] = {
            'format': CHECKPOINT_FORMAT,
            'config': asdict(self.config), 'config_sha256': self.config.sha256(),
            'init_seed': self.init_seed, 'state_sha256': self.state_sha256(),
            'state_dict': {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
        }
        torch.save(payload, file)

    @classmethod
    def restore_design_checkpoint(cls, file: Path, config: FlexibleDPNConfig) -> 'FlexibleDPN':
        payload = torch.load(file, map_location='cpu', weights_only=True)
        if (payload.get('format') != CHECKPOINT_FORMAT or
                payload.get('config_sha256') != config.sha256() or
                payload.get('config') != asdict(config)):
            raise ValueError('Model checkpoint configuration mismatch')
        model = cls(config, int(payload['init_seed']))
        model.load_state_dict(payload['state_dict'], strict=True)
        if model.state_sha256() != payload['state_sha256']:
            raise ValueError('Model checkpoint tensor hash mismatch')
        return model

    @classmethod
    def from_canonical(cls, canonical: nn.Module, *, normalization: str = 'none',
                       layernorm_affine: bool = False, dropout: float = 0.0,
                       allow_singleton_pre_activation_ln: bool = False) -> 'FlexibleDPN':
        """Transfer a *global* existing DPN's exact weights, masks and biases.

        Requires complete hidden/output block boundary (same restriction as
        canonical DPN's ReLU/logit separation). Does not alter the source model.
        """
        widths = [int(w.shape[0]) for w in canonical.weights]
        hidden_total = canonical.total_nodes - canonical.output_nodes
        split = 0
        accumulated = 0
        while split < len(widths) and accumulated < hidden_total:
            accumulated += widths[split]
            split += 1
        if accumulated != hidden_total:
            raise ValueError('Canonical hidden/output boundary splits a block')
        config = FlexibleDPNConfig(
            input_features=canonical.input_features, hidden_widths=tuple(widths[:split]),
            output_widths=tuple(widths[split:]), normalization=normalization,
            layernorm_affine=layernorm_affine, dropout=dropout,
            allow_singleton_pre_activation_ln=allow_singleton_pre_activation_ln)
        new = cls(config)
        old_parameters = list(canonical.weights)
        old_masks = [getattr(canonical, f'mask_{i}') for i in range(len(old_parameters))]
        biases = canonical.biases.detach()
        offset = 0
        with torch.no_grad():
            for i, old in enumerate(old_parameters):
                new.weights[i].copy_(old.detach())
                new.biases[i].copy_(biases[offset:offset+old.shape[0]])
                getattr(new, f'mask_{i}').copy_(old_masks[i].detach())
                offset += old.shape[0]
        return new

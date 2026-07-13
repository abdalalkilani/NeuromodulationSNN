import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import (
    DEFAULT_INPUT_BLOCKS,
    DEFAULT_OUTPUT_BLOCKS,
    ExperimentConfig,
    HIDDEN_PARAM_NAMES,
    MOD_PARAM_NAMES,
    OUTPUT_PARAM_NAMES,
    device,
    dtype,
    parse_mask,
)
from .snn import spike_fn


def group_count(size: int, group_size: int) -> int:
    return int(math.ceil(size / max(1, int(group_size))))


def expand_groups(values: torch.Tensor, target_size: int, group_size: int) -> torch.Tensor:
    if group_size <= 1:
        return values
    return values.repeat_interleave(int(group_size), dim=1)[:, :target_size]


def block_dim(name: str, cfg: ExperimentConfig, grouped: bool = False) -> int:
    if name in HIDDEN_PARAM_NAMES:
        return group_count(cfg.nb_hidden, cfg.group_size[0]) if grouped else cfg.nb_hidden
    if name in OUTPUT_PARAM_NAMES:
        return group_count(cfg.nb_outputs, cfg.group_size[1]) if grouped else cfg.nb_outputs
    if name == "in_flat":
        if cfg.channel_compress_enable and cfg.channel_compress_mode in {"all", "mod_only"}:
            return min(cfg.nb_inputs, cfg.channel_compress_target)
        return cfg.nb_inputs
    if name == "hid_flat":
        return cfg.nb_hidden
    if name == "out_flat":
        return cfg.nb_outputs
    raise KeyError(name)


def pack_params(params: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([params[name] for name in MOD_PARAM_NAMES], dim=1)


def map_substitution(raw: torch.Tensor, name: str, cfg: ExperimentConfig) -> torch.Tensor:
    lo, hi = cfg.param_ranges[name]
    return float(lo) + raw * (float(hi) - float(lo))


def apply_additive(base: torch.Tensor, delta: torch.Tensor, name: str, cfg: ExperimentConfig, scale: float = 0.05) -> torch.Tensor:
    lo, hi = cfg.param_ranges[name]
    width = float(hi) - float(lo)
    return (base + scale * width * delta).clamp(float(lo), float(hi))


class NeuromodulatorMapper(nn.Module):
    """Map low-dimensional neuromodulator channel values to parameter effects."""

    def __init__(self, cfg: ExperimentConfig, mode: str):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.hidden_channels, self.output_channels = cfg.nm_counts
        hidden_width = cfg.nm_mapper_hidden_size or max(8, self.hidden_channels * 4)
        output_width = cfg.nm_mapper_hidden_size or max(8, self.output_channels * 4)
        final = nn.Sigmoid if mode == "ann_sub" else nn.Tanh
        self.hidden_mapper = None
        self.output_mapper = None
        if self.hidden_channels > 0:
            self.hidden_mapper = nn.Sequential(
                nn.Linear(self.hidden_channels, hidden_width),
                nn.SiLU(),
                nn.Linear(hidden_width, len(HIDDEN_PARAM_NAMES)),
                final(),
            )
        if self.output_channels > 0:
            self.output_mapper = nn.Sequential(
                nn.Linear(self.output_channels, output_width),
                nn.SiLU(),
                nn.Linear(output_width, len(OUTPUT_PARAM_NAMES)),
                final(),
            )

    @property
    def hidden_flat_dim(self) -> int:
        return self.cfg.nb_hidden * max(0, int(self.hidden_channels))

    @property
    def output_flat_dim(self) -> int:
        return self.cfg.nb_outputs * max(0, int(self.output_channels))

    @property
    def total_dim(self) -> int:
        return self.hidden_flat_dim + self.output_flat_dim

    def forward(self, flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = flat.size(0)
        effects: Dict[str, torch.Tensor] = {}
        offset = 0
        if self.hidden_mapper is not None:
            end = offset + self.hidden_flat_dim
            h = flat[:, offset:end].reshape(batch * self.cfg.nb_hidden, self.hidden_channels)
            h = self.hidden_mapper(h).reshape(batch, self.cfg.nb_hidden, len(HIDDEN_PARAM_NAMES))
            for i, name in enumerate(HIDDEN_PARAM_NAMES):
                effects[name] = h[:, :, i]
            offset = end
        if self.output_mapper is not None:
            end = offset + self.output_flat_dim
            o = flat[:, offset:end].reshape(batch * self.cfg.nb_outputs, self.output_channels)
            o = self.output_mapper(o).reshape(batch, self.cfg.nb_outputs, len(OUTPUT_PARAM_NAMES))
            for i, name in enumerate(OUTPUT_PARAM_NAMES):
                effects[name] = o[:, :, i]
        return effects


class ANNModulator(nn.Module):
    """Paper-facing ANN modulator for substitution or additive updates."""

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.ann_mode
        if self.mode not in {"ann_sub", "ann_add"}:
            raise ValueError("ANNModulator supports ann_sub and ann_add")
        self.input_mask = parse_mask(cfg.ann_in_disable, DEFAULT_INPUT_BLOCKS)
        self.output_mask = parse_mask(cfg.ann_out_disable, DEFAULT_OUTPUT_BLOCKS)
        self.input_blocks = [name for name in DEFAULT_INPUT_BLOCKS if self.input_mask.get(name, False)]
        self.output_blocks = [name for name in MOD_PARAM_NAMES if self.output_mask.get(name, False)]
        self.input_dim = sum(block_dim(name, cfg, grouped=False) for name in self.input_blocks)

        self.nm_mapper: Optional[NeuromodulatorMapper] = None
        if cfg.nm_enable:
            if sum(cfg.nm_counts) <= 0:
                raise ValueError("nm_enable requires at least one positive value in nm_counts")
            self.nm_mapper = NeuromodulatorMapper(cfg, self.mode)
            self.output_dim = self.nm_mapper.total_dim
        else:
            self.out_slices = {}
            offset = 0
            for name in self.output_blocks:
                dim = block_dim(name, cfg, grouped=True)
                self.out_slices[name] = slice(offset, offset + dim)
                offset += dim
            self.output_dim = offset
        if self.output_dim <= 0:
            raise ValueError("The modulator has no enabled outputs")

        hidden_sizes = list(cfg.ann_hidden_sizes or [2048])
        layers: List[nn.Module] = []
        prev = self.input_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.ReLU())
            prev = int(width)
        layers.append(nn.Linear(prev, self.output_dim))
        layers.append(nn.Sigmoid() if self.mode == "ann_sub" and not cfg.nm_enable else nn.Tanh())
        self.net = nn.Sequential(*layers)

    def make_input(self, params, in_flat, hid_flat, out_flat):
        values = {**params, "in_flat": in_flat, "hid_flat": hid_flat, "out_flat": out_flat}
        return torch.cat([values[name] for name in self.input_blocks], dim=1)

    def effects_from_raw(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.nm_mapper is not None:
            effects = self.nm_mapper(raw)
            return {name: value for name, value in effects.items() if self.output_mask.get(name, False)}
        effects = {}
        for name, slc in self.out_slices.items():
            value = raw[:, slc]
            if name in HIDDEN_PARAM_NAMES:
                value = expand_groups(value, self.cfg.nb_hidden, self.cfg.group_size[0])
            else:
                value = expand_groups(value, self.cfg.nb_outputs, self.cfg.group_size[1])
            effects[name] = value
        return effects

    def forward(self, params, in_flat, hid_flat, out_flat) -> Dict[str, torch.Tensor]:
        raw = self.net(self.make_input(params, in_flat, hid_flat, out_flat))
        return self.effects_from_raw(raw)


class SNNAdditiveModulator(nn.Module):
    """Compact spiking modulator used for the paper's SNN-controller condition."""

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.input_mask = parse_mask(cfg.ann_in_disable, DEFAULT_INPUT_BLOCKS)
        self.output_mask = parse_mask(cfg.ann_out_disable, DEFAULT_OUTPUT_BLOCKS)
        self.input_blocks = [name for name in DEFAULT_INPUT_BLOCKS if self.input_mask.get(name, False)]
        self.output_blocks = [name for name in MOD_PARAM_NAMES if self.output_mask.get(name, False)]
        self.input_dim = sum(block_dim(name, cfg, grouped=False) for name in self.input_blocks)
        self.out_slices = {}
        offset = 0
        for name in self.output_blocks:
            dim = block_dim(name, cfg, grouped=True)
            self.out_slices[name] = slice(offset, offset + dim)
            offset += dim
        self.output_dim = offset
        if self.output_dim <= 0:
            raise ValueError("The spiking modulator has no enabled outputs")
        hidden = int(cfg.ann_hidden_sizes[0] if cfg.ann_hidden_sizes else max(64, self.output_dim))
        self.w_in = nn.Linear(self.input_dim, hidden, bias=False)
        self.w_out = nn.Linear(hidden, self.output_dim)
        self.alpha = 0.9
        self.beta = 0.85
        self.threshold = 1.0

    def make_input(self, params, in_flat, hid_flat, out_flat):
        values = {**params, "in_flat": in_flat, "hid_flat": hid_flat, "out_flat": out_flat}
        return torch.cat([values[name] for name in self.input_blocks], dim=1)

    def forward(self, feature_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, steps, _ = feature_sequence.shape
        syn = torch.zeros((batch, self.w_in.out_features), device=feature_sequence.device, dtype=feature_sequence.dtype)
        mem = torch.zeros_like(syn)
        spike_sum = torch.zeros_like(syn)
        for step in range(steps):
            syn = self.alpha * syn + self.w_in(feature_sequence[:, step])
            spk = spike_fn(mem - self.threshold)
            mem = self.beta * mem + (1.0 - self.beta) * syn - spk * self.threshold
            spike_sum = spike_sum + spk
        raw = torch.tanh(self.w_out(spike_sum / max(1, steps)))
        effects = {}
        for name, slc in self.out_slices.items():
            value = raw[:, slc]
            if name in HIDDEN_PARAM_NAMES:
                value = expand_groups(value, self.cfg.nb_hidden, self.cfg.group_size[0])
            else:
                value = expand_groups(value, self.cfg.nb_outputs, self.cfg.group_size[1])
            effects[name] = value
        return effects


def build_modulator(cfg: ExperimentConfig) -> nn.Module:
    if cfg.ann_mode in {"ann_sub", "ann_add"}:
        return ANNModulator(cfg).to(device)
    if cfg.ann_mode == "snn_add":
        return SNNAdditiveModulator(cfg).to(device)
    raise ValueError("This release supports ann_sub, ann_add, and snn_add")

# ===== Block 1: Core (SNN + Modulated SNN) =====

import os, math, random, ast, copy, json
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional, List, Callable

import h5py, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

# Device / dtype
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

SNN_PARAM_NAMES = [
    "w1", "w2", "v1",
    "alpha_hetero_1", "beta_hetero_1",
    "alpha_hetero_2", "beta_hetero_2",
    "thresholds_1", "reset_1", "rest_1",
    "input_delay_logits",
]
MOD_TARGET_PARAM_NAMES = ["alpha_1", "beta_1", "thr", "reset", "rest", "alpha_2", "beta_2"]
HIDDEN_PARAM_NAMES = ["alpha_1", "beta_1", "thr", "reset", "rest"]
OUTPUT_PARAM_NAMES = ["alpha_2", "beta_2"]
WEIGHT_PARAM_NAMES = ["w1", "v1", "w2"]
WEIGHT_SCOPE_CHOICES = {"source", "target", "block"}
WEIGHT_GRID_CHOICES = {"1d", "2d"}

PARAMETER_RANGE_DEFAULTS = {
    "alpha_1": (1.0 / math.e, 0.995),
    "beta_1": (1.0 / math.e, 0.995),
    "alpha_2": (1.0 / math.e, 0.995),
    "beta_2": (1.0 / math.e, 0.995),
    "thr": (0.5, 1.5),
    "reset": (-0.5, 0.5),
    "rest": (-0.5, 0.5),
}
PARAM_RANGE_TARGETS = {
    "alpha_hetero_1": "alpha_1",
    "beta_hetero_1": "beta_1",
    "alpha_hetero_2": "alpha_2",
    "beta_hetero_2": "beta_2",
    "thresholds_1": "thr",
    "reset_1": "reset",
    "rest_1": "rest",
}
ANN_OUTPUT_ACTIVATIONS = {
    "sigmoid": (nn.Sigmoid, (0.0, 1.0)),
    "tanh": (nn.Tanh, (-1.0, 1.0)),
    "linear": (nn.Identity, None),
}
GROUP_DISTRIBUTIONS = {"uniform", "normal"}
GROUP_NORMAL_CUTOFF = 0.05
PARAM_SMOOTH_TAU_INIT_DEFAULT = 0.2  # per-step mixing fraction (0-1)
PARAM_SMOOTH_TAU_MIN_DEFAULT = 0.0


def _resolve_h5_path(cache_dir: Union[str, Path], cache_subdir: Optional[Union[str, Path]], file_name: str) -> str:
    if os.path.isabs(file_name):
        return file_name
    base = Path(os.path.expanduser(str(cache_dir or ".")))
    if cache_subdir:
        base = base / str(cache_subdir)
    return str(base / file_name)

def _normalize_train_flags(flags: Optional[Dict[str, bool]]) -> Dict[str, bool]:
    return {name: bool((flags or {}).get(name, True)) for name in SNN_PARAM_NAMES}

def _snn_train_flags(settings: Dict[str, Union[bool, Dict[str, bool]]]) -> Dict[str, bool]:
    return _normalize_train_flags(settings.get("snn_train_flags"))

def _apply_snn_train_flags(state: Dict[str, torch.nn.Parameter], flags: Dict[str, bool]):
    for name, param in state.items():
        if isinstance(param, torch.Tensor):
            param.requires_grad_(flags.get(name, True))
    return state

def _trainable_params_from(state: Dict[str, torch.nn.Parameter], names: Optional[List[str]] = None) -> List[torch.nn.Parameter]:
    names = names or SNN_PARAM_NAMES
    return [state[n] for n in names if state.get(n) is not None and state[n].requires_grad]

# -------------------------
# Utilities
# -------------------------
def get_run_dir(base_dir: Union[str, Path]) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        d = base / f"Run_{i}"
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            i += 1


def _parse_run_index(name: str) -> Optional[int]:
    if not name.startswith("Run_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _existing_run_indices(base_dir: Union[str, Path]) -> List[int]:
    base = Path(base_dir)
    if not base.exists():
        return []
    indices = []
    for child in base.iterdir():
        if child.is_dir():
            idx = _parse_run_index(child.name)
            if idx is not None:
                indices.append(idx)
    return sorted(indices)


def next_aligned_run_index(*base_dirs: Union[str, Path, None]) -> int:
    max_idx = 0
    for base in base_dirs:
        if base is None:
            continue
        indices = _existing_run_indices(base)
        if indices:
            max_idx = max(max_idx, indices[-1])
    return max_idx + 1


def ensure_run_dir(base_dir: Union[str, Path], run_index: Optional[int] = None) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    if run_index is None:
        return get_run_dir(base)
    next_index = int(run_index)
    while True:
        run_dir = base / f"Run_{next_index}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            next_index += 1

def dist_fn(dist):
    return {
        'gamma': lambda mean, k, size: np.random.gamma(k, scale=mean/k, size=size),
        'normal': lambda mean, k, size: np.random.normal(loc=mean, scale=mean/np.sqrt(k), size=size),
        'uniform': lambda _, maximum, size: np.random.uniform(low=0, high=maximum, size=size),
    }[dist.lower()]

def load_h5(cache_dir: str, cache_subdir: str, train_file: str, test_file: str):
    train_path = _resolve_h5_path(cache_dir, cache_subdir, train_file)
    test_path  = _resolve_h5_path(cache_dir, cache_subdir, test_file)
    train_h5 = h5py.File(train_path, 'r'); test_h5  = h5py.File(test_path, 'r')
    return train_h5['spikes'], train_h5['labels'], test_h5['spikes'], test_h5['labels']


def load_validation_split(cache_dir: str, cache_subdir: str, val_file: Optional[str]):
    if not val_file:
        return None, None
    val_path = _resolve_h5_path(cache_dir, cache_subdir, val_file)
    val_h5 = h5py.File(val_path, 'r')
    return val_h5['spikes'], val_h5['labels']

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_int_list(arg: Optional[Union[str, List[int], Tuple[int, ...]]]) -> Optional[List[int]]:
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        return [int(v) for v in arg]
    text = str(arg).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [int(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass
    tokens = [tok for tok in text.replace(",", " ").split() if tok]
    if not tokens:
        return None
    return [int(tok) for tok in tokens]


def parse_float_list(arg: Optional[Union[str, List[float], Tuple[float, ...]]]) -> Optional[List[float]]:
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        return [float(v) for v in arg]
    text = str(arg).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [float(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass
    tokens = [tok for tok in text.replace(",", " ").split() if tok]
    if not tokens:
        return None
    return [float(tok) for tok in tokens]


def parse_str_list(arg: Optional[Union[str, List[str], Tuple[str, ...]]]) -> Optional[List[str]]:
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        if len(arg) == 1 and isinstance(arg[0], str):
            return parse_str_list(arg[0])
        return [str(v) for v in arg]
    text = str(arg).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass
    tokens = [tok for tok in text.replace(",", " ").split() if tok]
    if not tokens:
        return None
    return [str(tok) for tok in tokens]


def _normalize_combo_param_names(names: Optional[List[str]]) -> List[str]:
    if not names:
        return []
    normalized = []
    aliases = {
        "threshold": "thr",
        "thresholds": "thr",
        "thresh": "thr",
    }
    for name in names:
        if name is None:
            continue
        key = str(name).strip().lower()
        if not key:
            continue
        key = key.replace("-", "_")
        key = key.replace(" ", "")
        key = aliases.get(key, key)
        if key in {"alpha1", "alpha_1"}:
            key = "alpha_1"
        elif key in {"beta1", "beta_1"}:
            key = "beta_1"
        elif key in {"alpha2", "alpha_2"}:
            key = "alpha_2"
        elif key in {"beta2", "beta_2"}:
            key = "beta_2"
        if key in MOD_TARGET_PARAM_NAMES and key not in normalized:
            normalized.append(key)
    return normalized


def _normalize_param_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    normalized = _normalize_combo_param_names([name])
    return normalized[0] if normalized else None


def _parse_param_fraction_map(value, default: float = 1.0) -> Dict[str, float]:
    def _fill_defaults(overrides: Dict[str, float]) -> Dict[str, float]:
        filled = {name: float(default) for name in MOD_TARGET_PARAM_NAMES}
        for key, val in overrides.items():
            name = _normalize_param_name(key)
            if name is None:
                continue
            try:
                filled[name] = float(val)
            except (TypeError, ValueError):
                continue
        return filled

    if value is None:
        return {name: float(default) for name in MOD_TARGET_PARAM_NAMES}
    if isinstance(value, dict):
        return _fill_defaults(value)
    if isinstance(value, (list, tuple)):
        vals = list(value)
        if not vals:
            return {name: float(default) for name in MOD_TARGET_PARAM_NAMES}
        if len(vals) == 1:
            return {name: float(vals[0]) for name in MOD_TARGET_PARAM_NAMES}
        if len(vals) == 2:
            hidden_frac = float(vals[0])
            output_frac = float(vals[1])
            mapped = {name: hidden_frac for name in HIDDEN_PARAM_NAMES}
            mapped.update({name: output_frac for name in OUTPUT_PARAM_NAMES})
            return mapped
        mapped = {}
        for idx, name in enumerate(MOD_TARGET_PARAM_NAMES):
            if idx < len(vals):
                mapped[name] = float(vals[idx])
            else:
                mapped[name] = float(default)
        return mapped
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {name: float(default) for name in MOD_TARGET_PARAM_NAMES}
        parsed = None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
        if parsed is not None:
            return _parse_param_fraction_map(parsed, default=default)
        parsed_list = parse_float_list(text)
        if parsed_list is not None:
            return _parse_param_fraction_map(parsed_list, default=default)
        return {name: float(default) for name in MOD_TARGET_PARAM_NAMES}
    try:
        scalar = float(value)
        return {name: scalar for name in MOD_TARGET_PARAM_NAMES}
    except (TypeError, ValueError):
        return {name: float(default) for name in MOD_TARGET_PARAM_NAMES}

def _resolve_ann_combo_lists(
    additive: Optional[List[str]],
    multiplicative: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    full = list(MOD_TARGET_PARAM_NAMES)
    add_set = set(_normalize_combo_param_names(additive))
    mult_set = set(_normalize_combo_param_names(multiplicative))
    if not add_set and not mult_set:
        add_set = {"thr", "reset", "rest"}
    if not add_set:
        add_set = set(full) - mult_set
    if not mult_set:
        mult_set = set(full) - add_set
    mult_set -= add_set
    missing = set(full) - add_set - mult_set
    mult_set |= missing
    add_list = [p for p in full if p in add_set]
    mult_list = [p for p in full if p in mult_set]
    return add_list, mult_list


def _parse_pair(arg, caster):
    if arg is None:
        return None
    values = arg
    if isinstance(arg, str):
        text = arg.strip()
        if not text:
            return None
        parsed = None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, (list, tuple)):
            return _parse_pair(parsed, caster)
        if parsed is not None:
            values = [parsed]
        else:
            values = [arg]
    elif isinstance(arg, (list, tuple)):
        vals = list(arg)
        if not vals:
            return None
        if len(vals) == 1:
            return _parse_pair(vals[0], caster)
        values = vals
    else:
        values = [values]
    if not values:
        return None
    if len(values) == 1:
        values = [values[0], values[0]]
    else:
        values = [values[0], values[1]]
    return (caster(values[0]), caster(values[1]))


def _cast_str(value):
    return str(value)


def _parse_int_pair(arg):
    return _parse_pair(arg, int)


def _parse_float_pair(arg):
    return _parse_pair(arg, float)


def _parse_str_pair(arg):
    pair = _parse_pair(arg, _cast_str)
    if not pair:
        return None
    return tuple(p.strip() for p in pair)

def _count_snn_params(state: Optional[Dict[str, torch.nn.Parameter]]) -> Tuple[int, int]:
    trainable = frozen = 0
    if not state:
        return trainable, frozen
    for tensor in state.values():
        if tensor is None or not hasattr(tensor, "numel"):
            continue
        num = int(tensor.numel())
        if bool(getattr(tensor, "requires_grad", False)):
            trainable += num
        else:
            frozen += num
    return trainable, frozen

def _count_module_params(module: Optional[nn.Module]) -> Tuple[int, int]:
    trainable = frozen = 0
    if module is None:
        return trainable, frozen
    for p in module.parameters():
        num = int(p.numel())
        if p.requires_grad:
            trainable += num
        else:
            frozen += num
    return trainable, frozen


def _clamp_state_params(state: Dict[str, torch.Tensor], ranges: Dict[str, Tuple[float, float]]):
    for tensor_name, range_name in PARAM_RANGE_TARGETS.items():
        tensor = state.get(tensor_name)
        if tensor is None:
            continue
        _clamp_param_tensor(tensor, range_name, ranges)


def _resolve_param_ranges(settings: Dict) -> Dict[str, Tuple[float, float]]:
    cached = settings.get("_param_ranges")
    if cached is not None:
        return cached
    ranges = {}
    user_ranges = settings.get("param_ranges", {})
    for name, default in PARAMETER_RANGE_DEFAULTS.items():
        rng = user_ranges.get(name)
        if rng is None:
            rng = settings.get(f"{name}_range")
        if rng is None:
            ranges[name] = tuple(default)
            continue
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
            if hi < lo:
                lo, hi = hi, lo
            ranges[name] = (lo, hi)
        else:
            ranges[name] = tuple(default)
    settings["_param_ranges"] = ranges
    return ranges


def _clamp_param_tensor(tensor: torch.Tensor, name: str, ranges: Dict[str, Tuple[float, float]]):
    lo, hi = ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
    tensor.clamp_(float(lo), float(hi))


def _apply_additive_delta(base: torch.Tensor, delta: torch.Tensor, name: str,
                          ranges: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    lo, hi = ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
    return torch.clamp(base + delta, min=float(lo), max=float(hi))


def _map_substitution_output(raw: torch.Tensor, name: str, activation_kind: str,
                             ranges: Dict[str, Tuple[float, float]]) -> torch.Tensor:
    lo, hi = ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
    act_entry = ANN_OUTPUT_ACTIVATIONS.get(activation_kind, (None, None))
    act_range = act_entry[1]

    # Convert MLP output into [0,1] regardless of the activation’s native range.
    if activation_kind == "linear":
        # For linear heads, bias toward midpoint: map 0 -> 0.5 and allow spread around it.
        norm = torch.clamp(raw + 0.5, 0.0, 1.0)
    elif act_range is None:
        norm = torch.sigmoid(raw)  # other unbounded -> squash
    else:
        act_lo, act_hi = act_range
        norm = (torch.clamp(raw, act_lo, act_hi) - act_lo) / max(act_hi - act_lo, 1e-6)
    norm = torch.clamp(norm, 0.0, 1.0)

    mapped = lo + norm * (hi - lo)
    return torch.clamp(mapped, min=float(lo), max=float(hi))


def _psp_peak_gain(alpha: torch.Tensor, beta: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Analytic peak-normalizing gain for bi-exponential PSP.
    alpha = exp(-dt/tau_syn), beta = exp(-dt/tau_mem).
    gain = (tau_mem / tau_syn) ** (tau_mem / (tau_mem - tau_syn))
    Limit as tau_mem -> tau_syn is e.
    """
    eps = 1e-12
    # Recover continuous taus from decay factors and dt.
    tau_syn = -float(dt) / torch.log(torch.clamp(alpha, min=eps))
    tau_mem = -float(dt) / torch.log(torch.clamp(beta, min=eps))
    diff = tau_mem - tau_syn
    ratio = tau_mem / torch.clamp(tau_syn, min=eps)
    # Handle tau_mem ≈ tau_syn with the limit gain -> e.
    close = torch.isclose(diff, torch.zeros_like(diff), atol=1e-8, rtol=1e-6)
    gain = torch.exp(torch.log(torch.clamp(ratio, min=eps)) * (tau_mem / torch.clamp(diff, min=eps)))
    gain = torch.where(close, torch.full_like(gain, math.e), gain)
    gain = torch.clamp(gain, min=0.0)
    gain = torch.where(torch.isfinite(gain), gain, torch.ones_like(gain))
    return gain


def _init_input_delay_logits(settings: Dict, nb_inputs: int) -> torch.nn.Parameter:
    max_delay = max(1, int(settings.get("input_delay_steps", 1)))
    init_val = settings.get("input_delay_init", None)
    init_cap = settings.get("input_delay_init_cap", max_delay)
    if init_cap is None:
        init_cap = max_delay
    init_cap = int(init_cap)
    init_cap = max(0, min(init_cap, max_delay))
    logits = torch.zeros((nb_inputs, max_delay + 1), device=device, dtype=dtype, requires_grad=True)
    noise_scale = float(settings.get("input_delay_init_noise", 0.01))
    if noise_scale > 0:
        logits.data += noise_scale * torch.randn_like(logits)
    with torch.no_grad():
        if init_val is None:
            # Optional small bias toward random delays in [0, init_cap] without saturating logits
            if init_cap >= 0:
                init_delays = torch.randint(0, init_cap + 1, (nb_inputs,), device=device)
                logits[torch.arange(nb_inputs, device=device), init_delays] += float(settings.get("input_delay_init_bias", 0.1))
        else:
            init_delays = torch.full((nb_inputs,), min(max_delay, max(0, int(init_val))), device=device, dtype=torch.long)
            logits[torch.arange(nb_inputs, device=device), init_delays] += float(settings.get("input_delay_init_bias", 0.1))
    return logits


def _next_power_of_two(n: int) -> int:
    n = max(1, int(n))
    pow2 = 1
    while pow2 < n:
        pow2 <<= 1
    return pow2


class NeuromodulatorMapper(nn.Module):
    """Shared per-layer MLPs that map neuromodulator levels to parameter deltas/values."""

    def __init__(
        self,
        hidden_targets: int,
        output_targets: int,
        hidden_per_neuron: int,
        output_per_neuron: int,
        init_scale: float,
        cfg: Optional[Dict] = None,
    ):
        super().__init__()
        cfg = cfg or {}
        self.hidden_targets = max(0, int(hidden_targets))
        self.output_targets = max(0, int(output_targets))
        self.hidden_per_neuron = max(0, int(hidden_per_neuron))
        self.output_per_neuron = max(0, int(output_per_neuron))
        self.init_scale = float(init_scale)
        self.activation_kind = cfg.get("activation_kind")
        self.mapper_type = str(cfg.get("mapper_type", "mlp")).lower()
        self.hidden_activation = str(cfg.get("hidden_activation", "silu") or "silu").lower()
        self.flat_order = str(cfg.get("flat_order", "type_major") or "type_major").lower()
        if self.flat_order not in {"type_major", "target_major"}:
            self.flat_order = "type_major"
        self.init_identity = bool(cfg.get("init_identity", False))
        self.hidden_width = self._resolve_width(self.hidden_per_neuron, 5, cfg, "hidden_mlp_hidden")
        self.output_width = self._resolve_width(self.output_per_neuron, 2, cfg, "output_mlp_hidden")

        self.hidden_mlp: Optional[nn.Module] = None
        self.output_mlp: Optional[nn.Module] = None
        if self.hidden_per_neuron > 0 and self.hidden_targets > 0:
            self.hidden_mlp = self._build_mlp(self.hidden_per_neuron, self.hidden_width, 5)
        if self.output_per_neuron > 0 and self.output_targets > 0:
            self.output_mlp = self._build_mlp(self.output_per_neuron, self.output_width, 2)

    @staticmethod
    def _resolve_width(in_dim: int, out_dim: int, cfg: Optional[Dict], cfg_key: str) -> int:
        if cfg:
            override = cfg.get(cfg_key, None)
            if override is None:
                override = cfg.get("mlp_hidden", None)
            if override is not None:
                if cfg.get("mapper_hidden_exact"):
                    return max(1, int(override))
                # Ensure at least 4x the larger of in/out even when overridden.
                return max(1, 20 * max(in_dim, out_dim), int(override))
        base = max(in_dim, out_dim)
        return max(1, 20 * base)

    def _build_mlp(self, in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
        def _final_activation(kind: Optional[str]) -> nn.Module:
            k = (kind or "").lower()
            if k == "sigmoid":
                return nn.Sigmoid()
            if k == "tanh":
                return nn.Tanh()
            return nn.Identity()
        def _hidden_activation(kind: str) -> nn.Module:
            k = (kind or "silu").lower()
            if k == "relu":
                return nn.ReLU()
            if k == "gelu":
                return nn.GELU()
            if k == "leakyrelu":
                return nn.LeakyReLU(0.1)
            if k == "tanh":
                return nn.Tanh()
            if k == "none":
                return nn.Identity()
            return nn.SiLU()
        final_act = _final_activation(self.activation_kind)
        if self.mapper_type == "linear":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                final_act,
            )
            linear_layers = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
        else:
            hidden_act = _hidden_activation(self.hidden_activation)
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                hidden_act,
                nn.Linear(hidden_dim, hidden_dim),
                hidden_act,
                nn.Linear(hidden_dim, out_dim),
                final_act,
            )
            linear_layers = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
        # Init: default small normal for all layers; final layer bias set for neutral outputs.
        for idx, layer in enumerate(linear_layers):
            nn.init.normal_(layer.weight, mean=0.0, std=self.init_scale)
            bias_val = 0.0
            if idx == len(linear_layers) - 1 and (self.activation_kind or "").lower() == "sigmoid":
                # sigmoid(0)=0.5 midpoint
                bias_val = 0.0
            nn.init.constant_(layer.bias, bias_val)
        # Optional warm-start: identity mapping (logit passthrough) for linear mappers.
        if self.init_identity and linear_layers:
            # For MLP mappers, only apply identity init when hidden activation is linear (none),
            # otherwise the nonlinearity destroys the warm-start behavior.
            if self.mapper_type == "linear":
                lin = linear_layers[0]
                with torch.no_grad():
                    lin.weight.zero_()
                    lin.bias.zero_()
                    diag = min(out_dim, in_dim)
                    if diag > 0:
                        eye = torch.eye(diag, dtype=lin.weight.dtype, device=lin.weight.device)
                        lin.weight[:diag, :diag].copy_(eye)
            elif self.mapper_type == "mlp" and self.hidden_activation == "none" and len(linear_layers) >= 3:
                l0, l1, l2 = linear_layers[0], linear_layers[1], linear_layers[2]
                with torch.no_grad():
                    for lin in (l0, l1, l2):
                        lin.weight.zero_()
                        lin.bias.zero_()
                    # l0: embed input into first dims of hidden.
                    d0 = min(l0.out_features, l0.in_features)
                    if d0 > 0:
                        eye0 = torch.eye(d0, dtype=l0.weight.dtype, device=l0.weight.device)
                        l0.weight[:d0, :d0].copy_(eye0)
                    # l1: hidden identity.
                    d1 = min(l1.out_features, l1.in_features)
                    if d1 > 0:
                        eye1 = torch.eye(d1, dtype=l1.weight.dtype, device=l1.weight.device)
                        l1.weight[:d1, :d1].copy_(eye1)
                    # l2: project first dims back to output.
                    d2 = min(l2.out_features, l2.in_features)
                    if d2 > 0:
                        eye2 = torch.eye(d2, dtype=l2.weight.dtype, device=l2.weight.device)
                        l2.weight[:d2, :d2].copy_(eye2)
        return mlp

    @property
    def hidden_flat_dim(self) -> int:
        return self.hidden_targets * self.hidden_per_neuron

    @property
    def output_flat_dim(self) -> int:
        return self.output_targets * self.output_per_neuron

    @property
    def total_output_dim(self) -> int:
        return self.hidden_flat_dim + self.output_flat_dim

    def _split_levels(self, flat: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        offset = 0
        hidden_levels = output_levels = None
        if self.hidden_flat_dim > 0:
            hidden_slice = flat[:, offset:offset + self.hidden_flat_dim]
            if self.flat_order == "type_major":
                hidden_levels = hidden_slice.view(flat.size(0), self.hidden_per_neuron, self.hidden_targets).permute(0, 2, 1)
            else:
                hidden_levels = hidden_slice.view(flat.size(0), self.hidden_targets, self.hidden_per_neuron)
            offset += self.hidden_flat_dim
        if self.output_flat_dim > 0:
            out_slice = flat[:, offset:offset + self.output_flat_dim]
            if self.flat_order == "type_major":
                output_levels = out_slice.view(flat.size(0), self.output_per_neuron, self.output_targets).permute(0, 2, 1)
            else:
                output_levels = out_slice.view(flat.size(0), self.output_targets, self.output_per_neuron)
        return hidden_levels, output_levels

    def effects_from_levels(
        self,
        hidden_levels: Optional[torch.Tensor],
        output_levels: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ref = hidden_levels if hidden_levels is not None else output_levels
        batch = ref.size(0) if ref is not None else 0
        device = ref.device if ref is not None else None
        dtype = ref.dtype if ref is not None else None
        effects = {
            "alpha_1": torch.zeros((batch, self.hidden_targets), device=device, dtype=dtype),
            "beta_1": torch.zeros((batch, self.hidden_targets), device=device, dtype=dtype),
            "thr": torch.zeros((batch, self.hidden_targets), device=device, dtype=dtype),
            "reset": torch.zeros((batch, self.hidden_targets), device=device, dtype=dtype),
            "rest": torch.zeros((batch, self.hidden_targets), device=device, dtype=dtype),
            "alpha_2": torch.zeros((batch, self.output_targets), device=device, dtype=dtype),
            "beta_2": torch.zeros((batch, self.output_targets), device=device, dtype=dtype),
        }
        if hidden_levels is not None and self.hidden_mlp is not None:
            h = self.hidden_mlp(hidden_levels.reshape(-1, self.hidden_per_neuron))
            h = h.reshape(hidden_levels.size(0), self.hidden_targets, 5)
            effects["alpha_1"] = h[:, :, 0]
            effects["beta_1"] = h[:, :, 1]
            effects["thr"] = h[:, :, 2]
            effects["reset"] = h[:, :, 3]
            effects["rest"] = h[:, :, 4]
        if output_levels is not None and self.output_mlp is not None:
            o = self.output_mlp(output_levels.reshape(-1, self.output_per_neuron))
            o = o.reshape(output_levels.size(0), self.output_targets, 2)
            effects["alpha_2"] = o[:, :, 0]
            effects["beta_2"] = o[:, :, 1]
        return effects

    def effects_from_flat(self, flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden_levels, output_levels = self._split_levels(flat)
        return self.effects_from_levels(hidden_levels, output_levels)


class GroupLayout:
    def __init__(self, target_count: int, size: int, overlap: int, distribution: str, normal_std: float):
        self.target_count = max(1, int(target_count))
        self.size = max(1, min(int(size), self.target_count))
        self.overlap = max(0, min(int(overlap), self.size - 1))
        self.distribution = distribution.lower() if distribution else "uniform"
        if self.distribution not in GROUP_DISTRIBUTIONS:
            self.distribution = "uniform"
        self.normal_std = float(normal_std if normal_std is not None else 1.0)
        self.enabled = (self.size > 1) or (self.overlap > 0)
        self._expand_divisor = None
        if not self.enabled:
            self.group_count = self.target_count
            self.forward = None
            self.backward = None
            return
        is_normal = self.distribution == "normal"
        stride = max(1, self.size if is_normal else (self.size - self.overlap))
        groups = max(1, math.ceil(self.target_count / stride))
        self.group_count = groups
        forward = np.zeros((groups, self.target_count), dtype=np.float32)
        for g in range(groups):
            if self.distribution == "uniform":
                start = (g * stride) % self.target_count
                for k in range(self.size):
                    idx = (start + k) % self.target_count
                    forward[g, idx] += 1.0
                continue

            # Normal: soft Gaussian falloff over all targets. Overlap is redundant here; spacing uses size.
            center = (g * stride + (self.size - 1) / 2.0) % self.target_count
            denom = max(self.normal_std, 1e-6)
            for idx in range(self.target_count):
                dist = abs(idx - center)
                if dist > self.target_count / 2:
                    dist = self.target_count - dist
                weight = math.exp(-0.5 * (dist / denom) ** 2)
                if weight >= GROUP_NORMAL_CUTOFF:
                    forward[g, idx] += weight
        forward_tensor = torch.from_numpy(forward)
        if self.distribution == "uniform":
            self._expand_divisor = forward_tensor.sum(dim=0, keepdim=True).clamp_min(1e-6)
        norm = forward_tensor.sum(dim=1, keepdim=True).clamp_min(1e-6)
        backward = forward_tensor / norm
        self.forward = forward_tensor
        self.backward = backward

    def project(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.forward is None:
            return values
        weights = self.forward.t().to(device=values.device, dtype=values.dtype)
        return torch.matmul(values, weights)

    def enforce(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.forward is None or self.backward is None:
            return values
        back = self.backward.to(device=values.device, dtype=values.dtype)
        fwd = self.forward.to(device=values.device, dtype=values.dtype)
        group_vals = torch.matmul(values, back.t())
        return torch.matmul(group_vals, fwd)

    def expand(self, grouped: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.forward is None:
            return grouped
        weights = self.forward.to(device=grouped.device, dtype=grouped.dtype)
        expanded = torch.matmul(grouped, weights)
        if self._expand_divisor is not None and self.distribution == "uniform":
            divisor = self._expand_divisor.to(device=expanded.device, dtype=expanded.dtype)
            expanded = expanded / divisor
        return expanded


def _nearest_rect_grid(n: int) -> Tuple[int, int]:
    n = max(1, int(n))
    rows = max(1, int(math.floor(math.sqrt(n))))
    cols = max(1, int(math.ceil(n / rows)))
    while rows * cols < n:
        rows += 1
        cols = max(1, int(math.ceil(n / rows)))
    return rows, cols


class WeightAxisLayout:
    def __init__(self, target_count: int, size: int, overlap: int, distribution: str, normal_std: float, grid_dim: str):
        self.target_count = max(1, int(target_count))
        self.size = max(1, int(size))
        self.overlap = max(0, min(int(overlap), self.size - 1))
        self.distribution = distribution.lower() if distribution else "uniform"
        if self.distribution not in GROUP_DISTRIBUTIONS:
            self.distribution = "uniform"
        self.normal_std = float(normal_std if normal_std is not None else 1.0)
        self.grid_dim = str(grid_dim or "1d").lower()
        if self.grid_dim not in WEIGHT_GRID_CHOICES:
            self.grid_dim = "1d"
        self.enabled = (self.size > 1) or (self.overlap > 0) or (self.grid_dim == "2d")
        self._expand_divisor = None
        if self.grid_dim == "1d":
            base = GroupLayout(self.target_count, self.size, self.overlap, self.distribution, self.normal_std)
            self.group_count = base.group_count
            self.forward = base.forward
            self.backward = base.backward
            self._expand_divisor = base._expand_divisor
            self.enabled = base.enabled
            return
        self.rows, self.cols = _nearest_rect_grid(self.target_count)
        self._build_2d()

    def _build_2d(self):
        is_normal = self.distribution == "normal"
        stride = max(1, self.size if is_normal else (self.size - self.overlap))
        row_starts = list(range(0, self.rows, stride)) or [0]
        col_starts = list(range(0, self.cols, stride)) or [0]
        groups = max(1, len(row_starts) * len(col_starts))
        forward = np.zeros((groups, self.target_count), dtype=np.float32)
        g = 0
        for r0 in row_starts:
            for c0 in col_starts:
                if self.distribution == "uniform":
                    for dr in range(self.size):
                        for dc in range(self.size):
                            r = r0 + dr
                            c = c0 + dc
                            idx = r * self.cols + c
                            if r < self.rows and c < self.cols and idx < self.target_count:
                                forward[g, idx] += 1.0
                else:
                    cr = r0 + (self.size - 1) / 2.0
                    cc = c0 + (self.size - 1) / 2.0
                    denom = max(self.normal_std, 1e-6)
                    for idx in range(self.target_count):
                        r = idx // self.cols
                        c = idx % self.cols
                        dist = math.sqrt((float(r) - cr) ** 2 + (float(c) - cc) ** 2)
                        weight = math.exp(-0.5 * (dist / denom) ** 2)
                        if weight >= GROUP_NORMAL_CUTOFF:
                            forward[g, idx] += weight
                g += 1
        nonempty = forward.sum(axis=1) > 0
        if nonempty.any():
            forward = forward[nonempty]
        forward_tensor = torch.from_numpy(forward)
        if self.distribution == "uniform":
            self._expand_divisor = forward_tensor.sum(dim=0, keepdim=True).clamp_min(1e-6)
        norm = forward_tensor.sum(dim=1, keepdim=True).clamp_min(1e-6)
        self.group_count = int(forward_tensor.size(0))
        self.forward = forward_tensor
        self.backward = forward_tensor / norm

    def project(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.forward is None:
            return values
        weights = self.forward.t().to(device=values.device, dtype=values.dtype)
        return torch.matmul(values, weights)

    def expand(self, grouped: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.forward is None:
            return grouped
        weights = self.forward.to(device=grouped.device, dtype=grouped.dtype)
        expanded = torch.matmul(grouped, weights)
        if self._expand_divisor is not None and self.distribution == "uniform":
            divisor = self._expand_divisor.to(device=expanded.device, dtype=expanded.dtype)
            expanded = expanded / divisor
        return expanded


class WeightMatrixLayout:
    def __init__(
        self,
        name: str,
        source_count: int,
        target_count: int,
        scope: str,
        size: int,
        overlap: int,
        distribution: str,
        normal_std: float,
        grid_dim: str,
    ):
        self.name = name
        self.source_count = max(1, int(source_count))
        self.target_count = max(1, int(target_count))
        self.scope = str(scope or "block").lower()
        if self.scope not in WEIGHT_SCOPE_CHOICES:
            self.scope = "block"
        self.source_layout = WeightAxisLayout(self.source_count, size, overlap, distribution, normal_std, grid_dim)
        self.target_layout = WeightAxisLayout(self.target_count, size, overlap, distribution, normal_std, grid_dim)
        if self.scope == "source":
            self.effect_dim = self.source_layout.group_count
        elif self.scope == "target":
            self.effect_dim = self.target_layout.group_count
        else:
            self.effect_dim = self.source_layout.group_count * self.target_layout.group_count

    def project_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        if matrix.dim() == 2:
            matrix = matrix.unsqueeze(0)
        if self.scope == "source":
            src_vals = matrix.mean(dim=2)
            return self.source_layout.project(src_vals)
        if self.scope == "target":
            tgt_vals = matrix.mean(dim=1)
            return self.target_layout.project(tgt_vals)
        src_back = self.source_layout.backward.to(device=matrix.device, dtype=matrix.dtype)
        tgt_back = self.target_layout.backward.to(device=matrix.device, dtype=matrix.dtype)
        src_grouped = torch.einsum("bst,gs->bgt", matrix, src_back)
        grouped = torch.einsum("bgt,ht->bgh", src_grouped, tgt_back)
        return grouped.reshape(matrix.size(0), -1)

    def expand_effect(self, values: torch.Tensor) -> torch.Tensor:
        batch = values.size(0)
        if self.scope == "source":
            src_vals = self.source_layout.expand(values)
            return src_vals.unsqueeze(2).expand(batch, self.source_count, self.target_count)
        if self.scope == "target":
            tgt_vals = self.target_layout.expand(values)
            return tgt_vals.unsqueeze(1).expand(batch, self.source_count, self.target_count)
        grouped = values.view(batch, self.source_layout.group_count, self.target_layout.group_count)
        tgt_flat = grouped.reshape(batch * self.source_layout.group_count, self.target_layout.group_count)
        tgt_expanded = self.target_layout.expand(tgt_flat).view(batch, self.source_layout.group_count, self.target_count)
        src_flat = tgt_expanded.permute(0, 2, 1).reshape(batch * self.target_count, self.source_layout.group_count)
        src_expanded = self.source_layout.expand(src_flat).view(batch, self.target_count, self.source_count)
        return src_expanded.permute(0, 2, 1)


def _weight_matrix_shapes(settings: Dict) -> Dict[str, Tuple[int, int]]:
    return {
        "w1": (int(settings["nb_inputs"]), int(settings["nb_hidden"])),
        "v1": (int(settings["nb_hidden"]), int(settings["nb_hidden"])),
        "w2": (int(settings["nb_hidden"]), int(settings["nb_outputs"])),
    }


def _normalize_weight_targets(value) -> List[str]:
    names = parse_str_list(value)
    if names is None:
        if value is None:
            names = ["w1", "v1", "w2"]
        else:
            names = [str(value)]
    out: List[str] = []
    for name in names:
        key = str(name).strip().lower()
        if key in {"all", "*"}:
            return ["w1", "v1", "w2"]
        if key in WEIGHT_PARAM_NAMES and key not in out:
            out.append(key)
    return out


def _weight_mod_cfg(settings: Dict) -> Dict:
    return settings.get("weight_mod_cfg", {}) or {}


def _weight_mod_enabled(settings: Dict) -> bool:
    cfg = _weight_mod_cfg(settings)
    return bool(cfg.get("enable", False)) and bool(cfg.get("targets"))


def _get_weight_layout(settings: Dict, name: str) -> Optional[WeightMatrixLayout]:
    if not _weight_mod_enabled(settings):
        return None
    cfg = _weight_mod_cfg(settings)
    if name not in cfg.get("targets", []):
        return None
    shapes = _weight_matrix_shapes(settings)
    if name not in shapes:
        return None
    cache = settings.setdefault("_weight_layout_cache", {})
    source_count, target_count = shapes[name]
    key = (
        name,
        source_count,
        target_count,
        cfg.get("scope", "block"),
        cfg.get("size", 1),
        cfg.get("overlap", 0),
        cfg.get("distribution", "uniform"),
        cfg.get("normal_std", 1.0),
        cfg.get("grid_dim", "1d"),
    )
    layout = cache.get(key)
    if layout is None:
        layout = WeightMatrixLayout(
            name,
            source_count,
            target_count,
            cfg.get("scope", "block"),
            int(cfg.get("size", 1)),
            int(cfg.get("overlap", 0)),
            str(cfg.get("distribution", "uniform")),
            float(cfg.get("normal_std", 1.0)),
            str(cfg.get("grid_dim", "1d")),
        )
        cache[key] = layout
    return layout


def _weight_size_overrides(settings: Dict) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    if not _weight_mod_enabled(settings):
        return overrides
    for name in _weight_mod_cfg(settings).get("targets", []):
        layout = _get_weight_layout(settings, name)
        if layout is not None:
            overrides[name] = int(layout.effect_dim)
    return overrides


def _uniform_group_layout(layout: Optional[GroupLayout]) -> Optional[GroupLayout]:
    if layout is None or not layout.enabled:
        return layout
    if layout.distribution == "uniform":
        return layout
    return GroupLayout(layout.target_count, layout.size, layout.overlap, "uniform", layout.normal_std)


def _group_layer_cfg(settings: Dict, layer_index: int) -> Dict:
    base = settings.get("group_cfg")
    if not base:
        return {}
    def _select(key, default):
        pair = base.get(key)
        if isinstance(pair, (list, tuple)):
            if len(pair) > layer_index:
                return pair[layer_index]
            return pair[0]
        return pair if pair is not None else default
    size = max(1, int(_select("size", 1)))
    overlap = max(0, int(_select("overlap", 0)))
    enable_override = _select("enable", None)
    enabled = bool(enable_override) if enable_override is not None else (size > 1 or overlap > 0)
    return {
        "enable": enabled,
        "size": size,
        "overlap": overlap,
        "distribution": str(_select("distribution", "uniform")),
        "normal_std": float(_select("normal_std", 1.0)),
    }


def _get_group_layout(settings: Dict, layer_index: int, target_count: int) -> Optional[GroupLayout]:
    cfg = _group_layer_cfg(settings, layer_index)
    if not cfg.get("enable"):
        return None
    cache = settings.setdefault("_group_layout_cache", {})
    key = (layer_index, target_count, cfg["size"], cfg["overlap"], cfg["distribution"], cfg["normal_std"])
    layout = cache.get(key)
    if layout is None:
        layout = GroupLayout(target_count, cfg["size"], cfg["overlap"], cfg["distribution"], cfg["normal_std"])
        cache[key] = layout
    return layout


def _ann_out_size_overrides(settings: Dict) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    nb_hidden = settings["nb_hidden"]
    nb_outputs = settings["nb_outputs"]
    hidden_layout = _get_group_layout(settings, 0, nb_hidden)
    output_layout = _get_group_layout(settings, 1, nb_outputs)
    hidden_groups = hidden_layout.group_count if hidden_layout else nb_hidden
    out_groups = output_layout.group_count if output_layout else nb_outputs
    for name in ("alpha_1", "beta_1", "thr", "reset", "rest"):
        overrides[name] = hidden_groups
    overrides["alpha_2"] = out_groups
    overrides["beta_2"] = out_groups
    return overrides


def _ann_in_size_overrides(settings: Dict, substitution_mode: bool) -> Dict[str, int]:
    if not substitution_mode:
        return {}
    overrides: Dict[str, int] = {}
    nb_hidden = settings["nb_hidden"]
    nb_outputs = settings["nb_outputs"]
    hidden_layout = _get_group_layout(settings, 0, nb_hidden)
    output_layout = _get_group_layout(settings, 1, nb_outputs)
    hidden_groups = hidden_layout.group_count if hidden_layout else nb_hidden
    out_groups = output_layout.group_count if output_layout else nb_outputs
    for name in ("alpha_1", "beta_1", "thr", "reset", "rest"):
        overrides[name] = hidden_groups
    overrides["alpha_2"] = out_groups
    overrides["beta_2"] = out_groups
    return overrides


def _select_fraction_indices(count: int, frac: float, generator: Optional[torch.Generator], ensure_min: bool = True) -> torch.Tensor:
    count = max(0, int(count))
    if count == 0:
        return torch.empty((0,), dtype=torch.long)
    frac = float(frac)
    if frac >= 1.0:
        idx = torch.arange(count, dtype=torch.long)
    elif frac <= 0.0:
        idx = torch.empty((0,), dtype=torch.long)
    else:
        mask = torch.rand(count, generator=generator) < frac
        idx = mask.nonzero(as_tuple=False).flatten().to(dtype=torch.long)
    if ensure_min and idx.numel() == 0 and frac > 0.0:
        if generator is None:
            pick = torch.randint(0, count, (1,))
        else:
            pick = torch.randint(0, count, (1,), generator=generator)
        idx = pick.to(dtype=torch.long)
    return idx


def _build_fixed_mod_mask(
    settings: Dict,
    hidden_layout: Optional['GroupLayout'],
    output_layout: Optional['GroupLayout']
) -> Optional[Dict[str, torch.Tensor]]:
    cfg = settings.get("mod_mask_cfg", {}) or {}
    zero_fallback = bool(cfg.get("zero_fallback", False))
    if not cfg.get("fixed_enable"):
        return None
    nm_cfg = settings.get("nm_cfg", {}) or {}
    nm_neuron_frac_enable = bool(nm_cfg.get("neuron_fraction_enable", False))
    nm_param_frac_enable = bool(nm_cfg.get("param_fraction_enable", False))
    if not (nm_neuron_frac_enable or nm_param_frac_enable):
        return None
    cached = settings.get("_mod_fixed_mask")
    if cached is not None:
        return cached

    def _pair(val, default):
        if isinstance(val, (list, tuple)):
            if len(val) >= 2:
                return (float(val[0]), float(val[1]))
            if len(val) == 1:
                return (float(val[0]), float(val[0]))
        try:
            v = float(val)
            return (v, v)
        except Exception:
            return default

    nm_neuron_frac = _pair(nm_cfg.get("neuron_fraction", (1.0, 1.0)), (1.0, 1.0))
    nm_param_frac_map = _parse_param_fraction_map(nm_cfg.get("param_fraction", None), default=1.0)
    nb_hidden = int(settings.get("nb_hidden", 0))
    nb_outputs = int(settings.get("nb_outputs", 0))
    hidden_target = hidden_layout.group_count if hidden_layout else nb_hidden
    output_target = output_layout.group_count if output_layout else nb_outputs
    hidden_param_fracs = [float(nm_param_frac_map.get(name, 1.0)) for name in HIDDEN_PARAM_NAMES]
    output_param_fracs = [float(nm_param_frac_map.get(name, 1.0)) for name in OUTPUT_PARAM_NAMES]
    allow_empty_hidden = (nm_neuron_frac_enable and float(nm_neuron_frac[0]) <= 0.0) or (
        nm_param_frac_enable and max(hidden_param_fracs) <= 0.0
    )
    allow_empty_output = (nm_neuron_frac_enable and float(nm_neuron_frac[1]) <= 0.0) or (
        nm_param_frac_enable and max(output_param_fracs) <= 0.0
    )
    if zero_fallback:
        allow_empty_hidden = False
        allow_empty_output = False

    gen = None
    seed = cfg.get("fixed_seed")
    if seed is None:
        seed = cfg.get("seed")
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))

    if nm_neuron_frac_enable:
        hidden_neuron_idx = _select_fraction_indices(hidden_target, nm_neuron_frac[0], gen)
        output_neuron_idx = _select_fraction_indices(output_target, nm_neuron_frac[1], gen)
    else:
        hidden_neuron_idx = torch.arange(hidden_target, dtype=torch.long)
        output_neuron_idx = torch.arange(output_target, dtype=torch.long)

    def _select_param_mask_indices(base_idx: torch.Tensor, frac: float) -> torch.Tensor:
        if not nm_param_frac_enable:
            return base_idx
        if base_idx.numel() == 0:
            return base_idx
        mask = torch.rand(base_idx.numel(), generator=gen) < float(frac)
        idx = base_idx[mask].to(dtype=torch.long)
        if idx.numel() == 0 and frac > 0.0:
            if gen is None:
                pick = torch.randint(0, base_idx.numel(), (1,))
            else:
                pick = torch.randint(0, base_idx.numel(), (1,), generator=gen)
            idx = base_idx[pick].to(dtype=torch.long)
        return idx

    hidden_param_mask_idx = {
        name: _select_param_mask_indices(hidden_neuron_idx, nm_param_frac_map.get(name, 1.0))
        for name in HIDDEN_PARAM_NAMES
    }
    output_param_mask_idx = {
        name: _select_param_mask_indices(output_neuron_idx, nm_param_frac_map.get(name, 1.0))
        for name in OUTPUT_PARAM_NAMES
    }

    def _ensure_io_indices(
        mask_idx: torch.Tensor,
        base_idx: torch.Tensor,
        target_count: int,
        allow_empty: bool = False
    ) -> torch.Tensor:
        if mask_idx.numel() > 0:
            return mask_idx
        if allow_empty:
            return mask_idx
        if base_idx.numel() > 0:
            return base_idx
        if target_count > 0:
            return torch.arange(target_count, dtype=torch.long)
        return torch.empty((0,), dtype=torch.long)

    hidden_param_idx = {
        name: _ensure_io_indices(
            hidden_param_mask_idx[name],
            hidden_neuron_idx,
            hidden_target,
            allow_empty=allow_empty_hidden,
        )
        for name in HIDDEN_PARAM_NAMES
    }
    output_param_idx = {
        name: _ensure_io_indices(
            output_param_mask_idx[name],
            output_neuron_idx,
            output_target,
            allow_empty=allow_empty_output,
        )
        for name in OUTPUT_PARAM_NAMES
    }

    def _union_indices(indices: List[torch.Tensor], target_count: int, allow_empty: bool) -> torch.Tensor:
        if not indices or target_count <= 0:
            return torch.empty((0,), dtype=torch.long)
        non_empty = [idx for idx in indices if idx.numel() > 0]
        if not non_empty:
            if allow_empty:
                return torch.empty((0,), dtype=torch.long)
            return _select_fraction_indices(target_count, 1.0, gen)
        merged = torch.cat(non_empty, dim=0)
        if merged.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        merged = torch.unique(merged)
        merged, _ = torch.sort(merged)
        if merged.numel() == 0 and target_count > 0:
            if allow_empty:
                return torch.empty((0,), dtype=torch.long)
            merged = _select_fraction_indices(target_count, 1.0, gen)
        return merged.to(dtype=torch.long)

    hidden_union_idx = _union_indices(list(hidden_param_idx.values()), hidden_target, allow_empty_hidden)
    output_union_idx = _union_indices(list(output_param_idx.values()), output_target, allow_empty_output)

    def _indices_to_mask(count: int, idx: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((max(0, int(count)),), dtype=torch.float32)
        if idx.numel() > 0:
            mask[idx] = 1.0
        return mask

    param_masks_target: Dict[str, torch.Tensor] = {}
    for name, idx in {**hidden_param_mask_idx, **output_param_mask_idx}.items():
        count = hidden_target if name in HIDDEN_PARAM_NAMES else output_target
        param_masks_target[name] = _indices_to_mask(count, idx)

    def _mask_in_union(union_idx: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((union_idx.numel(),), dtype=torch.float32)
        if union_idx.numel() == 0 or group_idx.numel() == 0:
            return mask
        positions = {int(v): i for i, v in enumerate(union_idx.tolist())}
        for v in group_idx.tolist():
            pos = positions.get(int(v))
            if pos is not None:
                mask[pos] = 1.0
        return mask

    param_masks_union: Dict[str, torch.Tensor] = {}
    for name, idx in hidden_param_mask_idx.items():
        param_masks_union[name] = _mask_in_union(hidden_union_idx, idx)
    for name, idx in output_param_mask_idx.items():
        param_masks_union[name] = _mask_in_union(output_union_idx, idx)

    flat_inputs = bool(cfg.get("fixed_flat_inputs", False))
    flat_hidden_idx = None
    flat_output_idx = None
    if flat_inputs and nm_neuron_frac_enable:
        flat_hidden_idx = _select_fraction_indices(nb_hidden, nm_neuron_frac[0], gen)
        flat_output_idx = _select_fraction_indices(nb_outputs, nm_neuron_frac[1], gen)

    mask_cfg = {
        "hidden_target": int(hidden_target),
        "output_target": int(output_target),
        "hidden_neuron_idx": hidden_neuron_idx,
        "output_neuron_idx": output_neuron_idx,
        "hidden_param_idx": hidden_param_idx,
        "output_param_idx": output_param_idx,
        "hidden_union_idx": hidden_union_idx,
        "output_union_idx": output_union_idx,
        "param_masks_target": param_masks_target,
        "param_masks_union": param_masks_union,
        "flat_inputs": flat_inputs,
        "flat_hidden_idx": flat_hidden_idx,
        "flat_output_idx": flat_output_idx,
    }
    settings["_mod_fixed_mask"] = mask_cfg
    return mask_cfg


class ParamTimescaleController(nn.Module):
    def __init__(
        self,
        base_interval: int,
        names: List[str],
        dist: str = "fixed",
        scale: float = 0.0,
        std: float = 0.0,
        seed: Optional[int] = None,
        trainable: bool = True,
    ):
        super().__init__()
        self.base_interval = max(1, int(base_interval))
        self.names = list(names)
        self.dist = dist.lower() if isinstance(dist, str) else "fixed"
        self.scale = max(0.0, float(scale))
        self.std = max(0.0, float(std))
        rng = np.random.default_rng(seed) if seed is not None else np.random
        init = np.zeros(len(self.names), dtype=np.float32)
        if self.dist == "uniform" and self.scale > 0:
            init = rng.uniform(0.0, self.scale, size=len(self.names))
        elif self.dist == "normal" and (self.std > 0 or self.scale > 0):
            stdev = self.std if self.std > 0 else self.scale
            init = rng.normal(loc=0.0, scale=stdev, size=len(self.names))
            init = np.clip(init, 0.0, None)
        self.offsets = nn.Parameter(torch.from_numpy(init).float(), requires_grad=trainable)

    def intervals(self) -> Dict[str, torch.Tensor]:
        deltas = F.softplus(self.offsets)
        base = torch.tensor(float(self.base_interval), device=deltas.device, dtype=deltas.dtype)
        intervals = base + deltas
        # Straight-through rounding: forward uses round, backward flows through the continuous value.
        ste = (torch.round(intervals) - intervals).detach() + intervals
        ste = torch.clamp(ste, min=base)
        return {name: ste[idx] for idx, name in enumerate(self.names)}


class ParamSmoothingController(nn.Module):
    """
    Learnable per-parameter mixing fractions (0-1) applied each timestep toward targets.
    """

    def __init__(
        self,
        names: List[str],
        hidden_size: int,
        output_size: int,
        tau_init: float = PARAM_SMOOTH_TAU_INIT_DEFAULT,
        tau_min: float = PARAM_SMOOTH_TAU_MIN_DEFAULT,
        trainable: bool = True,
    ):
        super().__init__()
        self.names = list(names)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.mix_min = 0.0
        init = float(tau_init)
        init = min(max(init, self.mix_min), 0.9999)

        def _logit_scalar(x: float) -> float:
            eps = 1e-4
            x = max(min(float(x), 1 - eps), eps)
            return math.log(x / (1.0 - x))

        params = {}
        for name in self.names:
            dim = self.hidden_size if name in ("alpha_1", "beta_1", "thr", "reset", "rest") else self.output_size
            params[name] = nn.Parameter(torch.full((1, dim), float(_logit_scalar(init))), requires_grad=trainable)
        self.raw_mix = nn.ParameterDict(params)

    def mixes(self) -> Dict[str, torch.Tensor]:
        mixes = {}
        for name, tensor in self.raw_mix.items():
            mix = torch.sigmoid(tensor)
            mix = torch.clamp(mix, min=0.0, max=1.0)
            mixes[name] = mix
        return mixes

    def mixing_factors(self) -> Dict[str, torch.Tensor]:
        return self.mixes()


def _snn_shape_str(state: Optional[Dict[str, torch.nn.Parameter]]) -> Optional[str]:
    if not state:
        return None
    w1 = state.get("w1")
    w2 = state.get("w2")
    if w1 is None or w2 is None:
        return None
    if not hasattr(w1, "shape") or not hasattr(w2, "shape"):
        return None
    if w1.dim() < 2 or w2.dim() < 2:
        return None
    try:
        in_dim, hidden_dim = int(w1.shape[0]), int(w1.shape[1])
        out_dim = int(w2.shape[1])
    except Exception:
        return None
    return f"{in_dim}-{hidden_dim}-{out_dim}"


def _mlp_shape_str(mlp: Optional[nn.Module]) -> Optional[str]:
    if mlp is None:
        return None
    arch = getattr(mlp, "arch", None)
    if arch in {"rnn", "lstm"}:
        input_size = getattr(mlp, "input_size", None)
        output_size = getattr(mlp, "output_size", None)
        hidden_size = getattr(mlp, "rnn_hidden_size", None)
        num_layers = getattr(mlp, "rnn_num_layers", None)
        if None not in (input_size, output_size, hidden_size, num_layers):
            arch_name = arch.upper()
            return f"{arch_name}({int(input_size)}->{int(hidden_size)}x{int(num_layers)}->{int(output_size)})"
    linear_layers: List[nn.Module] = list(getattr(mlp, "_linear_layers", []) or [])
    if not linear_layers:
        linear_layers = [mod for mod in mlp.modules() if isinstance(mod, nn.Linear)]
    if not linear_layers:
        spiking_layers = getattr(mlp, "layers", None)
        if spiking_layers:
            linear_layers = [layer for layer in spiking_layers if hasattr(layer, "out_dim")]
    if not linear_layers:
        return None
    input_size = getattr(mlp, "input_size", None)
    if input_size is None:
        first = linear_layers[0]
        input_size = getattr(first, "in_features", None)
        if input_size is None:
            input_size = getattr(first, "in_dim", None)
    if input_size is None:
        return None
    dims: List[int] = [int(input_size)]
    for layer in linear_layers:
        out_features = getattr(layer, "out_features", None)
        if out_features is None:
            out_features = getattr(layer, "out_dim", None)
        if out_features is None:
            return None
        dims.append(int(out_features))
    return "-".join(str(d) for d in dims)


def _nm_shape_str(modulator) -> Optional[str]:
    mapper = getattr(modulator, "nm_mapper", None)
    if mapper is None:
        return None
    parts = []
    if getattr(mapper, "hidden_mlp", None) is not None and mapper.hidden_per_neuron > 0 and mapper.hidden_targets > 0:
        h_shape = _mlp_shape_str(mapper.hidden_mlp) or f"{mapper.hidden_per_neuron}->{mapper.hidden_width}->5"
        parts.append(f"H:{h_shape} (targets={mapper.hidden_targets})")
    if getattr(mapper, "output_mlp", None) is not None and mapper.output_per_neuron > 0 and mapper.output_targets > 0:
        o_shape = _mlp_shape_str(mapper.output_mlp) or f"{mapper.output_per_neuron}->{mapper.output_width}->2"
        parts.append(f"O:{o_shape} (targets={mapper.output_targets})")
    return " | ".join(parts) if parts else None


def _slice_len(slc: Optional[slice]) -> int:
    if slc is None:
        return 0
    try:
        return int(slc.stop) - int(slc.start)
    except Exception:
        return 0


def _ordered_slice_sizes(slices: Optional[Dict[str, slice]], order: List[str]) -> List[Tuple[str, int]]:
    if not slices:
        return []
    items: List[Tuple[str, int]] = []
    for name in order:
        slc = slices.get(name)
        if slc is None:
            continue
        size = _slice_len(slc)
        items.append((name, size))
    return items


def _count_modulated_neurons(
    layout: Optional['GroupLayout'],
    union_idx: Optional[torch.Tensor],
    target_count: int
) -> Optional[int]:
    if union_idx is None:
        return None
    idx = union_idx.detach().cpu()
    if layout is None or not getattr(layout, "enabled", False) or getattr(layout, "forward", None) is None:
        return int(idx.numel())
    if idx.numel() == 0:
        return 0
    forward = layout.forward
    try:
        selected = forward.index_select(0, idx)
    except Exception:
        return int(idx.numel())
    mask = selected.sum(dim=0) > 0
    return int(mask.sum().item())


def format_param_stats(
    state: Optional[Dict[str, torch.nn.Parameter]],
    modulator: Optional[nn.Module] = None,
    prefix: str = "",
    settings: Optional[Dict[str, Any]] = None,
) -> str:
    snn_trainable, snn_frozen = _count_snn_params(state)
    snn_total = snn_trainable + snn_frozen
    lines = []
    lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + "SNN trainable: {0:,} | SNN frozen: {1:,} | SNN total: {2:,}".format(
        snn_trainable, snn_frozen, snn_total
    ))
    snn_shape = _snn_shape_str(state)
    if snn_shape:
        lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + f"SNN shape: {snn_shape}")
    if modulator is not None:
        mod_trainable, mod_frozen = _count_module_params(modulator)
        mod_total = mod_trainable + mod_frozen
        mod_line = "Modulator trainable: {0:,} | Modulator frozen: {1:,} | Modulator total: {2:,}".format(
            mod_trainable, mod_frozen, mod_total
        )
        if snn_total > 0 and mod_total > 0:
            mod_pct = 100.0 * mod_total / snn_total
            mod_line += f" | Modulator vs SNN: {mod_pct:.2f}%"
        arch = getattr(modulator, "arch", None)
        if arch:
            mod_line += f" | Arch: {arch.upper()}"
        lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + mod_line)

        mod_shape = None
        if hasattr(modulator, "_linear_layers") or hasattr(modulator, "layers"):
            mod_shape = _mlp_shape_str(modulator)
        if mod_shape:
            lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + f"Modulator shape: {mod_shape}")
            in_slices = getattr(modulator, "in_slices", None)
            in_items = _ordered_slice_sizes(
                in_slices,
                ["alpha_1", "beta_1", "thr", "reset", "rest", "alpha_2", "beta_2"] + WEIGHT_PARAM_NAMES + ["in_flat", "hid_flat", "out_flat"],
            )
            if in_items:
                in_total = sum(size for _, size in in_items)
                parts = ", ".join(f"{name}={size}" for name, size in in_items)
                lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + f"Modulator input: {parts} (total={in_total})")
                if settings is not None:
                    hid_flat_size = 0
                    for name, size in in_items:
                        if name == "hid_flat":
                            hid_flat_size = size
                            break
                    hid_flat_enabled = bool(settings.get("mlp_in_mask", {}).get("hid_flat", True))
                    mod_hid_flat_group = bool(settings.get("mod_hid_flat_group", False))
                    mod_hid_flat_mod_only = bool(settings.get("mod_hid_flat_modulated_only", False))
                    lines.append(
                        "[ParamStats] "
                        + (f"{prefix.strip()} " if prefix else "")
                        + f"Hid_flat dim: {hid_flat_size} (enabled={hid_flat_enabled}, group={mod_hid_flat_group}, mod_only={mod_hid_flat_mod_only})"
                    )

            out_items: List[Tuple[str, int]] = []
            if getattr(modulator, "use_neuromodulators", False) and getattr(modulator, "nm_out_slices", None):
                nm_slices = getattr(modulator, "nm_out_slices", {})
                out_items = [(f"nm_{name}", size) for name, size in _ordered_slice_sizes(nm_slices, ["hidden", "output"])]
                weight_slices = getattr(modulator, "weight_out_slices", None)
                out_items.extend(_ordered_slice_sizes(weight_slices, WEIGHT_PARAM_NAMES))
            else:
                out_slices = getattr(modulator, "out_slices", None)
                out_items = _ordered_slice_sizes(
                    out_slices,
                    ["alpha_1", "beta_1", "thr", "reset", "rest", "alpha_2", "beta_2"] + WEIGHT_PARAM_NAMES,
                )
            if out_items:
                out_total = sum(size for _, size in out_items)
                parts = ", ".join(f"{name}={size}" for name, size in out_items)
                lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + f"Modulator output: {parts} (total={out_total})")

            if settings is not None:
                mod_mask = settings.get("_mod_fixed_mask")
                if mod_mask is not None:
                    nb_hidden = int(settings.get("nb_hidden", 0))
                    nb_outputs = int(settings.get("nb_outputs", 0))
                    hidden_layout = _get_group_layout(settings, 0, nb_hidden)
                    output_layout = _get_group_layout(settings, 1, nb_outputs)
                    hidden_count = _count_modulated_neurons(hidden_layout, mod_mask.get("hidden_union_idx"), nb_hidden)
                    output_count = _count_modulated_neurons(output_layout, mod_mask.get("output_union_idx"), nb_outputs)
                    if hidden_count is not None and output_count is not None:
                        lines.append(
                            "[ParamStats] " + (f"{prefix.strip()} " if prefix else "")
                            + f"Modulated neurons: hidden={hidden_count}/{nb_hidden}, output={output_count}/{nb_outputs}"
                        )
        nm_shape = _nm_shape_str(modulator)
        if nm_shape:
            lines.append("[ParamStats] " + (f"{prefix.strip()} " if prefix else "") + f"NM MLP: {nm_shape}")
    return "\n".join(lines)

# -------------------------
# Run header logging
# -------------------------
def _to_builtin(v):
    import numpy as _np
    if isinstance(v, _np.generic): return v.item()
    if isinstance(v, Path): return str(v)
    if isinstance(v, (list, tuple)): return type(v)(_to_builtin(x) for x in v)
    if isinstance(v, dict): return {str(k): _to_builtin(vv) for k, vv in v.items()}
    return v

def _config_value(v):
    import numpy as _np
    if isinstance(v, _np.generic):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, torch.Tensor):
        return {"tensor_shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_config_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _config_value(vv) for k, vv in v.items()}
    return str(v)

def _config_snapshot(settings: Dict[str, Any]) -> Dict[str, Any]:
    skip_keys = {"mlp_init_state_dict"}
    snapshot = {}
    for key, value in settings.items():
        if key in skip_keys or str(key).startswith("_"):
            continue
        snapshot[str(key)] = _config_value(value)
    return snapshot

def write_run_config(run_dir: Path, settings: Dict[str, Any], extras: Dict[str, Any]) -> Path:
    config_path = run_dir / "run_config.json"
    payload = {
        "settings": _config_snapshot(settings),
        "controls": _config_value(extras or {}),
    }
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return config_path

def _section(title, d):
    lines = [f"== {title} =="]
    for k in sorted(d.keys()):
        lines.append(f"{k}: {_to_builtin(d[k])}")
    return "\n".join(lines)

def log_run_header(log, run_dir, settings, **kw):
    # masks as disabled lists for readability
    in_mask  = settings.get("mlp_in_mask", {})
    out_mask = settings.get("mlp_out_mask", {})
    in_disabled  = [k for k,v in in_mask.items()  if v is False]
    out_disabled = [k for k,v in out_mask.items() if v is False]

    model = dict(
        nb_inputs=settings["nb_inputs"], nb_hidden=settings["nb_hidden"], nb_outputs=settings["nb_outputs"]
    )
    sim = dict(time_step=settings["time_step"], nb_steps=settings["nb_steps"], max_time=settings["max_time"])
    sim["input_delay_steps"] = settings.get("input_delay_steps")
    sim["use_input_delay"] = settings.get("use_input_delay")
    sim["input_delay_temp"] = settings.get("input_delay_temp")
    data = dict(
        batch_size=settings["batch_size"],
        cache_dir=settings["cache_dir"], cache_subdir=settings["cache_subdir"],
        train_file=settings["train_file"], test_file=settings["test_file"],
        val_file=settings.get("val_file"),
        nb_inputs_raw=settings.get("nb_inputs_raw"),
        channel_compress_enable=settings.get("channel_compress_enable", False),
        channel_compress_factor=settings.get("channel_compress_factor", 1),
        channel_compress_target=settings.get("channel_compress_target"),
        channel_compress_mode=settings.get("channel_compress_mode"),
        channel_compress_factor_snn=settings.get("channel_compress_factor_snn"),
        channel_compress_factor_mod=settings.get("channel_compress_factor_mod"),
        nb_inputs_mod=settings.get("nb_inputs_mod"),
        channel_compress_mlp_hidden_sizes=settings.get("channel_compress_mlp_hidden_sizes"),
    )
    train = dict(
        lr=settings["lr"], nb_epochs=settings["nb_epochs"],
        tau_syn=settings.get("tau_syn"), tau_mem=settings.get("tau_mem"),
        tau_match_clip=settings.get("tau_match_clip"),
        tau_decay_dt=_effective_tau_decay_dt(settings),
        tau_scale=settings.get("paper_tau_scale", 1.0),
        weight_scale=settings.get("weight_scale"),
        hidden_dropout_p=settings.get("hidden_dropout_p", 0.0),
    )

    aug_train = dict(**(settings.get("augment_train", {}) or {}), postbin_mask=settings.get("postbin_mask_train", 0.0))
    aug_eval  = dict(**(settings.get("augment_eval",  {}) or {}), postbin_mask=settings.get("postbin_mask_eval",  0.0))
    aug_test  = dict(**(settings.get("augment_test",  {}) or {}), postbin_mask=settings.get("postbin_mask_test",  0.0))

    nm_cfg = settings.get("nm_cfg", {}) or {}
    nm_cfg_structured = {
        "enable": bool(nm_cfg.get("enable", False)),
        "hidden_per_neuron": int(nm_cfg.get("hidden_per_neuron", 0)),
        "output_per_neuron": int(nm_cfg.get("output_per_neuron", 0)),
        "init_scale": nm_cfg.get("init_scale"),
        "activation_kind": nm_cfg.get("activation_kind"),
        "flat_order": nm_cfg.get("flat_order"),
        "warm_start": bool(nm_cfg.get("warm_start", False)),
        "init_identity": bool(nm_cfg.get("init_identity", False)),
        "neuron_fraction_enable": bool(nm_cfg.get("neuron_fraction_enable", False)),
        "neuron_fraction": nm_cfg.get("neuron_fraction"),
        "param_fraction_enable": bool(nm_cfg.get("param_fraction_enable", False)),
        "param_fraction": nm_cfg.get("param_fraction"),
        "mapper_type": nm_cfg.get("mapper_type"),
        "hidden_activation": nm_cfg.get("hidden_activation"),
        "hidden_mlp_hidden": nm_cfg.get("hidden_mlp_hidden"),
        "output_mlp_hidden": nm_cfg.get("output_mlp_hidden"),
        "mapper_hidden_exact": nm_cfg.get("mapper_hidden_exact"),
    }

    modulator = dict(
        mlp_hidden_sizes=settings.get("mlp_hidden_sizes"),
        mod_hidden_sizes=settings.get("mod_hidden_sizes"),
        mlp_interval=settings.get("mlp_interval"),
        mod_update_every_step=settings.get("mod_update_every_step"),
        mlp_mode=settings.get("mlp_mode", "mlp_sub"),
        mlp_in_disable=in_disabled, mlp_out_disable=out_disabled,
        snn_mod_hidden=settings.get("snn_mod_hidden"),
        snn_mod_gain_init=settings.get("snn_mod_gain_init"),
        snn_mod_weight_scale=settings.get("snn_mod_weight_scale"),
        snn_mod_hidden_recurrent=settings.get("snn_mod_hidden_recurrent"),
        snn_sub_scale_init=settings.get("snn_sub_scale_init"),
        snn_sub_bias_init=settings.get("snn_sub_bias_init"),
        ann_output_activation=settings.get("ann_output_activation_effective") or settings.get("ann_output_activation"),
        ann_output_activation_override=settings.get("ann_output_activation"),
        group_cfg=settings.get("group_cfg"),
        mod_fixed_mask_enable=(settings.get("mod_mask_cfg", {}) or {}).get("fixed_enable"),
        mod_fixed_mask_seed=(settings.get("mod_mask_cfg", {}) or {}).get("fixed_seed"),
        mod_fixed_mask_flat_inputs=(settings.get("mod_mask_cfg", {}) or {}).get("fixed_flat_inputs"),
        mod_fixed_mask_zero_fallback=(settings.get("mod_mask_cfg", {}) or {}).get("zero_fallback"),
        mod_hid_flat_group=settings.get("mod_hid_flat_group"),
        mod_hid_flat_modulated_only=settings.get("mod_hid_flat_modulated_only"),
    )

    misc = dict(save_dir=settings.get("save_dir"))
    extras = dict(**(kw or {}))
    config_path = write_run_config(run_dir, settings, extras)

    log("\n===== RUN CONFIG =====")
    log(f"[info] Wrote machine-readable run config: {config_path}")
    log(_section("Run Dir", {"path": str(run_dir)}))
    log(_section("Model", model))
    log(_section("Simulation", sim))
    log(_section("Data", data))
    log(_section("Training", train))
    log(_section("Augment (train)", aug_train))
    log(_section("Augment (eval)",  aug_eval))
    log(_section("Augment (test)",  aug_test))
    if any(v is not None for v in modulator.values()):
        log(_section("Modulator", modulator))
    if any(v is not None for v in nm_cfg_structured.values()):
        log(_section("Neuromodulation", nm_cfg_structured))
    ranges = settings.get("param_ranges")
    if ranges:
        log(_section("Param Ranges", ranges))
    if "param_timescales" in settings:
        log(_section("Param Timescales", settings.get("param_timescales", {})))
    if "param_smoothing" in settings:
        log(_section("Param Smoothing", settings.get("param_smoothing", {})))
    if "weight_mod_cfg" in settings:
        log(_section("Weight Modulation", settings.get("weight_mod_cfg", {})))
    log(_section("Controls", extras))
    log(_section("All Settings", _config_snapshot(settings)))
    if settings.get("mod_hid_flat_modulated_only") and not (settings.get("mod_mask_cfg", {}) or {}).get("fixed_enable"):
        log("[warn] mod_hid_flat_modulated_only is true but mod_fixed_mask_enable is false; hid_flat will not be filtered.")
    mod_output_size = settings.get("_mod_output_size")
    if isinstance(mod_output_size, int) and mod_output_size == 0:
        log("[warn] Modulator output size is 0; no parameters will be modulated. Training will still run.")
    log("=" * 24 + "\n")

# -------------------------
# Augment helpers (pre/post bin)
# -------------------------
def jitter_times(times, sigma_ms, max_time):
    if sigma_ms <= 0: return times
    noise = np.random.normal(0.0, sigma_ms/1000.0, size=times.shape)
    t = times + noise
    return np.clip(t, 0.0, max_time)

def shift_times(times, shift_ms, max_time):
    if shift_ms == 0: return times
    return np.clip(times + (shift_ms/1000.0), 0.0, max_time)

def scale_times(times, scale, max_time):
    return np.clip(times * scale, 0.0, max_time)

def event_drop(times, units, drop_p):
    if drop_p <= 0: return times, units
    keep = np.random.rand(times.shape[0]) > drop_p
    return times[keep], units[keep]

def event_insert(times, units, nb_units, rate=0.0, max_time=1.4):
    n_new = int(len(times) * rate)
    if n_new <= 0: return times, units
    new_t = np.random.rand(n_new) * max_time
    new_u = np.random.randint(0, nb_units, size=n_new)
    return np.concatenate([times, new_t]), np.concatenate([units, new_u])

def band_mask(times, units, nb_units, band_frac=0.0):
    if band_frac <= 0: return times, units
    band = max(1, int(nb_units * band_frac))
    start = np.random.randint(0, nb_units - band + 1)
    keep = (units < start) | (units >= start + band)
    return times[keep], units[keep]

def compress_units(units, factor: int, nb_units_new: int):
    if factor <= 1: return units
    u = (units // factor).astype(int)
    return np.clip(u, 0, nb_units_new - 1)

def compress_dense_inputs(inputs: torch.Tensor, factor: int, nb_units_new: Optional[int] = None) -> torch.Tensor:
    if factor <= 1:
        return inputs
    B, T, C = inputs.shape
    if nb_units_new is None:
        nb_units_new = int(math.ceil(C / float(factor)))
    pad = nb_units_new * factor - C
    if pad > 0:
        inputs = F.pad(inputs, (0, pad))
    reshaped = inputs.reshape(B, T, nb_units_new, factor)
    return reshaped.sum(dim=3)

def channel_jitter(units, nb_units, sigma_units=0.0):
    if sigma_units <= 0: return units
    jitter = np.random.normal(0.0, sigma_units, size=units.shape)
    u = np.round(units + jitter).astype(int)
    return np.clip(u, 0, nb_units - 1)

def inject_poisson_noise(times, units, nb_units, rate_hz=0.0, max_time=1.4, per_input: bool = False):
    if rate_hz <= 0: return times, units
    total_rate = rate_hz * (nb_units if per_input else 1.0)
    n_new = np.random.poisson(total_rate * max_time)
    if n_new <= 0: return times, units
    new_t = np.random.rand(n_new) * max_time
    new_u = np.random.randint(0, nb_units, size=n_new)
    return np.concatenate([times, new_t]), np.concatenate([units, new_u])

def time_mask_postbin(X_dense, mask_frac=0.0):
    if mask_frac <= 0: return X_dense
    T = X_dense.shape[1]
    L = max(1, int(T * mask_frac))
    s = np.random.randint(0, T - L + 1)
    X_dense[:, s:s+L, :] = 0.0
    return X_dense

def augment_spike_train(times, units, cfg, nb_units, max_time):
    compress_factor = int(cfg.get("compress_factor", 1))
    jitter_std = cfg.get("channel_jitter_std_eff")
    if jitter_std is None:
        jitter_std = cfg.get("channel_jitter_std", 0.0)
        if compress_factor > 1 and jitter_std > 0:
            jitter_std = jitter_std / float(compress_factor)
    noise_rate_hz = cfg.get("noise_rate_hz_eff")
    if noise_rate_hz is None:
        noise_rate_hz = cfg.get("noise_rate_hz", 0.0)
        if compress_factor > 1 and cfg.get("noise_per_input", False):
            noise_rate_hz = noise_rate_hz * float(compress_factor)
    if compress_factor > 1:
        units = compress_units(units, compress_factor, nb_units)
    if cfg.get("jitter_ms", 0) > 0:
        times = jitter_times(times, np.random.uniform(0, cfg["jitter_ms"]), max_time)
    if cfg.get("shift_ms", 0) > 0:
        s = np.random.uniform(-cfg["shift_ms"], cfg["shift_ms"])
        times = shift_times(times, s, max_time)
    if cfg.get("scale_low", 1.0) != 1.0 or cfg.get("scale_high", 1.0) != 1.0:
        scale = np.random.uniform(cfg.get("scale_low", 1.0), cfg.get("scale_high", 1.0))
        times = scale_times(times, scale, max_time)
    if jitter_std > 0:
        units = channel_jitter(units, nb_units, jitter_std)
    if cfg.get("drop_p", 0) > 0:
        times, units = event_drop(times, units, cfg["drop_p"])
    if cfg.get("insert_rate", 0) > 0:
        times, units = event_insert(times, units, nb_units, cfg["insert_rate"], max_time)
    if noise_rate_hz > 0:
        times, units = inject_poisson_noise(
            times, units, nb_units,
            rate_hz=noise_rate_hz,
            max_time=max_time,
            per_input=cfg.get("noise_per_input", False)
        )
    if cfg.get("band_frac", 0) > 0:
        times, units = band_mask(times, units, nb_units, cfg["band_frac"])
    return times, units

# -------------------------
# Data generator (with optional subset indices + augmentation)
# -------------------------
def labels_to_numpy(y):
    try:
        return np.asarray(y[()], dtype=int)
    except Exception:
        return np.asarray(y, dtype=int)


def sparse_data_generator_from_hdf5_spikes(
    X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True, indices=None,
    augment_cfg: dict=None, postbin_time_mask: float=0.0
):
    labels_full = labels_to_numpy(y)
    sample_index = np.arange(len(labels_full)) if indices is None else np.asarray(indices)
    if shuffle: np.random.shuffle(sample_index)
    firing_times, units_fired = X['times'], X['units']
    time_bins = np.linspace(0, max_time, num=nb_steps)

    for start in range(0, len(sample_index), batch_size):
        batch_index = sample_index[start:start + batch_size]
        if len(batch_index) == 0:
            continue
        coo = [[],[],[]]
        for bc, idx in enumerate(batch_index):
            times = np.array(firing_times[idx]); units = np.array(units_fired[idx])
            if augment_cfg:
                times, units = augment_spike_train(times, units, augment_cfg, nb_units, max_time)
            bins = np.digitize(times, time_bins)
            coo[0].extend([bc]*len(bins)); coo[1].extend(bins); coo[2].extend(units)
        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
        X_batch = torch.sparse_coo_tensor(
            i, v, torch.Size([len(batch_index), nb_steps, nb_units]),
            dtype=dtype, device=device
        )
        if postbin_time_mask > 0.0:
            Xd = X_batch.to_dense()
            Xd = torch.from_numpy(time_mask_postbin(Xd.cpu().numpy(), postbin_time_mask)).to(device)
            X_batch = Xd.to_sparse()
        y_batch = torch.tensor(labels_full[batch_index], device=device)
        yield X_batch, y_batch

# -------------------------
# Split helpers + persistence
# -------------------------
def stratified_split_indices(y, val_fraction=0.1, seed=0):
    y_np = labels_to_numpy(y)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(y_np):
        idx_c = np.where(y_np == c)[0]
        rng.shuffle(idx_c)
        n_val = max(1, int(len(idx_c) * val_fraction))
        val_idx.extend(idx_c[:n_val]); train_idx.extend(idx_c[n_val:])
    return np.array(train_idx), np.array(val_idx)

def make_kfold_splits(y, k_folds=5, seed=0):
    y_np = labels_to_numpy(y)
    rng = np.random.default_rng(seed)
    per_class = {}
    for c in np.unique(y_np):
        idx_c = np.where(y_np == c)[0]
        rng.shuffle(idx_c)
        per_class[c] = np.array_split(idx_c, k_folds)
    folds = []
    for k in range(k_folds):
        val_idx = np.concatenate([per_class[c][k] for c in per_class])
        train_idx = np.concatenate([np.concatenate([per_class[c][i] for i in range(k_folds) if i != k]) for c in per_class])
        folds.append((train_idx, val_idx))
    return folds

def save_indices(path: Path, train_idx: np.ndarray, val_idx: Optional[np.ndarray], meta: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train_idx=train_idx,
                        val_idx=val_idx if val_idx is not None else np.array([], dtype=int),
                        meta=meta or {})

def load_indices(path: Path):
    data = np.load(path, allow_pickle=True)
    train_idx = data["train_idx"]
    val_idx_arr = data["val_idx"]
    val_idx = val_idx_arr if val_idx_arr.size > 0 else None
    meta = dict(data["meta"].item()) if "meta" in data else {}
    return train_idx, val_idx, meta

# -------------------------
# Surrogate gradient spike
# -------------------------
class SurrGradSpike(torch.autograd.Function):
    scale = 100.0
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out
    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input / (SurrGradSpike.scale * torch.abs(inp) + 1.0) ** 2

spike_fn = SurrGradSpike.apply

# -------------------------
# Tau/time-step helpers
# -------------------------
def _effective_tau_decay_dt(settings: Dict[str, Union[int, float, bool]]) -> float:
    dt = float(settings.get("time_step", 1.0))
    if settings.get("tau_match_clip") and settings.get("max_time", 0.0) > 0:
        steps = max(1, int(settings.get("nb_steps", 1)))
        return float(settings["max_time"]) / steps
    return dt


def _apply_input_delay(
    inputs_dense: torch.Tensor,
    delay_logits: Optional[torch.Tensor],
    max_delay: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply a learnable axonal delay per input channel by shifting spike trains.
    Straight-through argmax over delays 0..max_delay: forward uses one-hot,
    backward flows through softmax for gradients.
    """
    if delay_logits is None or max_delay <= 0:
        return inputs_dense
    B, T, C = inputs_dense.shape
    max_delay = min(max_delay, delay_logits.size(1) - 1)
    logits = delay_logits[:, : max_delay + 1].to(device=inputs_dense.device, dtype=inputs_dense.dtype)  # [C,K]
    temp = max(1e-3, float(temperature))
    probs = F.softmax(logits / temp, dim=1)
    hard_idx = probs.argmax(dim=1)
    hard = F.one_hot(hard_idx, num_classes=probs.size(1)).to(device=inputs_dense.device, dtype=inputs_dense.dtype)
    weights = hard + (probs - probs.detach())  # straight-through
    out = torch.zeros_like(inputs_dense)
    for k in range(max_delay + 1):
        if k >= T:
            break
        shifted = torch.cat(
            [torch.zeros((B, k, C), device=inputs_dense.device, dtype=inputs_dense.dtype), inputs_dense[:, : T - k, :]],
            dim=1,
        )
        out = out + shifted * weights[:, k].view(1, 1, C)
    return out

# -------------------------
# Base SNN state + forward
# -------------------------
def setup_model(settings: Dict):
    nb_inputs   = settings["nb_inputs"]
    nb_hidden   = settings["nb_hidden"]
    nb_outputs  = settings["nb_outputs"]
    time_step   = settings["time_step"]
    tau_syn     = settings["tau_syn"]
    tau_mem     = settings["tau_mem"]
    weight_scale= settings["weight_scale"]

    # Optionally align decay step with clip duration so taus stay in real-time units
    decay_time_step = _effective_tau_decay_dt(settings)

    param_ranges = _resolve_param_ranges(settings)
    thr_lo, thr_hi = param_ranges["thr"]
    reset_lo, reset_hi = param_ranges["reset"]
    rest_lo, rest_hi = param_ranges["rest"]
    thresholds_1 = torch.empty((1, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.uniform_(thresholds_1, a=float(thr_lo), b=float(thr_hi))
    reset_1 = torch.empty((1, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.uniform_(reset_1, a=float(reset_lo), b=float(reset_hi))
    rest_1 = torch.empty((1, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.uniform_(rest_1, a=float(rest_lo), b=float(rest_hi))

    homo = bool(settings.get("homo_init", False))
    if homo:
        thr_val = 1.0
        reset_val = 0.0
        rest_val = 0.0
        thresholds_1.data.fill_(thr_val)
        reset_1.data.fill_(reset_val)
        rest_1.data.fill_(rest_val)

        alpha_val = math.exp(-decay_time_step / tau_syn)
        beta_val = math.exp(-decay_time_step / tau_mem)
        alpha_hetero_1 = torch.full((1, nb_hidden), alpha_val, device=device, dtype=dtype, requires_grad=True)
        beta_hetero_1  = torch.full((1, nb_hidden), beta_val, device=device, dtype=dtype, requires_grad=True)
        alpha_hetero_2 = torch.full((1, nb_outputs), alpha_val, device=device, dtype=dtype, requires_grad=True)
        beta_hetero_2  = torch.full((1, nb_outputs), beta_val, device=device, dtype=dtype, requires_grad=True)
    else:
        distribution = dist_fn('gamma')
        alpha_hetero_1_dist = torch.tensor(distribution(tau_syn, 3, (1, nb_hidden)), device=device, dtype=dtype)
        alpha_hetero_1 = torch.exp(-decay_time_step / alpha_hetero_1_dist).requires_grad_(True)
        beta_hetero_1_dist = torch.tensor(distribution(tau_mem, 3, (1, nb_hidden)), device=device, dtype=dtype)
        beta_hetero_1 = torch.exp(-decay_time_step / beta_hetero_1_dist).requires_grad_(True)
        alpha_hetero_2_dist = torch.tensor(distribution(tau_syn, 3, (1, nb_outputs)), device=device, dtype=dtype)
        alpha_hetero_2 = torch.exp(-decay_time_step / alpha_hetero_2_dist).requires_grad_(True)
        beta_hetero_2_dist = torch.tensor(distribution(tau_mem, 3, (1, nb_outputs)), device=device, dtype=dtype)
        beta_hetero_2 = torch.exp(-decay_time_step / beta_hetero_2_dist).requires_grad_(True)
        with torch.no_grad():
            alpha_hetero_1.clamp_(float(param_ranges["alpha_1"][0]), float(param_ranges["alpha_1"][1]))
            beta_hetero_1.clamp_(float(param_ranges["beta_1"][0]), float(param_ranges["beta_1"][1]))
            alpha_hetero_2.clamp_(float(param_ranges["alpha_2"][0]), float(param_ranges["alpha_2"][1]))
            beta_hetero_2.clamp_(float(param_ranges["beta_2"][0]), float(param_ranges["beta_2"][1]))

    # w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    # torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / math.sqrt(nb_inputs))
    # w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
    # torch.nn.init.normal_(w2, mean=0.0, std=weight_scale / math.sqrt(nb_hidden))
    # v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    # torch.nn.init.normal_(v1, mean=0.0, std=weight_scale / math.sqrt(nb_hidden))

    # ---- weight initialization ----

    # w1 ~ Normal(-0.00046, 0.0215²)
    w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(w1, mean=-0.00046421893, std=0.021497283)

    # v1 ~ Normal(-0.00443, 0.0257²)
    v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(v1, mean=-0.004429796, std=0.025746185)

    # w2 ~ mixture of 2 Gaussians
    weights = [0.61827673, 0.38172327]
    means   = [0.01217757, -0.03004124]
    stds    = [0.03018986, 0.04652500]

    comp_mask = torch.rand((nb_hidden, nb_outputs), device=device, dtype=dtype) < weights[0]
    z = torch.randn((nb_hidden, nb_outputs), device=device, dtype=dtype)
    w2 = torch.where(
        comp_mask,
        means[0] + stds[0] * z,
        means[1] + stds[1] * z
    ).requires_grad_()

    # learnable input delay kernels (per-input softmax over [0, max_delay])
    delay_logits = _init_input_delay_logits(settings, nb_inputs) if settings.get("use_input_delay", False) else None

    # --------------------------------

    state = dict(
        thresholds_1=thresholds_1, reset_1=reset_1, rest_1=rest_1,
        alpha_hetero_1=alpha_hetero_1, beta_hetero_1=beta_hetero_1,
        alpha_hetero_2=alpha_hetero_2, beta_hetero_2=beta_hetero_2,
        w1=w1, w2=w2, v1=v1,
    )
    if delay_logits is not None:
        state["input_delay_logits"] = delay_logits
    return _apply_snn_train_flags(state, _snn_train_flags(settings))

def run_snn_hetero(inputs_dense, state: Dict, settings: Dict):
    nb_hidden   = settings["nb_hidden"]
    nb_outputs  = settings["nb_outputs"]
    nb_steps    = settings["nb_steps"]
    batch_size  = inputs_dense.size(0)
    training = bool(settings.get("training", False))
    hidden_dropout_p = float(settings.get("hidden_dropout_p", 0.0))
    psp_norm = bool(settings.get("psp_norm_peak", False))

    w1, w2, v1 = state["w1"], state["w2"], state["v1"]
    alpha_hetero_1 = state["alpha_hetero_1"]
    beta_hetero_1  = state["beta_hetero_1"]
    alpha_hetero_2 = state["alpha_hetero_2"]
    beta_hetero_2  = state["beta_hetero_2"]
    thresholds_1   = state["thresholds_1"]
    reset_1        = state["reset_1"]
    rest_1         = state["rest_1"]

    syn = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    decay_dt = _effective_tau_decay_dt(settings)
    gain_hidden = None
    gain_out = None
    if psp_norm:
        gain_hidden = _psp_peak_gain(alpha_hetero_1, beta_hetero_1, dt=decay_dt)
        gain_out = _psp_peak_gain(alpha_hetero_2, beta_hetero_2, dt=decay_dt)

    mem_rec, spk_rec = [], []

    # feedforward input current sequence to hidden: [B,T,H]
    delay_logits = state.get("input_delay_logits")
    max_delay = max(1, int(settings.get("input_delay_steps", 1))) if settings.get("use_input_delay", False) else 0
    inputs_delayed = _apply_input_delay(
        inputs_dense, delay_logits, max_delay, temperature=settings.get("input_delay_temp", 1.0)
    )
    h1_from_input = torch.einsum("btc,cd->btd", inputs_delayed, w1)

    out = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

    for t in range(nb_steps):
        # total hidden current at time t (input + recurrent)
        h1_t = h1_from_input[:, t] + torch.einsum("bd,dc->bc", out, v1)  # [B,H]

        # SNN update
        mthr = mem - thresholds_1
        spk  = spike_fn(mthr)
        if hidden_dropout_p > 0.0 and training:
            spk = F.dropout(spk, p=hidden_dropout_p, training=True)
        rst  = (mthr > 0).float()

        syn = alpha_hetero_1 * syn + h1_t
        syn_term = (1 - beta_hetero_1) * syn
        if gain_hidden is not None:
            syn_term = gain_hidden * syn_term
        mem = beta_hetero_1 * (mem - rest_1) + rest_1 + syn_term - rst * (thresholds_1 - reset_1)

        mem_rec.append(mem)
        spk_rec.append(spk)
        out = spk

    mem_rec = torch.stack(mem_rec, dim=1)   # [B,T,H]
    spk_rec = torch.stack(spk_rec, dim=1)   # [B,T,H]

    # raw output current from hidden spikes (before output filters)
    h2 = torch.einsum("bth,ho->bto", spk_rec, w2)  # [B,T,O]

    flt = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    for t in range(nb_steps):
        flt = alpha_hetero_2 * flt + h2[:, t]
        flt_term = (1 - beta_hetero_2) * flt
        if gain_out is not None:
            flt_term = gain_out * flt_term
        out = beta_hetero_2 * out + flt_term
        out_rec.append(out)
    out_rec = torch.stack(out_rec, dim=1)
    return out_rec, (mem_rec, spk_rec)

# -------------------------
# Eval helpers (SNN)
# -------------------------
@torch.no_grad()
def eval_val_metrics(state, settings, x_data, y_data, indices: Optional[np.ndarray] = None):
    loss_fn = nn.NLLLoss(reduction="mean")
    log_softmax_fn = nn.LogSoftmax(dim=1)
    nlls, accs = [], []
    gen = sparse_data_generator_from_hdf5_spikes(
        x_data, y_data, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False, indices=indices,
        augment_cfg=settings.get("augment_eval"), postbin_time_mask=settings.get("postbin_mask_eval", 0.0)
    )
    for x_local, y_local in gen:
        output, _ = run_snn_hetero(x_local.to_dense(), state, settings)
        m, _ = torch.max(output, 1)
        log_p_y = log_softmax_fn(m)
        nlls.append(loss_fn(log_p_y, y_local).item())
        _, am = torch.max(m, 1)
        accs.append(torch.mean((y_local == am).float()).item())
    return float(np.mean(nlls)) if nlls else float("nan"), float(np.mean(accs)) if accs else 0.0

@torch.no_grad()
def evaluate_testset(state, settings, x_test, y_test):
    loss_fn = nn.NLLLoss(reduction="mean")
    log_softmax_fn = nn.LogSoftmax(dim=1)
    nlls, accs = [], []
    gen = sparse_data_generator_from_hdf5_spikes(
        x_test, y_test, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False,
        augment_cfg=settings.get("augment_test"),
        postbin_time_mask=settings.get("postbin_mask_test", 0.0)
    )
    for x_local, y_local in gen:
        output, _ = run_snn_hetero(x_local.to_dense(), state, settings)
        m, _ = torch.max(output, 1)
        log_p_y = log_softmax_fn(m)
        nlls.append(loss_fn(log_p_y, y_local).item())
        _, am = torch.max(m, 1)
        accs.append(torch.mean((y_local == am).float()).item())
    test_nll = float(np.mean(nlls)) if nlls else float("nan")
    test_acc = float(np.mean(accs)) if accs else 0.0
    print(f"Final Test — NLL: {test_nll:.5f}, Accuracy: {test_acc:.5f}")
    return test_nll, test_acc

# -------------------------
# Train SNN (returns best ckpt path too)
# -------------------------
def train_snn_hetero(
    state: dict, settings: dict,
    x_train, y_train, x_test, y_test,
    x_val=None, y_val=None,
    save_every_epoch: bool = True,
    use_validation: bool = True,
    val_fraction: float = 0.1,
    k_folds: int = 1,
    patience: int = 10,
    eval_test_each_epoch: bool = False,
    seed: int = 0,
    fixed_split_path: str = None,
    reinit_per_fold: bool = True,
    chosen_metric: str = "val_acc",                      # "val_acc" or "train_acc"
    chosen_threshold: Optional[float] = None,            # threshold for chosen metric
    run_index: Optional[int] = None,
):
    run_dir = ensure_run_dir(settings["save_dir"], run_index=run_index)
    print(f"Saving to: {run_dir}")
    log_path = run_dir / "training_log.txt"
    log_file = open(log_path, "a", buffering=1)
    print(f"Logging to: {log_path}")
    def log(msg: str):
        print(msg); print(msg, file=log_file)

    snn_disabled = sorted([k for k, v in (settings.get("snn_train_flags") or {}).items() if not v])

    external_val_available = (x_val is not None) and (y_val is not None)
    val_dataset_name = settings.get("val_file")
    use_validation_flag = use_validation or external_val_available
    if external_val_available:
        try:
            val_size = len(y_val)
        except TypeError:
            val_size = "unknown"
        log(f"[info] External validation split detected ({val_dataset_name or 'custom'}), size={val_size}.")

    log_run_header(
        log, run_dir, settings,
        mode="snn",
        use_validation=use_validation_flag,
        val_fraction=val_fraction,
        k_folds=k_folds,
        patience=patience,
        eval_test_each_epoch=eval_test_each_epoch,
        seed=seed,
        fixed_split_path=fixed_split_path,
        reinit_per_fold=reinit_per_fold,
        chosen_metric=chosen_metric,
        chosen_threshold=chosen_threshold,
        input_delay_steps=settings.get("input_delay_steps"),
        use_input_delay=settings.get("use_input_delay"),
        snn_reg_enabled=settings.get("use_snn_reg"),
        snn_train_disabled=snn_disabled,
        external_val=external_val_available,
        external_val_file=val_dataset_name,
    )
    log(format_param_stats(state, prefix="Initial:", settings=settings))

    # ----- build splits (unchanged) -----
    folds = []
    if external_val_available:
        tr_idx = np.arange(len(y_train))
        val_idx = None
        idx_path = Path(run_dir) / "Fold_1" / "split_indices.npz"
        save_indices(idx_path, tr_idx, val_idx, meta={"external_validation": True})
        folds = [(tr_idx, val_idx, {"source": "external_val", "path": str(idx_path)})]
        if k_folds and k_folds > 1:
            log("[info] External validation provided; ignoring k-fold configuration.")
        if fixed_split_path:
            log("[info] External validation provided; ignoring --fixed_split_path.")
    elif k_folds and k_folds > 1:
        if fixed_split_path:
            for k in range(1, k_folds + 1):
                p = Path(fixed_split_path) if str(fixed_split_path).endswith(".npz") else Path(fixed_split_path) / f"fold_{k}_indices.npz"
                tr_idx, val_idx, meta = load_indices(p)
                folds.append((tr_idx, val_idx, {"source": "loaded", "path": str(p)}))
            log(f"Loaded {k_folds} fixed folds from {fixed_split_path}")
        else:
            kfold_pairs = make_kfold_splits(y_train, k_folds=k_folds, seed=seed)
            for k, (tr_idx, val_idx) in enumerate(kfold_pairs, start=1):
                idx_path = (Path(run_dir) / f"Fold_{k}" / "fold_indices.npz")
                save_indices(idx_path, tr_idx, val_idx, meta={"k_folds": k_folds, "fold": k, "seed": seed})
                folds.append((tr_idx, val_idx, {"source": "created", "path": str(idx_path)}))
            log(f"Created & saved {k_folds} stratified folds.")
    else:
        if use_validation_flag:
            if fixed_split_path:
                p = Path(fixed_split_path)
                tr_idx, val_idx, meta = load_indices(p)
                folds = [(tr_idx, val_idx, {"source": "loaded", "path": str(p)})]
                log(f"Loaded fixed train/val split from {p}")
            else:
                tr_idx, val_idx = stratified_split_indices(y_train, val_fraction=val_fraction, seed=seed)
                idx_path = Path(run_dir) / "Fold_1" / "split_indices.npz"
                save_indices(idx_path, tr_idx, val_idx, meta={"val_fraction": val_fraction, "seed": seed})
                folds = [(tr_idx, val_idx, {"source": "created", "path": str(idx_path)})]
                log(f"Created & saved train/val split to {idx_path}")
        else:
            tr_idx = np.arange(len(y_train))
            val_idx = None
            idx_path = Path(run_dir) / "Fold_1" / "split_indices.npz"
            save_indices(idx_path, tr_idx, val_idx, meta={"no_validation": True})
            folds = [(tr_idx, val_idx, {"source": "created", "path": str(idx_path)})]
            log("No validation: saved train indices for reproducibility.")

    best_ckpt_paths = []
    index_manifest = []

    # ----- train per fold (unchanged control flow) -----
    for fold_id, (tr_idx, val_idx, idx_meta) in enumerate(folds, start=1):
        multi_fold = (k_folds and k_folds > 1) and not external_val_available
        fold_dir = run_dir / (f"Fold_{fold_id}" if multi_fold else "Fold_1")
        fold_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== Fold {fold_id}/{len(folds)} ({idx_meta['source']}) — indices: {idx_meta['path']} ===")
        if external_val_available and y_val is not None:
            log(f"Train size: {len(tr_idx)} / External Val size: {len(y_val)}")
        elif val_idx is None:
            log(f"Train size (no validation): {len(tr_idx)}")
        else:
            log(f"Train/Val sizes: {len(tr_idx)} / {len(val_idx)}")

        state_fold = setup_model(settings) if reinit_per_fold else state
        log(format_param_stats(state_fold, prefix=f"Fold {fold_id}:", settings=settings))
        snn_trainable_params = _trainable_params_from(state_fold)
        optimizer = torch.optim.Adam(snn_trainable_params, lr=settings["lr"], betas=(0.9, 0.999)) if snn_trainable_params else None
        loss_fn = nn.NLLLoss(); log_softmax_fn = nn.LogSoftmax(dim=1)
        if optimizer is None:
            log("[info] All SNN parameters frozen for this fold; training will run without optimizer updates.")

        def _make_params_dict():
            return {k: v.clone() for k, v in state_fold.items()}
        def _save_checkpoint(path: Path, epoch: int, metrics: dict, hist: dict):
            torch.save({
                'epoch': epoch,
                'metrics': metrics,
                'params': _make_params_dict(),
                'history': hist,
                'run_config': _config_snapshot(settings),
            }, path)

        history = {"train_loss": [], "val_nll": [], "val_acc": [], "train_acc": [], "test_nll": [], "test_acc": []}
        best_val_nll, best_val_acc, best_val_epoch = float("inf"), -1.0, -1
        best_ckpt_epoch = -1
        epochs_no_improve = 0; chosen_saved = False
        warned_no_val_for_chosen = False
        val_data_x = x_val if external_val_available else x_train
        val_data_y = y_val if external_val_available else y_train
        val_indices = None if external_val_available else val_idx
        val_count = len(val_data_y) if (external_val_available and val_data_y is not None) else (len(val_idx) if val_idx is not None else 0)
        has_validation = use_validation_flag and (val_count > 0)

        settings_train = dict(settings)
        settings_train["training"] = True
        for e in range(settings["nb_epochs"]):
            local_loss, accs = [], []
            train_gen = sparse_data_generator_from_hdf5_spikes(
                x_train, y_train, settings["batch_size"], settings["nb_steps"],
                settings["nb_inputs"], settings["max_time"], shuffle=True, indices=tr_idx,
                augment_cfg=settings.get("augment_train"), postbin_time_mask=settings.get("postbin_mask_train", 0.0)
            )
            for x_local, y_local in train_gen:
                output, recs = run_snn_hetero(x_local.to_dense(), state_fold, settings_train)
                _, spks = recs
                m, _ = torch.max(output, 1)
                _, am = torch.max(m, 1)
                accs.append(torch.mean((y_local == am).float()).item())

                log_p_y = log_softmax_fn(m)
                ground_loss = loss_fn(log_p_y, y_local)

                if settings.get("use_snn_reg", True):
                    nb_hidden = settings["nb_hidden"]; nb_outputs = settings["nb_outputs"]
                    N_samp, T = 8128, settings["nb_steps"]
                    sl, thetal = 1.0, 0.01; su, thetau = 0.06, 100.0
                    tmp  = (torch.clamp((1/T) * torch.sum(spks, 1) - thetal, min=0.0)) ** 2
                    L1   = torch.sum(tmp, (0,1))
                    tmp2 = (torch.clamp((1/nb_hidden) * torch.sum(spks, (1,2)) - thetau, min=0.0)) ** 2
                    L2   = torch.sum(tmp2)
                    reg_loss = (sl/(N_samp + nb_hidden + nb_outputs)) * L1 + (su/N_samp) * L2
                    reg_loss = settings.get("snn_reg_scale", 1.0) * reg_loss
                    loss_val = ground_loss + reg_loss
                else:
                    loss_val = ground_loss
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss_val.backward()
                    optimizer.step()
                elif loss_val.requires_grad:
                    loss_val.backward()

                with torch.no_grad():
                    _clamp_state_params(state_fold, _resolve_param_ranges(settings))

                local_loss.append(float(loss_val.item()))

            train_loss = float(np.mean(local_loss)) if local_loss else float("nan")
            train_acc = float(np.mean(accs)) if accs else 0.0
            history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)

            test_nll = test_acc = None
            if eval_test_each_epoch:
                test_nll, test_acc = evaluate_testset(state_fold, settings, x_test, y_test)
                history["test_nll"].append(test_nll)
                history["test_acc"].append(test_acc)

            if has_validation and val_data_x is not None and val_data_y is not None:
                val_nll, val_acc = eval_val_metrics(
                    state_fold, settings, val_data_x, val_data_y, indices=val_indices
                )
                history["val_nll"].append(val_nll); history["val_acc"].append(val_acc)
                log_msg = (
                    f"[Fold {fold_id}] Epoch {e+1}: train_loss={train_loss:.5f} "
                    f"train_acc={train_acc:.5f} val_nll={val_nll:.5f} val_acc={val_acc:.5f}"
                )
                if eval_test_each_epoch:
                    log_msg += f" test_nll={test_nll:.5f} test_acc={test_acc:.5f}"
                log(log_msg)

                metrics_payload = {
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_nll": val_nll, "val_acc": val_acc,
                }
                if eval_test_each_epoch:
                    metrics_payload.update({"test_nll": test_nll, "test_acc": test_acc})
                if save_every_epoch:
                    _save_checkpoint(fold_dir / f"snn_{e+1}.pth", e+1, metrics_payload, history)

                # 'chosen' checkpoint logic (unchanged)
                if (chosen_threshold is not None) and (not chosen_saved):
                    metric_value = val_acc if chosen_metric == "val_acc" else train_acc if chosen_metric == "train_acc" else None
                    if (metric_value is not None) and (metric_value >= chosen_threshold):
                        _save_checkpoint(fold_dir / "chosen.pth", e+1, metrics_payload, history)
                        log(f"[Fold {fold_id}] Saved 'chosen' ({chosen_metric} {metric_value:.5f} ≥ {chosen_threshold:.5f})")
                        chosen_saved = True

                improved = val_nll < best_val_nll - 1e-6
                if improved:
                    best_val_nll, best_val_acc, best_val_epoch = val_nll, val_acc, e + 1
                    best_ckpt_epoch = best_val_epoch
                    epochs_no_improve = 0
                    _save_checkpoint(fold_dir / "snn_best.pth", best_ckpt_epoch, metrics_payload, history)
                    log(f"[Fold {fold_id}] New best validation loss at epoch {best_val_epoch} "
                        f"(val_nll={best_val_nll:.5f}, val_acc={best_val_acc:.5f})")
                else:
                    epochs_no_improve += 1
                    log(f"[Fold {fold_id}] No validation-loss improvement "
                        f"({epochs_no_improve}/{patience}) — best val_nll={best_val_nll:.5f} "
                        f"at epoch {best_val_epoch}")
                if epochs_no_improve >= patience:
                    log(f"[Fold {fold_id}] Early stopping."); break
            else:
                log_msg = (
                    f"[Fold {fold_id}] Epoch {e+1}: train_loss={train_loss:.5f} "
                    f"train_acc={train_acc:.5f} (no validation)"
                )
                if eval_test_each_epoch:
                    log_msg += f" test_nll={test_nll:.5f} test_acc={test_acc:.5f}"
                log(log_msg)

                metrics_payload = {"train_loss": train_loss, "train_acc": train_acc}
                if eval_test_each_epoch:
                    metrics_payload.update({"test_nll": test_nll, "test_acc": test_acc})
                if save_every_epoch:
                    _save_checkpoint(fold_dir / f"snn_{e+1}.pth", e+1, metrics_payload, history)

                if (chosen_threshold is not None) and (not chosen_saved):
                    if chosen_metric == "val_acc":
                        if not warned_no_val_for_chosen:
                            log("[info] chosen_metric=val_acc but validation is disabled; 'chosen' cannot be saved on val_acc.")
                            warned_no_val_for_chosen = True
                    elif chosen_metric == "train_acc" and train_acc >= chosen_threshold:
                        _save_checkpoint(fold_dir / "chosen.pth", e + 1, metrics_payload, history)
                        log(f"[Fold {fold_id}] Saved 'chosen' (train_acc {train_acc:.5f} ≥ {chosen_threshold:.5f})")
                        chosen_saved = True

                if train_loss < best_val_nll - 1e-6:
                    best_val_nll, best_val_epoch = train_loss, e + 1
                    best_ckpt_epoch = best_val_epoch
                    _save_checkpoint(fold_dir / "snn_best.pth", best_ckpt_epoch, metrics_payload, history)

        index_manifest.append({"fold": fold_id, "indices_path": idx_meta["path"], "best_epoch": best_ckpt_epoch})
        best_ckpt_paths.append(str(fold_dir / "snn_best.pth"))

        best_ckpt_path = fold_dir / "snn_best.pth"
        if best_ckpt_path.exists():
            try:
                state_eval = load_base_snn_state(best_ckpt_path, settings.get("snn_train_flags"), settings=settings)
                final_test_nll, final_test_acc = evaluate_testset(state_eval, settings, x_test, y_test)
                log(f"[Fold {fold_id}] Selected-checkpoint test — "
                    f"NLL: {final_test_nll:.5f}, Accuracy: {final_test_acc:.5f}")
                ckpt = torch.load(best_ckpt_path, map_location="cpu")
                ckpt["final_test"] = {"nll": final_test_nll, "accuracy": final_test_acc}
                torch.save(ckpt, best_ckpt_path)

                spike_stats = compute_snn_test_spike_stats(state_eval, settings, x_test, y_test)
                log(f"[Fold {fold_id}] Test spike stats — {_format_spike_stats(spike_stats)}")
            except Exception as exc:
                log(f"[Fold {fold_id}] Final test/spike-stats computation failed: {exc}")
        else:
            log(f"[Fold {fold_id}] Spike stats skipped: {best_ckpt_path} not found.")

    log_file.close()
    return {"fold_summaries": index_manifest, "run_dir": str(run_dir), "best_ckpt_paths": best_ckpt_paths}

# -------------------------
# MLP (configurable IO groups)
# -------------------------
def _mlp_io_slices(nb_inputs, nb_hidden, nb_outputs, in_mask: Dict, out_mask: Dict,
                   in_size_override: Optional[Dict[str, int]] = None,
                   out_size_override: Optional[Dict[str, int]] = None):
    in_groups = [
        ("alpha_1", nb_hidden), ("beta_1", nb_hidden),
        ("thr", nb_hidden), ("reset", nb_hidden), ("rest", nb_hidden),
        ("alpha_2", nb_outputs), ("beta_2", nb_outputs),
        ("w1", 0), ("v1", 0), ("w2", 0),
        ("in_flat", nb_inputs), ("hid_flat", nb_hidden), ("out_flat", nb_outputs),
    ]
    out_groups = [
        ("alpha_1", nb_hidden), ("beta_1", nb_hidden),
        ("thr", nb_hidden), ("reset", nb_hidden), ("rest", nb_hidden),
        ("alpha_2", nb_outputs), ("beta_2", nb_outputs),
        ("w1", 0), ("v1", 0), ("w2", 0),
    ]
    in_slices, in_size, cur = {}, 0, 0
    for name, size in in_groups:
        override = (in_size_override or {}).get(name)
        default_enabled = name not in WEIGHT_PARAM_NAMES
        enabled = in_mask.get(name, default_enabled)
        if enabled and (override is not None or size > 0):
            length = int(override) if override is not None else size
            if length > 0:
                in_slices[name] = slice(cur, cur+length); cur += length
    in_size = cur
    out_slices, out_size, cur = {}, 0, 0
    for name, size in out_groups:
        override = (out_size_override or {}).get(name)
        default_enabled = name not in WEIGHT_PARAM_NAMES
        enabled = out_mask.get(name, default_enabled)
        if enabled and (override is not None or size > 0):
            length = int(override) if override is not None else size
            if length > 0:
                out_slices[name] = slice(cur, cur+length); cur += length
    out_size = cur
    return in_slices, out_slices, in_size, out_size

DEFAULT_MLP_MODE = "mlp_sub"
DEFAULT_MLP_ARCH = "mlp"
VALID_MLP_ARCHS = {"mlp", "rnn", "lstm"}
MLP_MODE_SPECS = {
    "mlp_sub": {"activation": nn.Sigmoid, "activation_name": "sigmoid", "kind": "mlp"},
    "mlp_add": {"activation": nn.Tanh, "activation_name": "tanh", "kind": "mlp"},
    "mlp_combo": {"activation": nn.Tanh, "activation_name": "tanh", "kind": "mlp"},
    "snn_add": {"activation": nn.Identity, "activation_name": "linear", "kind": "snn"},
    "snn_sub": {"activation": nn.Identity, "activation_name": "linear", "kind": "snn"},
}

ADDITIVE_CLAMP_RANGES = {
    "alpha_1": (1.0/math.e, 0.995),
    "beta_1":  (1.0/math.e, 0.995),
    "alpha_2": (1.0/math.e, 0.995),
    "beta_2":  (1.0/math.e, 0.995),
    "thr":     (0.5, 1.5),
    "reset":   (-0.5, 0.5),
    "rest":    (-0.5, 0.5),
}
TINY_MOD_INIT_SCALE = 1e-7
TINY_MOD_GAIN_INIT = 1e-7
SOFT_MOD_INIT_SCALE = 1e-4
SOFT_MOD_GAIN_INIT = 1e-4
SNN_ADD_WEIGHT_SCALE_DEFAULT = 1e-2
SNN_ADD_GAIN_INIT_DEFAULT = 1e-2
SNN_SUB_SCALE_INIT_DEFAULT = 1.0
SNN_SUB_BIAS_INIT_DEFAULT = 0.0
ANN_ADD_FINAL_STD = 1e-4
COMBO_MULT_SCALE_MIN = 1.0 / 3.0
COMBO_MULT_SCALE_MAX = 3.0
COMBO_MULT_LOG_SPAN = math.log(COMBO_MULT_SCALE_MAX)


def _init_tiny_diag(matrix: torch.Tensor, scale: float = TINY_MOD_INIT_SCALE):
    """Sparse, near-zero init: zero everywhere with a tiny diagonal."""
    if matrix is None or matrix.dim() < 2:
        return
    with torch.no_grad():
        matrix.zero_()
        if scale <= 0:
            return
        diag = min(matrix.size(0), matrix.size(1))
        if diag > 0:
            eye = torch.eye(diag, dtype=matrix.dtype, device=matrix.device) * float(scale)
            matrix[:diag, :diag] = eye


def _init_tiny_linear(lin: nn.Linear, scale: float = TINY_MOD_INIT_SCALE):
    if not isinstance(lin, nn.Linear):
        return
    with torch.no_grad():
        _init_tiny_diag(lin.weight, scale=scale)
        if lin.bias is not None:
            lin.bias.zero_()

def _normalize_mlp_mode(mode: Optional[str]) -> str:
    if mode in MLP_MODE_SPECS:
        return mode
    if mode == "ann_sub":
        return "mlp_sub"
    if mode == "ann_add":
        return "mlp_add"
    if mode == "ann_combo":
        return "mlp_combo"
    return DEFAULT_MLP_MODE

def _is_substitution_mode(mode: str) -> bool:
    return _normalize_mlp_mode(mode) in {"mlp_sub", "snn_sub"}

def _normalize_mlp_arch(arch: Optional[str]) -> str:
    if arch is None:
        return DEFAULT_MLP_ARCH
    arch = str(arch).lower()
    if arch in VALID_MLP_ARCHS:
        return arch
    return DEFAULT_MLP_ARCH

def _is_snn_mode(mode: str) -> bool:
    return MLP_MODE_SPECS.get(mode, {}).get("kind") == "snn"

def _final_activation_for_mode(mode: str, override: Optional[str] = None) -> Tuple[nn.Module, str]:
    if override:
        lowered = override.lower()
        if lowered in {"none", "default"}:
            override = None
        elif lowered in ANN_OUTPUT_ACTIVATIONS:
            ctor, _ = ANN_OUTPUT_ACTIVATIONS[lowered]
            return ctor(), lowered
    spec = MLP_MODE_SPECS.get(mode, MLP_MODE_SPECS[DEFAULT_MLP_MODE])
    default_name = spec.get("activation_name", "sigmoid")
    entry = ANN_OUTPUT_ACTIVATIONS.get(default_name, ANN_OUTPUT_ACTIVATIONS["sigmoid"])
    ctor, _ = entry
    return ctor(), default_name

def _select_mlp_mode(settings: Dict, override: Optional[str] = None) -> str:
    if override is not None:
        return _normalize_mlp_mode(override)
    return _normalize_mlp_mode(settings.get("mlp_mode"))

def _resolve_mlp_hidden_sizes(settings: Dict) -> Optional[List[int]]:
    sizes = settings.get("mlp_hidden_sizes")
    if sizes is None:
        return None
    if isinstance(sizes, (list, tuple)):
        return [int(v) for v in sizes]
    return sizes

class InputCompressionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes: Optional[List[int]] = None):
        super().__init__()
        sizes = list(hidden_sizes or [])
        layers: List[nn.Module] = []
        prev = int(in_dim)
        for sz in sizes:
            layers.append(nn.Linear(prev, int(sz)))
            layers.append(nn.ReLU())
            prev = int(sz)
        layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if C != self.in_dim:
            raise ValueError(f"InputCompressionMLP expected {self.in_dim} inputs, got {C}.")
        x_flat = x.reshape(B * T, C)
        y = self.net(x_flat)
        return y.view(B, T, self.out_dim)

class ModulatingMLP(nn.Module):
    def __init__(self, nb_inputs, nb_hidden, nb_outputs,
                 hidden_sizes: Optional[List[int]] = None,
                 in_mask: Dict=None, out_mask: Dict=None,
                 mode: str = "mlp_sub", arch: str = "mlp",
                 in_size_override: Optional[Dict[str, int]] = None,
                 out_size_override: Optional[Dict[str, int]] = None,
                 output_activation: Optional[str] = None,
                 nm_cfg: Optional[Dict] = None,
                 nm_target_overrides: Optional[Dict[str, int]] = None,
                 hidden_group_layout: Optional['GroupLayout'] = None,
                 output_group_layout: Optional['GroupLayout'] = None,
                 current_cfg: Optional[Dict] = None,
                 current_target_overrides: Optional[Dict[str, int]] = None):
        super().__init__()
        self.nb_inputs, self.nb_hidden, self.nb_outputs = nb_inputs, nb_hidden, nb_outputs
        self.in_mask = in_mask or {}
        self.out_mask = out_mask or {}
        self.mode = _normalize_mlp_mode(mode)
        self.arch = _normalize_mlp_arch(arch)
        self.stateful = self.arch in {"rnn", "lstm"}
        self.output_activation_override = output_activation
        self.output_activation_kind: Optional[str] = None
        self.in_slices, base_out_slices, in_size, out_size = _mlp_io_slices(
            nb_inputs, nb_hidden, nb_outputs, self.in_mask, self.out_mask,
            in_size_override=in_size_override, out_size_override=out_size_override
        )
        self.input_size, self.output_size = in_size, out_size
        sizes = list(hidden_sizes) if hidden_sizes else [2048]
        self.hidden_sizes = sizes
        self.final_activation: Optional[nn.Module] = None
        self.layers: Optional[nn.Sequential] = None
        self.core: Optional[nn.Module] = None
        self.output_layer: Optional[nn.Linear] = None
        self._linear_layers: List[nn.Linear] = []
        self.rnn_hidden_size: Optional[int] = None
        self.rnn_num_layers: Optional[int] = None
        self._state = None
        self._state_batch = None
        self.nm_cfg = nm_cfg or {}
        self.current_cfg = current_cfg or {}
        self.current_mode = bool(self.current_cfg.get("enable", False))
        self.current_out_slices: Dict[str, slice] = {}
        self.current_target = str(self.current_cfg.get("target", "both")).lower()
        self.current_target_overrides = current_target_overrides or {}
        self.hidden_nm_per_neuron = max(0, int(self.nm_cfg.get("hidden_per_neuron", 0)))
        self.output_nm_per_neuron = max(0, int(self.nm_cfg.get("output_per_neuron", 0)))
        self.use_neuromodulators = bool(self.nm_cfg.get("enable")) and (
            self.hidden_nm_per_neuron > 0 or self.output_nm_per_neuron > 0
        )
        self.hidden_group_layout = hidden_group_layout
        self.output_group_layout = output_group_layout
        self.nm_target_overrides = nm_target_overrides or {}
        self.nm_mapper: Optional[NeuromodulatorMapper] = None
        self.nm_out_slices: Dict[str, slice] = {}
        self.weight_out_slices: Dict[str, slice] = {}
        self.hidden_nm_targets = None
        self.output_nm_targets = None
        self.out_slices = base_out_slices
        # NM path: primary outputs raw neuromodulator levels (linear head) so the secondary mapper
        # sees unclamped values. Everything downstream is handled by the secondary MLPs.
        if self.current_mode:
            self.use_neuromodulators = False
        if self.use_neuromodulators and self.mode in {"mlp_sub", "mlp_add", "mlp_combo"}:
            self.output_activation_override = "linear"
        if self.use_neuromodulators:
            hidden_override = self.nm_target_overrides.get("hidden")
            output_override = self.nm_target_overrides.get("output")
            self.hidden_nm_targets = max(0, int(hidden_override)) if hidden_override is not None else (
                self.hidden_group_layout.group_count if self.hidden_group_layout else nb_hidden
            )
            self.output_nm_targets = max(0, int(output_override)) if output_override is not None else (
                self.output_group_layout.group_count if self.output_group_layout else nb_outputs
            )
            self.nm_mapper = NeuromodulatorMapper(
                self.hidden_nm_targets,
                self.output_nm_targets,
                self.hidden_nm_per_neuron,
                self.output_nm_per_neuron,
                self.nm_cfg.get("init_scale", SOFT_MOD_GAIN_INIT),
                cfg=self.nm_cfg,
            )
            self.out_slices = {}
            offset = 0
            if self.hidden_nm_per_neuron > 0 and self.hidden_nm_targets > 0:
                hid_len = self.nm_mapper.hidden_flat_dim
                self.nm_out_slices["hidden"] = slice(offset, offset + hid_len)
                offset += hid_len
            if self.output_nm_per_neuron > 0 and self.output_nm_targets > 0:
                out_len = self.nm_mapper.output_flat_dim
                self.nm_out_slices["output"] = slice(offset, offset + out_len)
                offset += out_len
            for name in WEIGHT_PARAM_NAMES:
                s = base_out_slices.get(name)
                if s is None:
                    continue
                length = int(s.stop - s.start)
                if length <= 0:
                    continue
                self.weight_out_slices[name] = slice(offset, offset + length)
                offset += length
            if self.weight_out_slices:
                self.out_slices = dict(self.weight_out_slices)
            self.output_size = self.nm_mapper.total_output_dim
            if offset > self.output_size:
                self.output_size = offset
        else:
            self.out_slices = base_out_slices

        if self.current_mode:
            current_hidden = self.current_target_overrides.get("hidden", nb_hidden)
            current_output = self.current_target_overrides.get("output", nb_outputs)
            if self.current_target == "hidden":
                current_output = 0
            elif self.current_target == "output":
                current_hidden = 0
            offset = 0
            if current_hidden > 0:
                self.current_out_slices["hidden_current"] = slice(offset, offset + int(current_hidden))
                offset += int(current_hidden)
            if current_output > 0:
                self.current_out_slices["output_current"] = slice(offset, offset + int(current_output))
                offset += int(current_output)
            self.output_size = offset
            self.out_slices = dict(self.current_out_slices)
            current_activation = (self.current_cfg.get("activation") or "").lower()
            if current_activation in {"sigmoid", "tanh"}:
                self.output_activation_override = current_activation

        if self.arch == "mlp":
            self._build_mlp_core(sizes)
        else:
            self._build_recurrent_core(sizes)

    def _build_mlp_core(self, sizes: List[int]):
        layers: List[nn.Module] = []
        linear_layers: List[nn.Linear] = []
        in_dim = self.input_size
        for sz in sizes:
            lin = nn.Linear(in_dim, sz)
            layers.extend([lin, nn.ReLU()])
            linear_layers.append(lin)
            in_dim = sz
        out_lin = nn.Linear(in_dim, self.output_size)
        final_activation, activation_name = _final_activation_for_mode(self.mode, self.output_activation_override)
        self.output_activation_kind = activation_name
        layers.extend([out_lin, final_activation])
        linear_layers.append(out_lin)
        self.layers = nn.Sequential(*layers)
        self._linear_layers = linear_layers
        if self.mode == "mlp_add":
            self._init_additive_weights_soft()
        elif self.mode == "mlp_combo":
            self._init_additive_weights_soft()
        elif self.mode == "mlp_sub":
            self._init_partial_identity()
        else:
            self._init_additive_weights()

    def _build_recurrent_core(self, sizes: List[int]):
        if not sizes:
            sizes = [2048]
        hidden_size = int(sizes[0])
        if any(int(sz) != hidden_size for sz in sizes):
            raise ValueError("All hidden sizes must match when using an RNN/LSTM modulator.")
        num_layers = max(1, len(sizes))
        rnn_kwargs = dict(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        if self.arch == "rnn":
            rnn_kwargs["nonlinearity"] = "tanh"
            self.core = nn.RNN(**rnn_kwargs)
        else:
            self.core = nn.LSTM(**rnn_kwargs)
        self.output_layer = nn.Linear(hidden_size, self.output_size)
        self._linear_layers = [self.output_layer]
        self.rnn_hidden_size = hidden_size
        self.rnn_num_layers = num_layers
        final_activation, activation_name = _final_activation_for_mode(self.mode, self.output_activation_override)
        self.final_activation = final_activation
        self.output_activation_kind = activation_name
        if self.mode == "mlp_add":
            self._init_additive_weights_soft()
        elif self.mode == "mlp_combo":
            self._init_additive_weights_soft()

    def _init_partial_identity(self):
        linear_layers = getattr(self, "_linear_layers", None) or []
        if not linear_layers:
            return
        first, last = linear_layers[0], linear_layers[-1]
        # For substitution (snn_sub / mlp_sub), default to full passthrough on all overlapping dims.
        K = min(first.in_features, first.out_features, last.in_features, last.out_features)
        with torch.no_grad():
            for lin in (first, last):
                lin.weight.zero_()
                if lin.bias is not None:
                    lin.bias.zero_()
            if K > 0:
                eyeK = torch.eye(K, dtype=first.weight.dtype, device=first.weight.device)
                first.weight[:K, :K].copy_(eyeK)
                last.weight[:K, :K].copy_(eyeK)

    def _init_additive_weights(self, scale: float = TINY_MOD_INIT_SCALE):
        linear_layers = getattr(self, "_linear_layers", None) or []
        if not linear_layers:
            return
        for lin in linear_layers:
            _init_tiny_linear(lin, scale=scale)

    def _init_additive_weights_soft(self, hidden_scale: float = SOFT_MOD_INIT_SCALE, final_std: float = ANN_ADD_FINAL_STD):
        """
        Soft-start for additive modulation: tiny weights in hidden layers, very small random
        output layer with zero bias so initial deltas are near-zero but not silent.
        """
        linear_layers = getattr(self, "_linear_layers", None) or []
        if not linear_layers:
            return
        # Hidden layers stay tiny/near-identity
        for lin in linear_layers[:-1]:
            _init_tiny_linear(lin, scale=hidden_scale)
        # Final layer gets a small random normal; bias zeroed for neutrality
        last = linear_layers[-1]
        with torch.no_grad():
            last.weight.normal_(mean=0.0, std=final_std)
            if last.bias is not None:
                last.bias.zero_()

    def _zero_state(self, batch_size: int, device, dtype):
        if not self.stateful or batch_size <= 0:
            return None
        layers = self.rnn_num_layers or 1
        hidden = self.rnn_hidden_size or self.output_size
        zeros = torch.zeros(layers, batch_size, hidden, device=device, dtype=dtype)
        if self.arch == "lstm":
            return (zeros.clone(), zeros.clone())
        return zeros

    def reset_sequence_state(self, batch_size: int = 0, device=None, dtype=None):
        if not self.stateful:
            return
        if batch_size <= 0:
            self._state = None
            self._state_batch = None
            return
        ref = self.output_layer if self.output_layer is not None else self._linear_layers[-1]
        dev = device or ref.weight.device
        dt = dtype or ref.weight.dtype
        self._state = self._zero_state(batch_size, dev, dt)
        self._state_batch = batch_size

    def _ensure_state(self, batch_size: int, device, dtype):
        if not self.stateful:
            return None
        if (self._state is None) or (self._state_batch != batch_size):
            self.reset_sequence_state(batch_size, device=device, dtype=dtype)
        return self._state

    def forward(self, x):
        if self.arch == "mlp":
            return self.layers(x)
        batch_size = x.size(0)
        seq = x.unsqueeze(1)
        state = self._ensure_state(batch_size, x.device, x.dtype)
        if self.arch == "lstm":
            out, new_state = self.core(seq, state)
        else:
            out, new_state = self.core(seq, state)
        self._state = new_state
        self._state_batch = batch_size
        logits = self.output_layer(out[:, -1, :])
        if self.final_activation is not None:
            return self.final_activation(logits)
        return logits


class SpikingLinearLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_step: float,
        tau_syn: float,
        tau_mem: float,
        weight_scale: float = SOFT_MOD_INIT_SCALE,
        recurrent: bool = False,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_scale = weight_scale
        self.recurrent = recurrent
        self.param_ranges = param_ranges or PARAMETER_RANGE_DEFAULTS

        self.weight = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.alpha = nn.Parameter(torch.full((1, out_dim), float(np.exp(-time_step / max(tau_syn, 1e-6)))))
        self.beta = nn.Parameter(torch.full((1, out_dim), float(np.exp(-time_step / max(tau_mem, 1e-6)))))
        self.threshold = nn.Parameter(torch.empty((1, out_dim)))
        self.reset = nn.Parameter(torch.empty((1, out_dim)))
        self.rest = nn.Parameter(torch.empty((1, out_dim)))
        if recurrent:
            self.rec_weight = nn.Parameter(torch.zeros(out_dim, out_dim))
        else:
            self.register_parameter("rec_weight", None)

        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            _init_tiny_diag(self.weight, scale=self.weight_scale)
            if self.rec_weight is not None:
                _init_tiny_diag(self.rec_weight, scale=self.weight_scale)
            thr_lo, thr_hi = self.param_ranges.get("thr", (0.5, 1.5))
            reset_lo, reset_hi = self.param_ranges.get("reset", (-0.5, 0.5))
            rest_lo, rest_hi = self.param_ranges.get("rest", (-0.5, 0.5))
            self.threshold.uniform_(float(thr_lo), float(thr_hi))
            self.reset.uniform_(float(reset_lo), float(reset_hi))
            self.rest.uniform_(float(rest_lo), float(rest_hi))

    def zero_state(self, batch_size: int, device, dtype) -> Dict[str, torch.Tensor]:
        zeros = torch.zeros((batch_size, self.out_dim), device=device, dtype=dtype)
        return {"syn": zeros.clone(), "mem": zeros.clone(), "spikes": zeros.clone()}

    def forward_step(self, inputs: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h = torch.matmul(inputs, self.weight)
        if self.recurrent and self.rec_weight is not None:
            h = h + torch.matmul(state["spikes"], self.rec_weight)

        alpha = self.alpha.expand_as(state["syn"])
        beta = self.beta.expand_as(state["mem"])
        thr = self.threshold.expand_as(state["mem"])
        rst = self.reset.expand_as(state["mem"])
        rest = self.rest.expand_as(state["mem"])

        mthr = state["mem"] - thr
        spk = spike_fn(mthr)
        rst_mask = (mthr > 0).float()

        syn = alpha * state["syn"] + h
        mem = beta * (state["mem"] - rest) + rest + (1 - beta) * syn - rst_mask * (thr - rst)
        return spk, {"syn": syn, "mem": mem, "spikes": spk}


class NonSpikingLinearLayer(nn.Module):
    """
    Linear layer with synaptic and membrane filtering but no spiking/thresholding.
    Used for parameter readouts where the membrane itself encodes the value.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_step: float,
        tau_syn: float,
        tau_mem: float,
        weight_scale: float = SOFT_MOD_INIT_SCALE,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_scale = weight_scale

        self.weight = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.alpha = nn.Parameter(torch.full((1, out_dim), float(np.exp(-time_step / max(tau_syn, 1e-6)))))
        self.beta = nn.Parameter(torch.full((1, out_dim), float(np.exp(-time_step / max(tau_mem, 1e-6)))))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            _init_tiny_diag(self.weight, scale=self.weight_scale)

    def zero_state(self, batch_size: int, device, dtype) -> Dict[str, torch.Tensor]:
        zeros = torch.zeros((batch_size, self.out_dim), device=device, dtype=dtype)
        return {"syn": zeros.clone(), "mem": zeros.clone()}

    def forward_step(self, inputs: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h = torch.matmul(inputs, self.weight)
        alpha = self.alpha.expand_as(state["syn"])
        beta = self.beta.expand_as(state["mem"])
        syn = alpha * state["syn"] + h
        baseline = state.get("baseline")
        if baseline is None:
            baseline = torch.zeros_like(state["mem"])
        # Keep input influence decoupled from the decay-to-baseline term:
        # - With no input, mem decays toward `baseline` with rate beta.
        # - With input, syn is added directly (not scaled by (1-beta)).
        mem = beta * (state["mem"] - baseline) + baseline + syn
        return mem, {"syn": syn, "mem": mem, "baseline": baseline}


class SNNSubstitutionModulator(nn.Module):
    """Secondary SNN that directly proposes parameter values via output membrane potentials."""

    def __init__(
        self,
        nb_inputs: int,
        nb_hidden: int,
        nb_outputs: int,
        time_step: float,
        tau_syn: float,
        tau_mem: float,
        hidden_layer_sizes: Optional[List[int]] = None,
        hidden_recurrent: bool = False,
        weight_scale: float = SOFT_MOD_INIT_SCALE,
        scale_init: float = SNN_SUB_SCALE_INIT_DEFAULT,
        bias_init: float = SNN_SUB_BIAS_INIT_DEFAULT,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        nm_cfg: Optional[Dict] = None,
        hidden_group_layout: Optional['GroupLayout'] = None,
        output_group_layout: Optional['GroupLayout'] = None,
        param_hidden_dim: Optional[int] = None,
        param_output_dim: Optional[int] = None,
        spike_hidden_dim: Optional[int] = None,
        spike_output_dim: Optional[int] = None,
        effect_hidden_dim: Optional[int] = None,
        effect_output_dim: Optional[int] = None,
        weight_effect_dims: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_outputs = nb_outputs
        self.time_step = time_step
        self.tau_syn = tau_syn
        self.tau_mem = tau_mem
        self.hidden_layer_sizes = list(hidden_layer_sizes or [])
        self.hidden_recurrent = hidden_recurrent
        self.weight_scale = weight_scale
        self.scale_init = float(scale_init)
        self.bias_init = float(bias_init)
        self.param_ranges = param_ranges or PARAMETER_RANGE_DEFAULTS
        self._param_dims: Dict[str, int] = {}
        self.nm_cfg = nm_cfg or {}
        self.hidden_nm_per_neuron = max(0, int(self.nm_cfg.get("hidden_per_neuron", 0)))
        self.output_nm_per_neuron = max(0, int(self.nm_cfg.get("output_per_neuron", 0)))
        self.use_neuromodulators = bool(self.nm_cfg.get("enable")) and (
            self.hidden_nm_per_neuron > 0 or self.output_nm_per_neuron > 0
        )
        self.nm_init_scale = float(self.nm_cfg.get("init_scale", weight_scale))
        self.nm_mapper: Optional[NeuromodulatorMapper] = None
        self.weight_effect_dims = {k: int(v) for k, v in (weight_effect_dims or {}).items() if int(v) > 0}

        self.hidden_group_layout = hidden_group_layout
        self.output_group_layout = output_group_layout
        hidden_effect = self.hidden_group_layout.group_count if self.hidden_group_layout else nb_hidden
        output_effect = self.output_group_layout.group_count if self.output_group_layout else nb_outputs
        if effect_hidden_dim is not None:
            hidden_effect = int(effect_hidden_dim)
        if effect_output_dim is not None:
            output_effect = int(effect_output_dim)
        self.hidden_effect_dim = max(0, int(hidden_effect))
        self.output_effect_dim = max(0, int(output_effect))

        self.param_hidden_dim = max(0, int(param_hidden_dim)) if param_hidden_dim is not None else nb_hidden
        self.param_output_dim = max(0, int(param_output_dim)) if param_output_dim is not None else nb_outputs
        self.spike_hidden_dim = max(0, int(spike_hidden_dim)) if spike_hidden_dim is not None else nb_hidden
        self.spike_output_dim = max(0, int(spike_output_dim)) if spike_output_dim is not None else nb_outputs

        self.weight_feature_dim = sum(self.weight_effect_dims.values())
        self.feature_block = 5 * self.param_hidden_dim + 2 * self.param_output_dim + self.weight_feature_dim
        self.effect_block = 5 * self.hidden_effect_dim + 2 * self.output_effect_dim + self.weight_feature_dim
        self.param_block = self.feature_block
        self.input_dim = nb_inputs + self.spike_hidden_dim + self.spike_output_dim + 1 + self.param_block
        if self.use_neuromodulators:
            self.nm_mapper = NeuromodulatorMapper(
                self.hidden_effect_dim,
                self.output_effect_dim,
                self.hidden_nm_per_neuron,
                self.output_nm_per_neuron,
                self.nm_init_scale,
                cfg=self.nm_cfg,
            )
            self.output_dim = self.nm_mapper.total_output_dim
        else:
            self.output_dim = self.effect_block

        layer_dims = [self.input_dim] + self.hidden_layer_sizes + [self.output_dim]
        self.layers = nn.ModuleList()
        for idx in range(len(layer_dims) - 1):
            in_dim = layer_dims[idx]
            out_dim = layer_dims[idx + 1]
            is_hidden = idx < len(layer_dims) - 2
            recurrent = self.hidden_recurrent and is_hidden
            if is_hidden:
                self.layers.append(
                    SpikingLinearLayer(
                        in_dim,
                        out_dim,
                        time_step=time_step,
                        tau_syn=tau_syn,
                        tau_mem=tau_mem,
                        weight_scale=weight_scale,
                        recurrent=recurrent,
                        param_ranges=self.param_ranges,
                    )
                )
            else:
                self.layers.append(
                    NonSpikingLinearLayer(
                        in_dim,
                        out_dim,
                        time_step=time_step,
                        tau_syn=tau_syn,
                        tau_mem=tau_mem,
                        weight_scale=weight_scale,
                    )
                )

        self.mod_hidden = self.layers[-1].out_dim
        if not self.use_neuromodulators:
            self._build_output_slices()
            scales: Dict[str, nn.Parameter] = {}
            biases: Dict[str, nn.Parameter] = {}
            for name, dim in self._param_dims.items():
                scales[name] = nn.Parameter(torch.full((1, dim), float(self.scale_init)))
                biases[name] = nn.Parameter(torch.full((1, dim), float(self.bias_init)))
            self.param_scale = nn.ParameterDict(scales)
            self.param_bias = nn.ParameterDict(biases)
            # Prefer "no decay" for direct parameter readout units (they should hold values unless driven).
            out_layer = self.layers[-1]
            if isinstance(out_layer, NonSpikingLinearLayer):
                with torch.no_grad():
                    out_layer.beta.fill_(1.0)
        else:
            self.param_scale = nn.ParameterDict({})
            self.param_bias = nn.ParameterDict({})
        self._init_random_weights_like_primary()

    def _init_random_weights_like_primary(self):
        # For snn_sub, prefer a random init closer to the primary SNN rather than a tiny diagonal.
        # Keep this localized to SNNSubstitutionModulator so snn_add behavior is unchanged.
        with torch.no_grad():
            for layer in self.layers:
                w = getattr(layer, "weight", None)
                if isinstance(w, torch.Tensor) and w.dim() == 2:
                    in_dim = max(1, int(w.size(0)))
                    # Use at least ~primary-SNN-scale init unless user explicitly increases it.
                    std = float(max(self.weight_scale, 2e-2))
                    nn.init.normal_(w, mean=0.0, std=std / math.sqrt(in_dim))
                rec = getattr(layer, "rec_weight", None)
                if isinstance(rec, torch.Tensor) and rec.dim() == 2:
                    d = max(1, int(rec.size(0)))
                    std = float(max(self.weight_scale, 2e-2))
                    nn.init.normal_(rec, mean=0.0, std=std / math.sqrt(d))

    def _build_output_slices(self):
        self._out_slices: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for name, dim in [
            ("alpha_1", self.hidden_effect_dim),
            ("beta_1", self.hidden_effect_dim),
            ("thr", self.hidden_effect_dim),
            ("reset", self.hidden_effect_dim),
            ("rest", self.hidden_effect_dim),
            ("alpha_2", self.output_effect_dim),
            ("beta_2", self.output_effect_dim),
        ]:
            start = offset
            end = start + dim
            self._out_slices[name] = (start, end)
            self._param_dims[name] = dim
            offset = end
        for name in WEIGHT_PARAM_NAMES:
            dim = int(self.weight_effect_dims.get(name, 0))
            if dim <= 0:
                continue
            start = offset
            end = start + dim
            self._out_slices[name] = (start, end)
            self._param_dims[name] = dim
            offset = end
        if offset != self.output_dim:
            raise ValueError("Substitution slice construction failed: mismatched neuron count.")

    def modulation_effects(self, output_mem: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.use_neuromodulators and self.nm_mapper is not None:
            return self.nm_mapper.effects_from_flat(output_mem)
        return self.decode_parameters(output_mem, self.param_ranges)

    def zero_state(self, batch_size: int, init_params: Optional[Dict[str, torch.Tensor]] = None) -> List[Dict[str, torch.Tensor]]:
        device = self.layers[0].weight.device
        dtype = self.layers[0].weight.dtype
        states = [layer.zero_state(batch_size, device, dtype) for layer in self.layers]
        # Initialize output membrane so decoded parameters match the current SNN parameters.
        # Also store a baseline so output units decay back to the init values when un-driven.
        out_state = states[-1]
        if "mem" in out_state:
            if self.use_neuromodulators:
                out_state["mem"] = out_state["mem"].zero_()
            else:
                base_mem = torch.zeros((batch_size, self.output_dim), device=device, dtype=dtype)
                if init_params:
                    eps = 1e-6
                    for name, (start, end) in self._out_slices.items():
                        tensor = init_params.get(name)
                        if tensor is None:
                            continue
                        val = tensor.to(device=device, dtype=dtype)
                        if name in WEIGHT_PARAM_NAMES:
                            ref = val.detach().abs().amax(dim=1, keepdim=True).clamp_min(eps)
                            norm = torch.clamp(val / ref, min=-1.0 + eps, max=1.0 - eps)
                            z0 = torch.atanh(norm)
                            scale = self.param_scale[name].to(device=device, dtype=dtype)
                            bias = self.param_bias[name].to(device=device, dtype=dtype)
                            scale_safe = torch.where(scale.abs() < eps, scale.new_full(scale.shape, eps), scale)
                            base_mem[:, start:end] = (z0 - bias) / scale_safe
                            continue
                        lo, hi = self.param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-1.0, 1.0)))
                        lo_f = float(lo); hi_f = float(hi)
                        mid = 0.5 * (lo_f + hi_f)
                        half = 0.5 * (hi_f - lo_f)
                        if half <= 0:
                            base_mem[:, start:end] = 0.0
                            continue
                        norm = (val - mid) / half
                        norm = torch.clamp(norm, min=-1.0 + eps, max=1.0 - eps)
                        z0 = torch.atanh(norm)
                        scale = self.param_scale[name].to(device=device, dtype=dtype)
                        bias = self.param_bias[name].to(device=device, dtype=dtype)
                        scale_safe = torch.where(scale.abs() < eps, scale.new_full(scale.shape, eps), scale)
                        base_mem[:, start:end] = (z0 - bias) / scale_safe
                out_state["mem"] = base_mem
                out_state["baseline"] = base_mem.clone()
        return states

    def build_features(
        self,
        alpha_1: torch.Tensor,
        beta_1: torch.Tensor,
        thr: torch.Tensor,
        reset: torch.Tensor,
        rest: torch.Tensor,
        alpha_2: torch.Tensor,
        beta_2: torch.Tensor,
        input_t: torch.Tensor,
        hidden_spikes: torch.Tensor,
        readout_spikes: torch.Tensor,
        step_idx: int,
        weight_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        param_parts = [alpha_1, beta_1, thr, reset, rest, alpha_2, beta_2]
        if weight_features:
            param_parts.extend(weight_features)
        param_stack = torch.cat(param_parts, dim=1)
        time_feat = torch.full((input_t.size(0), 1), float(step_idx), device=input_t.device, dtype=input_t.dtype)
        return torch.cat(
            [param_stack, time_feat, input_t, hidden_spikes, readout_spikes],
            dim=1
        )

    def forward_step(
        self,
        features: torch.Tensor,
        states: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        x = features
        new_states = []
        for layer, state in zip(self.layers, states):
            spk, next_state = layer.forward_step(x, state)
            new_states.append(next_state)
            x = spk
        return x, new_states

    def decode_parameters(self, mem_values: torch.Tensor, param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, torch.Tensor]:
        effects: Dict[str, torch.Tensor] = {}
        for name, (start, end) in self._out_slices.items():
            raw = mem_values[:, start:end]
            scale = self.param_scale[name].to(device=raw.device, dtype=raw.dtype)
            bias = self.param_bias[name].to(device=raw.device, dtype=raw.dtype)
            z = scale * raw + bias
            if name in WEIGHT_PARAM_NAMES:
                effects[name] = torch.tanh(z)
                continue
            lo, hi = param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
            lo_f = float(lo); hi_f = float(hi)
            mid = 0.5 * (lo_f + hi_f)
            half = 0.5 * (hi_f - lo_f)
            # Soft-bounded mapping: diminishing effect near boundaries without hard clamping.
            effects[name] = mid + half * torch.tanh(z)
        return effects


class SNNAdditiveModulator(nn.Module):
    """Secondary SNN ("Double" architecture) with optional hidden spiking layers."""

    def __init__(
        self,
        nb_inputs: int,
        nb_hidden: int,
        nb_outputs: int,
        time_step: float,
        tau_syn: float,
        tau_mem: float,
        hidden_layer_sizes: Optional[List[int]] = None,
        hidden_recurrent: bool = False,
        weight_scale: float = SOFT_MOD_INIT_SCALE,
        gain_init: float = SOFT_MOD_GAIN_INIT,
        nm_cfg: Optional[Dict] = None,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        hidden_group_layout: Optional['GroupLayout'] = None,
        output_group_layout: Optional['GroupLayout'] = None,
        param_hidden_dim: Optional[int] = None,
        param_output_dim: Optional[int] = None,
        spike_hidden_dim: Optional[int] = None,
        spike_output_dim: Optional[int] = None,
        effect_hidden_dim: Optional[int] = None,
        effect_output_dim: Optional[int] = None,
        balanced_init: bool = False,
        init_effect_frac: float = 0.05,
        weight_effect_dims: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_outputs = nb_outputs
        self.time_step = time_step
        self.tau_syn = tau_syn
        self.tau_mem = tau_mem
        self.hidden_layer_sizes = list(hidden_layer_sizes or [])
        self.hidden_recurrent = hidden_recurrent
        self.weight_scale = weight_scale
        self._param_dims: Dict[str, int] = {}
        self.param_ranges = param_ranges or PARAMETER_RANGE_DEFAULTS
        self.nm_cfg = nm_cfg or {}
        self.hidden_nm_per_neuron = max(0, int(self.nm_cfg.get("hidden_per_neuron", 0)))
        self.output_nm_per_neuron = max(0, int(self.nm_cfg.get("output_per_neuron", 0)))
        self.use_neuromodulators = bool(self.nm_cfg.get("enable")) and (
            self.hidden_nm_per_neuron > 0 or self.output_nm_per_neuron > 0
        )
        self.nm_init_scale = float(self.nm_cfg.get("init_scale", gain_init))
        self.shared_mod_gain: Optional[nn.ParameterDict] = None
        self.balanced_init = bool(balanced_init)
        self.init_effect_frac = float(init_effect_frac)
        self.weight_effect_dims = {k: int(v) for k, v in (weight_effect_dims or {}).items() if int(v) > 0}

        self.hidden_group_layout = hidden_group_layout
        self.output_group_layout = output_group_layout
        hidden_effect = self.hidden_group_layout.group_count if self.hidden_group_layout else nb_hidden
        output_effect = self.output_group_layout.group_count if self.output_group_layout else nb_outputs
        if effect_hidden_dim is not None:
            hidden_effect = int(effect_hidden_dim)
        if effect_output_dim is not None:
            output_effect = int(effect_output_dim)
        self.hidden_effect_dim = max(0, int(hidden_effect))
        self.output_effect_dim = max(0, int(output_effect))

        self.param_hidden_dim = max(0, int(param_hidden_dim)) if param_hidden_dim is not None else nb_hidden
        self.param_output_dim = max(0, int(param_output_dim)) if param_output_dim is not None else nb_outputs
        self.spike_hidden_dim = max(0, int(spike_hidden_dim)) if spike_hidden_dim is not None else nb_hidden
        self.spike_output_dim = max(0, int(spike_output_dim)) if spike_output_dim is not None else nb_outputs

        self.weight_feature_dim = sum(self.weight_effect_dims.values())
        self.feature_block = 5 * self.param_hidden_dim + 2 * self.param_output_dim + self.weight_feature_dim
        self.effect_block = 5 * self.hidden_effect_dim + 2 * self.output_effect_dim + self.weight_feature_dim
        self.param_block = self.feature_block
        self.input_dim = nb_inputs + self.spike_hidden_dim + self.spike_output_dim + 1 + self.param_block
        self.nm_mapper: Optional[NeuromodulatorMapper] = None
        if self.use_neuromodulators:
            self.nm_mapper = NeuromodulatorMapper(
                self.hidden_effect_dim,
                self.output_effect_dim,
                self.hidden_nm_per_neuron,
                self.output_nm_per_neuron,
                self.nm_init_scale,
                cfg=self.nm_cfg,
            )
            self.output_dim = self.nm_mapper.total_output_dim
        else:
            self.output_dim = 2 * self.effect_block

        layer_dims = [self.input_dim] + self.hidden_layer_sizes + [self.output_dim]
        self.layers = nn.ModuleList()
        for idx in range(len(layer_dims) - 1):
            in_dim = layer_dims[idx]
            out_dim = layer_dims[idx + 1]
            is_hidden = idx < len(layer_dims) - 2
            recurrent = self.hidden_recurrent and is_hidden
            self.layers.append(
                SpikingLinearLayer(
                    in_dim,
                    out_dim,
                    time_step=time_step,
                    tau_syn=tau_syn,
                    tau_mem=tau_mem,
                    weight_scale=weight_scale,
                    recurrent=recurrent,
                    param_ranges=self.param_ranges,
                )
            )

        self.mod_hidden = self.layers[-1].out_dim
        if self.use_neuromodulators:
            self._build_neuromodulators()
            self.mod_factors = None
            self.shared_mod_gain = None
        else:
            self._build_modulation_slices()
            factors = torch.full((1, self.mod_hidden), float(max(gain_init, 0.0)))
            if gain_init > 0:
                nn.init.normal_(factors, mean=float(gain_init), std=float(gain_init))
            self.mod_factors = nn.Parameter(factors)
            if self.balanced_init:
                self._apply_balanced_init()
            self.shared_mod_gain = None

    def _apply_balanced_init(self):
        if self.mod_factors is None:
            return
        if not hasattr(self, "_mod_slices"):
            return
        out_layer = self.layers[-1] if self.layers else None
        if not isinstance(out_layer, SpikingLinearLayer):
            return
        with torch.no_grad():
            # Force paired positive/negative neurons to be identical so effects cancel at init.
            for _, (start, mid, end) in self._mod_slices.items():
                if end <= mid or mid <= start:
                    continue
                out_layer.weight[:, mid:end] = out_layer.weight[:, start:mid]
                out_layer.alpha[:, mid:end] = out_layer.alpha[:, start:mid]
                out_layer.beta[:, mid:end] = out_layer.beta[:, start:mid]
                out_layer.threshold[:, mid:end] = out_layer.threshold[:, start:mid]
                out_layer.reset[:, mid:end] = out_layer.reset[:, start:mid]
                out_layer.rest[:, mid:end] = out_layer.rest[:, start:mid]

            # Initialize gains so a 1-spike imbalance corresponds to ~init_effect_frac of each param's range width.
            frac = max(0.0, float(self.init_effect_frac))
            for name, (start, mid, end) in self._mod_slices.items():
                lo, hi = self.param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-1.0, 1.0)))
                width = float(hi) - float(lo)
                gain = frac * width
                self.mod_factors.data[:, start:mid] = gain
                self.mod_factors.data[:, mid:end] = gain

    def _build_modulation_slices(self):
        self._mod_slices: Dict[str, Tuple[int, int, int]] = {}
        offset = 0
        for name, dim in [
            ("alpha_1", self.hidden_effect_dim),
            ("beta_1", self.hidden_effect_dim),
            ("thr", self.hidden_effect_dim),
            ("reset", self.hidden_effect_dim),
            ("rest", self.hidden_effect_dim),
            ("alpha_2", self.output_effect_dim),
            ("beta_2", self.output_effect_dim),
        ]:
            start = offset
            mid = start + dim
            end = mid + dim
            self._mod_slices[name] = (start, mid, end)
            self._param_dims[name] = dim
            offset = end
        for name in WEIGHT_PARAM_NAMES:
            dim = int(self.weight_effect_dims.get(name, 0))
            if dim <= 0:
                continue
            start = offset
            mid = start + dim
            end = mid + dim
            self._mod_slices[name] = (start, mid, end)
            self._param_dims[name] = dim
            offset = end
        if offset != self.mod_hidden:
            raise ValueError("Modulation slice construction failed: mismatched neuron count.")

    def zero_state(self, batch_size: int, init_params: Optional[Dict[str, torch.Tensor]] = None) -> List[Dict[str, torch.Tensor]]:
        device = self.layers[0].weight.device
        dtype = self.layers[0].weight.dtype
        states = [layer.zero_state(batch_size, device, dtype) for layer in self.layers]
        return states

    def build_features(
        self,
        alpha_1: torch.Tensor,
        beta_1: torch.Tensor,
        thr: torch.Tensor,
        reset: torch.Tensor,
        rest: torch.Tensor,
        alpha_2: torch.Tensor,
        beta_2: torch.Tensor,
        input_t: torch.Tensor,
        hidden_spikes: torch.Tensor,
        readout_spikes: torch.Tensor,
        step_idx: int,
        weight_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        param_parts = [alpha_1, beta_1, thr, reset, rest, alpha_2, beta_2]
        if weight_features:
            param_parts.extend(weight_features)
        param_stack = torch.cat(param_parts, dim=1)
        time_feat = torch.full((input_t.size(0), 1), float(step_idx), device=input_t.device, dtype=input_t.dtype)
        return torch.cat(
            [param_stack, time_feat, input_t, hidden_spikes, readout_spikes],
            dim=1
        )

    def forward_step(
        self,
        features: torch.Tensor,
        states: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        x = features
        new_states = []
        for layer, state in zip(self.layers, states):
            spk, next_state = layer.forward_step(x, state)
            new_states.append(next_state)
            x = spk
        return x, new_states

    def _build_neuromodulators(self):
        # NeuromodulatorMapper is constructed in __init__; nothing to do here.
        return

    def modulation_effects(self, spike_accum: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.use_neuromodulators:
            if self.nm_mapper is None:
                return {}
            return self.nm_mapper.effects_from_flat(spike_accum)
        effects = {}
        for name, (start, mid, end) in self._mod_slices.items():
            gains_pos = self.mod_factors[:, start:mid]
            gains_neg = self.mod_factors[:, mid:end]
            # Element-wise +/- contributions: keep "which neuron spiked" information.
            pos_spikes = spike_accum[:, start:mid]
            neg_spikes = spike_accum[:, mid:end]
            base = pos_spikes * gains_pos - neg_spikes * gains_neg
            effects[name] = base
        return effects

    def clamp_mod_factors_(self):
        if self.mod_factors is None:
            return
        with torch.no_grad():
            self.mod_factors.clamp_(min=0.0)


def build_modulator(
    settings: Dict,
    override_mode: Optional[str] = None,
    hidden_sizes: Optional[List[int]] = None,
) -> nn.Module:
    mode = _select_mlp_mode(settings, override=override_mode)
    settings["_mod_output_size"] = None
    param_ranges = _resolve_param_ranges(settings)
    substitution_mode = _is_substitution_mode(mode)
    mod_input_size = int(settings.get("nb_inputs_mod", settings["nb_inputs"]))
    smoothing_cfg = settings.get("param_smoothing", {})
    smoothing_modes = {_normalize_mlp_mode(m) for m in smoothing_cfg.get("modes", [])} if smoothing_cfg else set()
    smoothing_allowed = (
        bool(smoothing_cfg.get("enable"))
        and (not smoothing_modes or mode in smoothing_modes)
    )
    hidden_layout = _get_group_layout(settings, 0, settings["nb_hidden"])
    output_layout = _get_group_layout(settings, 1, settings["nb_outputs"])
    weight_size_override = _weight_size_overrides(settings)
    mod_mask = _build_fixed_mod_mask(settings, hidden_layout, output_layout)
    if _is_snn_mode(mode):
        nm_cfg_snn = settings.get("nm_cfg", {}) or {}
        nm_active_snn = bool(nm_cfg_snn.get("enable")) and (
            int(nm_cfg_snn.get("hidden_per_neuron", 0)) > 0 or int(nm_cfg_snn.get("output_per_neuron", 0)) > 0
        )
        if weight_size_override and nm_active_snn:
            raise ValueError("weight modulation with nm_enable is currently supported for ANN/RNN/LSTM modulators, not SNN modulators.")
        snn_hidden_layers = hidden_sizes
        if snn_hidden_layers is None:
            snn_hidden_layers = settings.get("mod_hidden_sizes")
        param_hidden_dim = param_output_dim = None
        spike_hidden_dim = spike_output_dim = None
        effect_hidden_dim = effect_output_dim = None
        if mod_mask is not None:
            nm_cfg = settings.get("nm_cfg", {}) or {}
            nm_active = bool(nm_cfg.get("enable")) and (
                int(nm_cfg.get("hidden_per_neuron", 0)) > 0 or int(nm_cfg.get("output_per_neuron", 0)) > 0
            )
            hidden_union = int(mod_mask["hidden_union_idx"].numel())
            output_union = int(mod_mask["output_union_idx"].numel())
            hidden_neuron = int(mod_mask["hidden_neuron_idx"].numel())
            output_neuron = int(mod_mask["output_neuron_idx"].numel())
            if nm_active:
                effect_hidden_dim = hidden_neuron
                effect_output_dim = output_neuron
            else:
                effect_hidden_dim = hidden_union
                effect_output_dim = output_union
            param_hidden_dim = hidden_union
            param_output_dim = output_union
            if mod_mask.get("flat_inputs") and mod_mask.get("flat_hidden_idx") is not None:
                spike_hidden_dim = int(mod_mask["flat_hidden_idx"].numel())
                spike_output_dim = int(mod_mask["flat_output_idx"].numel()) if mod_mask.get("flat_output_idx") is not None else None
        if mode == "snn_sub":
            mod = SNNSubstitutionModulator(
                nb_inputs=mod_input_size,
                nb_hidden=settings["nb_hidden"],
                nb_outputs=settings["nb_outputs"],
                time_step=settings["time_step"],
                tau_syn=settings.get("tau_syn", 10e-3),
                tau_mem=settings.get("tau_mem", 20e-3),
                hidden_layer_sizes=snn_hidden_layers,
                hidden_recurrent=settings.get("snn_mod_hidden_recurrent", False),
                weight_scale=settings.get("snn_mod_weight_scale", SNN_ADD_WEIGHT_SCALE_DEFAULT),
                scale_init=settings.get("snn_sub_scale_init", SNN_SUB_SCALE_INIT_DEFAULT),
                bias_init=settings.get("snn_sub_bias_init", SNN_SUB_BIAS_INIT_DEFAULT),
                param_ranges=param_ranges,
                nm_cfg=settings.get("nm_cfg", {}),
                hidden_group_layout=hidden_layout,
                output_group_layout=output_layout,
                param_hidden_dim=param_hidden_dim,
                param_output_dim=param_output_dim,
                spike_hidden_dim=spike_hidden_dim,
                spike_output_dim=spike_output_dim,
                effect_hidden_dim=effect_hidden_dim,
                effect_output_dim=effect_output_dim,
                weight_effect_dims=weight_size_override,
            ).to(device)
        else:
            mod = SNNAdditiveModulator(
                nb_inputs=mod_input_size,
                nb_hidden=settings["nb_hidden"],
                nb_outputs=settings["nb_outputs"],
                time_step=settings["time_step"],
                tau_syn=settings.get("tau_syn", 10e-3),
                tau_mem=settings.get("tau_mem", 20e-3),
                hidden_layer_sizes=snn_hidden_layers,
                hidden_recurrent=settings.get("snn_mod_hidden_recurrent", False),
                weight_scale=settings.get("snn_mod_weight_scale", SNN_ADD_WEIGHT_SCALE_DEFAULT),
                gain_init=settings.get("snn_mod_gain_init", SNN_ADD_GAIN_INIT_DEFAULT),
                nm_cfg=settings.get("nm_cfg", {}),
                param_ranges=param_ranges,
                hidden_group_layout=hidden_layout,
                output_group_layout=output_layout,
                param_hidden_dim=param_hidden_dim,
                param_output_dim=param_output_dim,
                spike_hidden_dim=spike_hidden_dim,
                spike_output_dim=spike_output_dim,
                effect_hidden_dim=effect_hidden_dim,
                effect_output_dim=effect_output_dim,
                balanced_init=settings.get("snn_add_balanced_init", False),
                init_effect_frac=settings.get("snn_add_init_effect_frac", 0.05),
                weight_effect_dims=weight_size_override,
            ).to(device)
        if settings.get("snn_mod_hidden_recurrent", False) and settings.get("snn_mod_rec_init_zero", False):
            for layer in getattr(mod, "layers", []):
                rec = getattr(layer, "rec_weight", None)
                if rec is not None:
                    with torch.no_grad():
                        rec.zero_()
        if settings.get("param_timescales", {}).get("enable"):
            mod.param_timescales = ParamTimescaleController(
                settings.get("ann_interval", settings.get("mlp_interval", 3)),
                MOD_TARGET_PARAM_NAMES,
                dist=settings.get("param_timescales", {}).get("distribution", "fixed"),
                scale=settings.get("param_timescales", {}).get("scale", 0.0),
                std=settings.get("param_timescales", {}).get("std", 0.0),
                seed=settings.get("param_timescales", {}).get("seed"),
                trainable=settings.get("param_timescales", {}).get("trainable", True),
            ).to(device)
        if smoothing_allowed:
            mod.param_smoothing = ParamSmoothingController(
                MOD_TARGET_PARAM_NAMES,
                settings["nb_hidden"],
                settings["nb_outputs"],
                tau_init=smoothing_cfg.get("tau_init", PARAM_SMOOTH_TAU_INIT_DEFAULT),
                tau_min=smoothing_cfg.get("tau_min", PARAM_SMOOTH_TAU_MIN_DEFAULT),
                trainable=smoothing_cfg.get("trainable", True),
            ).to(device)
        if settings.get("channel_compress_mode") == "mod_mlp" and settings.get("channel_compress_enable"):
            hidden_sizes = settings.get("channel_compress_mlp_hidden_sizes") or []
            mod.input_compressor = InputCompressionMLP(
                settings["nb_inputs"], mod_input_size, hidden_sizes=hidden_sizes
            ).to(device)
        settings["snn_mod_hidden"] = mod.mod_hidden
        return mod
    resolved_hidden = hidden_sizes if hidden_sizes is not None else _resolve_mlp_hidden_sizes(settings)
    additive_mode = (mode == "mlp_add")
    out_size_override = _ann_out_size_overrides(settings)
    in_size_override = _ann_in_size_overrides(settings, substitution_mode=substitution_mode)
    if weight_size_override:
        in_size_override = dict(in_size_override)
        out_size_override = dict(out_size_override)
        in_size_override.update(weight_size_override)
        out_size_override.update(weight_size_override)
    nm_target_overrides = None
    current_target_overrides = None
    mod_hid_flat_group = bool(settings.get("mod_hid_flat_group", False))
    mod_hid_flat_mod_only = bool(settings.get("mod_hid_flat_modulated_only", False))
    if mod_mask is not None:
        param_counts = {}
        for name, idx in mod_mask.get("hidden_param_idx", {}).items():
            param_counts[name] = int(idx.numel())
        for name, idx in mod_mask.get("output_param_idx", {}).items():
            param_counts[name] = int(idx.numel())
        if param_counts:
            out_size_override = dict(out_size_override)
            in_size_override = dict(in_size_override)
            out_size_override.update(param_counts)
            in_size_override.update(param_counts)
        if mod_mask.get("flat_inputs"):
            if mod_mask.get("flat_hidden_idx") is not None and not (mod_hid_flat_group or mod_hid_flat_mod_only):
                in_size_override["hid_flat"] = int(mod_mask["flat_hidden_idx"].numel())
            if mod_mask.get("flat_output_idx") is not None:
                in_size_override["out_flat"] = int(mod_mask["flat_output_idx"].numel())
        nm_target_overrides = {
            "hidden": int(mod_mask["hidden_neuron_idx"].numel()),
            "output": int(mod_mask["output_neuron_idx"].numel()),
        }
        current_target_overrides = {
            "hidden": int(mod_mask["hidden_union_idx"].numel()),
            "output": int(mod_mask["output_union_idx"].numel()),
        }
    hid_flat_dim = None
    if mod_hid_flat_group:
        hid_flat_dim = hidden_layout.group_count if hidden_layout else settings["nb_hidden"]
    if mod_hid_flat_mod_only and mod_mask is not None:
        if mod_hid_flat_group:
            hid_flat_dim = int(mod_mask["hidden_union_idx"].numel())
        else:
            mod_only_count = _count_modulated_neurons(
                hidden_layout,
                mod_mask.get("hidden_union_idx"),
                settings["nb_hidden"],
            )
            if mod_only_count is not None:
                hid_flat_dim = int(mod_only_count)
    if hid_flat_dim is not None:
        in_size_override = dict(in_size_override)
        in_size_override["hid_flat"] = int(hid_flat_dim)
    output_activation = settings.get("ann_output_activation") or settings.get("mlp_output_activation")
    hidden_nm_layout = _uniform_group_layout(hidden_layout) if substitution_mode else hidden_layout
    output_nm_layout = _uniform_group_layout(output_layout) if substitution_mode else output_layout
    current_cfg = None
    if settings.get("mod_current_enable"):
        current_cfg = {
            "enable": True,
            "target": settings.get("mod_current_target", "both"),
            "activation": settings.get("mod_current_activation", "tanh"),
        }
    ann_mod = ModulatingMLP(
        mod_input_size, settings["nb_hidden"], settings["nb_outputs"],
        hidden_sizes=resolved_hidden,
        in_mask=settings.get("mlp_in_mask", {}),
        out_mask=settings.get("mlp_out_mask", {}),
        mode=mode,
        arch=settings.get("mlp_arch", DEFAULT_MLP_ARCH),
        in_size_override=in_size_override,
        out_size_override=out_size_override,
        output_activation=output_activation,
        nm_cfg=settings.get("nm_cfg", {}),
        nm_target_overrides=nm_target_overrides,
        hidden_group_layout=hidden_nm_layout,
        output_group_layout=output_nm_layout,
        current_cfg=current_cfg,
        current_target_overrides=current_target_overrides,
    ).to(device)
    settings["_mod_output_size"] = int(getattr(ann_mod, "output_size", 0))
    if settings.get("param_timescales", {}).get("enable"):
        ann_mod.param_timescales = ParamTimescaleController(
            settings.get("ann_interval", settings.get("mlp_interval", 3)),
            MOD_TARGET_PARAM_NAMES,
            dist=settings.get("param_timescales", {}).get("distribution", "fixed"),
            scale=settings.get("param_timescales", {}).get("scale", 0.0),
            std=settings.get("param_timescales", {}).get("std", 0.0),
            seed=settings.get("param_timescales", {}).get("seed"),
            trainable=settings.get("param_timescales", {}).get("trainable", True),
        ).to(device)
    if smoothing_allowed:
        ann_mod.param_smoothing = ParamSmoothingController(
            MOD_TARGET_PARAM_NAMES,
            settings["nb_hidden"],
            settings["nb_outputs"],
            tau_init=smoothing_cfg.get("tau_init", PARAM_SMOOTH_TAU_INIT_DEFAULT),
            tau_min=smoothing_cfg.get("tau_min", PARAM_SMOOTH_TAU_MIN_DEFAULT),
            trainable=smoothing_cfg.get("trainable", True),
        ).to(device)
    if settings.get("channel_compress_mode") == "mod_mlp" and settings.get("channel_compress_enable"):
        hidden_sizes = settings.get("channel_compress_mlp_hidden_sizes") or []
        ann_mod.input_compressor = InputCompressionMLP(
            settings["nb_inputs"], mod_input_size, hidden_sizes=hidden_sizes
        ).to(device)
    return ann_mod

# -------------------------
# Load base SNN params from ckpt (for mod run)
# -------------------------
def _collect_stockpile_checkpoints(stockpile_dir: Union[str, Path]) -> List[Path]:
    """Gather candidate checkpoints from a stockpile of Run_* folders."""
    root = Path(os.path.expanduser(str(stockpile_dir))).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Stockpile directory not found: {root}")
    run_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("Run_")]
    if not run_dirs:
        run_dirs = [p for p in root.glob("**/Run_*") if p.is_dir()]
    ckpts: List[Path] = []
    for run_dir in run_dirs:
        fold_dir = run_dir / "Fold_1"
        candidates = [
            fold_dir / "chosen.pth",
            fold_dir / "chosen",
            fold_dir / "snn_best.pth",
            fold_dir / "best_snn.pth",
        ]
        ckpt = next((c for c in candidates if c.exists()), None)
        if ckpt is not None:
            ckpts.append(ckpt)
    return ckpts


def pick_random_stockpile_ckpt(stockpile_dir: Union[str, Path], seed: Optional[int] = None) -> Path:
    """Select a checkpoint from a stockpile directory (optionally seeded for reproducibility)."""
    ckpts = _collect_stockpile_checkpoints(stockpile_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {stockpile_dir}")
    # If seed is provided, pin the choice; otherwise use a fresh RNG so global seeds don't pin it.
    rng = random.Random(int(seed)) if seed is not None else random.Random()
    chosen = rng.choice(ckpts)
    print(f"[info] Using stockpile base SNN: {chosen}")
    return chosen


def load_base_snn_state(ckpt_path: Union[str, Path], train_flags: Optional[Dict[str, bool]] = None, settings: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=device)
    params = ckpt['params'] if 'params' in ckpt else ckpt
    def P(t):
        t = t.to(device)
        return torch.nn.Parameter(t.clone().detach(), requires_grad=True)
    state = {
        "w1": P(params['w1']), "w2": P(params['w2']), "v1": P(params['v1']),
        "alpha_hetero_1": P(params.get('alpha_hetero_1', params.get('alpha'))),
        "beta_hetero_1":  P(params.get('beta_hetero_1',  params.get('beta'))),
        "thresholds_1":   P(params.get('thresholds_1',  params.get('threshold'))),
        "reset_1":        P(params.get('reset_1',       params.get('reset'))),
        "rest_1":         P(params.get('rest_1',        params.get('rest'))),
        "alpha_hetero_2": P(params.get('alpha_hetero_2', params.get('alpha_2'))),
        "beta_hetero_2":  P(params.get('beta_hetero_2',  params.get('beta_2'))),
    }
    if "input_delay_logits" in params:
        state["input_delay_logits"] = P(params["input_delay_logits"])
    elif settings is not None and settings.get("use_input_delay", False) and "nb_inputs" in settings:
        state["input_delay_logits"] = _init_input_delay_logits(settings, int(settings["nb_inputs"]))
    flags = _normalize_train_flags(train_flags)
    return _apply_snn_train_flags(state, flags)

def load_modulated_checkpoint(ckpt_path: Union[str, Path], settings: Dict, quiet: bool = False) -> Tuple[Dict[str, torch.Tensor], nn.Module]:
    ckpt = torch.load(ckpt_path, map_location=device)
    snn_params = ckpt.get("snn_params")
    mlp_state = ckpt.get("mlp_state")
    if snn_params is None or mlp_state is None:
        raise ValueError(f"Checkpoint at {ckpt_path} is missing SNN or MLP parameters.")
    state = {k: torch.nn.Parameter(v.to(device), requires_grad=False) for k, v in snn_params.items()}
    if settings.get("use_input_delay", False) and "input_delay_logits" not in state:
        state["input_delay_logits"] = _init_input_delay_logits(settings, int(settings["nb_inputs"]))
    mode = _select_mlp_mode(settings, ckpt.get("mlp_mode"))

    def _list_from(obj):
        if obj is None:
            return None
        if isinstance(obj, (list, tuple)):
            return [int(v) for v in obj]
        return None

    hidden_sizes = _resolve_mlp_hidden_sizes(settings)
    if hidden_sizes is None:
        hidden_sizes = _list_from(ckpt.get("mlp_hidden_sizes"))

    mod_hidden_sizes = settings.get("mod_hidden_sizes")
    if mod_hidden_sizes is None:
        mod_hidden_sizes = _list_from(ckpt.get("mod_hidden_sizes"))

    if settings.get("snn_mod_hidden") is None and ckpt.get("snn_mod_hidden") is not None:
        settings["snn_mod_hidden"] = ckpt.get("snn_mod_hidden")
    if settings.get("snn_mod_hidden_recurrent") is None:
        settings["snn_mod_hidden_recurrent"] = ckpt.get("snn_mod_hidden_recurrent", False)

    modulator = build_modulator(
        settings,
        override_mode=mode,
        hidden_sizes=mod_hidden_sizes if _is_snn_mode(mode) else hidden_sizes,
    )
    if not quiet:
        print(format_param_stats(state, modulator, prefix="load_modulated_checkpoint:", settings=settings))
    load_status = modulator.load_state_dict(mlp_state, strict=False)
    if not quiet:
        missing = load_status.missing_keys
        unexpected = load_status.unexpected_keys
        if missing:
            print(f"[warn] Missing keys when loading modulator: {missing}")
        if unexpected:
            print(f"[warn] Unexpected keys when loading modulator: {unexpected}")
    modulator.eval()
    return state, modulator

# -------------------------
# Modulated forward
# -------------------------
def run_snn_modulated(
    inputs,
    state: Dict,
    settings: Dict,
    modulator: nn.Module,
    mlp_interval: int,
    mlp_input_block_mask: Optional[Dict[str, bool]] = None,
    mlp_output_block_mask: Optional[Dict[str, bool]] = None,
    trace_fn: Optional[Callable[[int, Dict[str, torch.Tensor]], None]] = None,
    training: bool = False,
):
    """
    Core simulation loop. At each interval:
      - Gather recent spikes/params for the primary modulator (MLP or SNN).
      - If NM is enabled, the primary emits raw neuromodulator levels (linear head).
        The shared secondary NM mapper (sigmoid/tanh head) turns those into per-param
        values/deltas, which are then applied (substitution or additive).
      - Otherwise, apply the primary modulator outputs directly.
    """
    nb_hidden   = settings["nb_hidden"]
    nb_outputs  = settings["nb_outputs"]
    nb_steps    = settings["nb_steps"]
    psp_norm = bool(settings.get("psp_norm_peak", False))
    batch_size  = inputs.size(0)
    base_interval = max(1, int(mlp_interval))
    param_ranges = _resolve_param_ranges(settings)
    mlp_mode = _select_mlp_mode(settings)
    hidden_group_layout = _get_group_layout(settings, 0, nb_hidden)
    output_group_layout = _get_group_layout(settings, 1, nb_outputs)
    weight_mod_active = _weight_mod_enabled(settings)
    weight_layouts = {}
    if weight_mod_active:
        for name in WEIGHT_PARAM_NAMES:
            layout = _get_weight_layout(settings, name)
            if layout is not None:
                weight_layouts[name] = layout
    substitution_mode = _is_substitution_mode(mlp_mode)
    hidden_ann_layout = hidden_group_layout
    output_ann_layout = output_group_layout
    if substitution_mode:
        hidden_ann_layout = _uniform_group_layout(hidden_group_layout)
        output_ann_layout = _uniform_group_layout(output_group_layout)
    mod_mask = _build_fixed_mod_mask(settings, hidden_group_layout, output_group_layout)
    mod_mask_enabled = mod_mask is not None
    mod_hid_flat_group = bool(settings.get("mod_hid_flat_group", False))
    mod_hid_flat_mod_only = bool(settings.get("mod_hid_flat_modulated_only", False))
    param_names = MOD_TARGET_PARAM_NAMES
    timescale_ctrl = getattr(modulator, "param_timescales", None)
    intervals = {name: torch.tensor(float(base_interval), device=device, dtype=dtype) for name in param_names}
    if timescale_ctrl is not None:
        intervals.update({k: v.to(device=device, dtype=dtype) for k, v in timescale_ctrl.intervals().items()})
    update_ticks = {name: intervals[name].clone() for name in param_names}

    smoothing_ctrl = getattr(modulator, "param_smoothing", None)
    smoothing_cfg = settings.get("param_smoothing", {})
    smoothing_modes = set(smoothing_cfg.get("modes", []))
    smoothing_allowed = bool(smoothing_ctrl is not None) and bool(smoothing_cfg.get("enable")) and (
        not smoothing_modes or mlp_mode in smoothing_modes
    )
    smoothing_factors = {}
    if smoothing_allowed:
        smoothing_factors = {k: v.to(device=device, dtype=dtype) for k, v in smoothing_ctrl.mixing_factors().items()}
    smoothing_active = bool(smoothing_factors)

    mod_current_enable = bool(settings.get("mod_current_enable", False))
    raw_current_target = settings.get("mod_current_target", "both")
    def _normalize_current_target(val):
        key = str(val or "both").strip().lower()
        if key in {"hid", "hidden", "h"}:
            return "hidden"
        if key in {"out", "output", "o"}:
            return "output"
        if key in {"both", "all", "ho", "hidden_output", "hidden+output"}:
            return "both"
        return "both"
    mod_current_target = _normalize_current_target(raw_current_target)
    use_current_hidden = mod_current_enable and mod_current_target in {"hidden", "both"}
    use_current_output = mod_current_enable and mod_current_target in {"output", "both"}

    def _avg_mix(names: List[str]) -> Optional[torch.Tensor]:
        if not smoothing_active:
            return None
        mixes = [smoothing_factors[name] for name in names if name in smoothing_factors]
        if not mixes:
            return None
        return torch.stack(mixes, dim=0).mean(dim=0)

    hidden_current_mix = _avg_mix(HIDDEN_PARAM_NAMES) if (mod_current_enable and use_current_hidden) else None
    output_current_mix = _avg_mix(OUTPUT_PARAM_NAMES) if (mod_current_enable and use_current_output) else None

    w1 = state["w1"]; w2 = state["w2"]; v1 = state["v1"]
    a1 = state["alpha_hetero_1"]; b1 = state["beta_hetero_1"]
    thr = state["thresholds_1"];   rst = state["reset_1"]; rpo = state["rest_1"]
    a2 = state["alpha_hetero_2"];  b2 = state["beta_hetero_2"]

    alpha_1_local = a1.expand(batch_size, nb_hidden).clone()
    beta_1_local  = b1.expand(batch_size, nb_hidden).clone()
    thr_local     = thr.expand(batch_size, nb_hidden).clone()
    rst_local     = rst.expand(batch_size, nb_hidden).clone()
    rpo_local     = rpo.expand(batch_size, nb_hidden).clone()
    alpha_2_local = a2.expand(batch_size, nb_outputs).clone()
    beta_2_local  = b2.expand(batch_size, nb_outputs).clone()
    base_param_map = {
        "alpha_1": alpha_1_local.clone(),
        "beta_1": beta_1_local.clone(),
        "thr": thr_local.clone(),
        "reset": rst_local.clone(),
        "rest": rpo_local.clone(),
        "alpha_2": alpha_2_local.clone(),
        "beta_2": beta_2_local.clone(),
    }
    if weight_mod_active:
        w1_local = w1.unsqueeze(0).expand(batch_size, -1, -1).clone()
        v1_local = v1.unsqueeze(0).expand(batch_size, -1, -1).clone()
        w2_local = w2.unsqueeze(0).expand(batch_size, -1, -1).clone()
    else:
        w1_local = v1_local = w2_local = None
    alpha_1_target = beta_1_target = thr_target = rst_target = rpo_target = alpha_2_target = beta_2_target = None
    if smoothing_active:
        alpha_1_target = alpha_1_local.clone()
        beta_1_target = beta_1_local.clone()
        thr_target = thr_local.clone()
        rst_target = rst_local.clone()
        rpo_target = rpo_local.clone()
        alpha_2_target = alpha_2_local.clone()
        beta_2_target = beta_2_local.clone()
    decay_dt = _effective_tau_decay_dt(settings)
    gain_hidden = None
    gain_out = None
    if psp_norm:
        gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
        gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)

    hidden_current = None
    output_current = None
    hidden_current_target = None
    output_current_target = None
    if mod_current_enable:
        if use_current_hidden:
            hidden_current = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
            if smoothing_active:
                hidden_current_target = hidden_current.clone()
        if use_current_output:
            output_current = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
            if smoothing_active:
                output_current_target = output_current.clone()

    syn = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    out_h = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    hidden_dropout_p = float(settings.get("hidden_dropout_p", 0.0))

    delay_logits = state.get("input_delay_logits")
    max_delay = max(1, int(settings.get("input_delay_steps", 1))) if settings.get("use_input_delay", False) else 0
    inputs_delayed = _apply_input_delay(
        inputs, delay_logits, max_delay, temperature=settings.get("input_delay_temp", 1.0)
    )
    inputs_mod = inputs_delayed
    compress_mode = settings.get("channel_compress_mode")
    if compress_mode == "mod_only":
        factor = int(settings.get("channel_compress_factor_mod", settings.get("channel_compress_factor", 1)))
        if factor > 1:
            inputs_mod = compress_dense_inputs(inputs_delayed, factor, settings.get("nb_inputs_mod"))
    elif compress_mode == "mod_mlp":
        compressor = getattr(modulator, "input_compressor", None)
        if compressor is None:
            raise RuntimeError("channel_compress_mode=mod_mlp requires an input compressor on the modulator.")
        inputs_mod = compressor(inputs_delayed)
    h1_in = None
    if not weight_mod_active:
        h1_in = torch.einsum("btc,cd->btd", inputs_delayed, w1)  # [B,T,H]

    flt2 = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out2 = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out_rec = []

    mem_rec, spk_rec = [], []
    in_block_mask = mlp_input_block_mask or {}
    out_block_mask = mlp_output_block_mask or {}
    mlp_mode = _select_mlp_mode(settings)
    snn_mode = _is_snn_mode(mlp_mode)
    snn_add_mode = snn_mode and (mlp_mode == "snn_add")
    snn_sub_mode = snn_mode and (mlp_mode == "snn_sub")
    if snn_mode and not isinstance(modulator, (SNNAdditiveModulator, SNNSubstitutionModulator)):
        raise TypeError("snn mode requires an SNNAdditiveModulator or SNNSubstitutionModulator instance.")
    mlp = modulator if not snn_mode else None
    additive_mode = (mlp_mode == "mlp_add")
    combo_mode = (mlp_mode == "mlp_combo")
    combo_additive_list, combo_multiplicative_list = _resolve_ann_combo_lists(
        settings.get("ann_combo_additive"),
        settings.get("ann_combo_multiplicative"),
    )
    combo_additive = set(combo_additive_list)
    combo_multiplicative = set(combo_multiplicative_list)
    combo_activation_kind = (getattr(mlp, "output_activation_kind", "tanh") or "tanh").lower() if mlp is not None else "tanh"
    substitution_snn_values = False
    if mlp is not None and hasattr(mlp, "reset_sequence_state"):
        mlp.reset_sequence_state(batch_size, device=device, dtype=dtype)
    mlp_state_each_step = bool(settings.get("ann_rnn_state_every_step", False) and mlp is not None and getattr(mlp, "stateful", False))
    update_every_step = bool(settings.get("mod_update_every_step", False))

    if snn_mode:
        mod_hidden = modulator.mod_hidden
        init_param_groups: Optional[Dict[str, torch.Tensor]] = None
        if snn_sub_mode and isinstance(modulator, SNNSubstitutionModulator):
            def _group_param(name: str, tensor: torch.Tensor, layout: Optional['GroupLayout']):
                if layout is not None and (substitution_mode or mod_mask_enabled):
                    tensor = layout.project(tensor)
                if mod_mask_enabled and not getattr(modulator, "use_neuromodulators", False):
                    idx = mod_mask["hidden_union_idx"] if name in HIDDEN_PARAM_NAMES else mod_mask["output_union_idx"]
                    if idx is not None and idx.numel() > 0:
                        tensor = tensor.index_select(1, idx.to(device=device))
                return tensor
            init_param_groups = {
                "alpha_1": _group_param("alpha_1", alpha_1_local, hidden_group_layout),
                "beta_1": _group_param("beta_1", beta_1_local, hidden_group_layout),
                "thr": _group_param("thr", thr_local, hidden_group_layout),
                "reset": _group_param("reset", rst_local, hidden_group_layout),
                "rest": _group_param("rest", rpo_local, hidden_group_layout),
                "alpha_2": _group_param("alpha_2", alpha_2_local, output_group_layout),
                "beta_2": _group_param("beta_2", beta_2_local, output_group_layout),
            }
            for name in WEIGHT_PARAM_NAMES:
                layout = weight_layouts.get(name)
                if layout is not None:
                    if name == "w1":
                        matrix = w1_local if weight_mod_active else w1.unsqueeze(0).expand(batch_size, -1, -1)
                    elif name == "v1":
                        matrix = v1_local if weight_mod_active else v1.unsqueeze(0).expand(batch_size, -1, -1)
                    else:
                        matrix = w2_local if weight_mod_active else w2.unsqueeze(0).expand(batch_size, -1, -1)
                    init_param_groups[name] = layout.project_matrix(matrix)
        mod_state = modulator.zero_state(batch_size, init_params=init_param_groups)
        mod_buffer = torch.zeros((base_interval, batch_size, mod_hidden), device=device, dtype=dtype) if snn_add_mode else None
        capture_mod_spikes = bool(training and settings.get("use_snn_mod_reg", False))
        mod_spk_rec = [] if capture_mod_spikes else None
        if capture_mod_spikes:
            setattr(modulator, "_last_spk_rec", None)

    def _apply_modulated_update(name: str, base_value: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        if additive_mode:
            lo, hi = param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
            lo_t = base_value.new_full(base_value.shape, float(lo))
            hi_t = base_value.new_full(base_value.shape, float(hi))
            headroom = torch.where(raw >= 0, hi_t - base_value, base_value - lo_t)
            delta = raw * headroom  # k=1 headroom scaling
            return _apply_additive_delta(base_value, delta, name, param_ranges)
        if substitution_mode:
            # Substitution expects values mapped through an activation into a known range.
            # In NM mode, let the NM mapper decide whether outputs are already squashed (e.g. sigmoid/tanh) or are logits.
            if getattr(mlp, "use_neuromodulators", False) and getattr(mlp, "nm_mapper", None) is not None:
                activation_kind = getattr(mlp.nm_mapper, "activation_kind", None) or "none"
            else:
                activation_kind = "sigmoid"
        else:
            activation_kind = getattr(mlp, "output_activation_kind", "sigmoid") if mlp is not None else "sigmoid"
        return _map_substitution_output(raw, name, activation_kind, param_ranges)

    def _aggregate_currents(effects: Dict[str, torch.Tensor], ready: Optional[Dict[str, torch.Tensor]]):
        if not effects:
            return None, None
        def _collect(names: List[str]) -> Optional[torch.Tensor]:
            parts = []
            for name in names:
                eff = effects.get(name)
                if eff is None:
                    continue
                if ready is not None and name in ready:
                    eff = eff * ready[name]
                parts.append(eff)
            if not parts:
                return None
            total = parts[0]
            for term in parts[1:]:
                total = total + term
            return total / float(len(parts))
        return _collect(HIDDEN_PARAM_NAMES), _collect(OUTPUT_PARAM_NAMES)

    def _update_mod_currents(effects: Dict[str, torch.Tensor], ready: Optional[Dict[str, torch.Tensor]]):
        nonlocal hidden_current, output_current, hidden_current_target, output_current_target
        if not mod_current_enable:
            return
        hid_cur, out_cur = _aggregate_currents(effects, ready)
        if use_current_hidden and hid_cur is not None:
            if smoothing_active and hidden_current_target is not None:
                hidden_current_target = hid_cur
            else:
                hidden_current = hid_cur
        if use_current_output and out_cur is not None:
            if smoothing_active and output_current_target is not None:
                output_current_target = out_cur
            else:
                output_current = out_cur

    def _expand_current_effect(vals: Optional[torch.Tensor], idx: Optional[torch.Tensor], full_dim: int) -> Optional[torch.Tensor]:
        if vals is None:
            return None
        if idx is None or vals.size(1) == full_dim:
            return vals
        out = vals.new_zeros((vals.size(0), full_dim))
        out.index_copy_(1, idx.to(device=vals.device), vals)
        return out

    def _normalize_combo_multiplier(raw: torch.Tensor) -> torch.Tensor:
        if combo_activation_kind == "sigmoid":
            raw = raw * 2.0 - 1.0
        return torch.clamp(raw, -1.0, 1.0)

    def _combo_target(name: str, base_value: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        if name in combo_additive:
            lo, hi = param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
            lo_t = base_value.new_full(base_value.shape, float(lo))
            hi_t = base_value.new_full(base_value.shape, float(hi))
            headroom = torch.where(raw >= 0, hi_t - base_value, base_value - lo_t)
            delta = raw * headroom
            return _apply_additive_delta(base_value, delta, name, param_ranges)
        lo, hi = param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
        raw_norm = _normalize_combo_multiplier(raw)
        scale = torch.exp(raw_norm * COMBO_MULT_LOG_SPAN)
        proposed = base_value * scale
        return torch.clamp(proposed, min=float(lo), max=float(hi))

    def _apply_combo_update(
        name: str,
        current: torch.Tensor,
        raw: torch.Tensor,
        ready: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_value = base_param_map.get(name, current)
        proposed = _combo_target(name, base_value, raw)
        if target is not None:
            return proposed * ready + target * (1.0 - ready)
        return proposed * ready + current * (1.0 - ready)

    def _advance_param_timers() -> Dict[str, torch.Tensor]:
        ready = {}
        base_dt = torch.tensor(float(base_interval), device=device, dtype=dtype)
        for pname in param_names:
            update_ticks[pname] = update_ticks[pname] - base_dt
            mask = torch.sigmoid(-update_ticks[pname])
            # When mask ~1, add the interval back in; when ~0, just keep subtracting base_dt.
            update_ticks[pname] = update_ticks[pname] + intervals[pname] * mask
            ready[pname] = mask
        return ready

    def _smooth_param(current: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
        mix = smoothing_factors.get(name)
        if mix is None:
            return target
        return current + mix * (target - current)

    def _prepare_grouped_input(tensor: torch.Tensor, layout: Optional[GroupLayout]) -> torch.Tensor:
        if layout is None:
            return tensor
        if substitution_mode or mod_mask_enabled:
            return layout.project(tensor)
        return tensor

    def _scale_headroom_effects_additive(effects: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not effects:
            return effects
        scaled = {}
        cur_map = {
            "alpha_1": alpha_1_local,
            "beta_1": beta_1_local,
            "thr": thr_local,
            "reset": rst_local,
            "rest": rpo_local,
            "alpha_2": alpha_2_local,
            "beta_2": beta_2_local,
        }
        for name, eff in effects.items():
            cur = cur_map.get(name)
            if cur is None:
                scaled[name] = eff
                continue
            lo, hi = param_ranges.get(name, PARAMETER_RANGE_DEFAULTS.get(name, (-float("inf"), float("inf"))))
            lo_t = cur.new_full(cur.shape, float(lo))
            hi_t = cur.new_full(cur.shape, float(hi))
            headroom = torch.where(eff >= 0, hi_t - cur, cur - lo_t)
            scaled[name] = eff * headroom
        return scaled

    def _direct_substitute(name: str, current: torch.Tensor, proposed: torch.Tensor, ready: torch.Tensor) -> torch.Tensor:
        # Proposed values are already soft-bounded (e.g., tanh-mapped) by the modulator;
        # avoid an extra hard clamp here to preserve gradients and prevent "double clamp".
        return proposed * ready + current * (1.0 - ready)

    nm_cfg = settings.get("nm_cfg", {}) or {}
    def _get_pair(val, default):
        if isinstance(val, (list, tuple)):
            if len(val) >= 2:
                return (float(val[0]), float(val[1]))
            if len(val) == 1:
                return (float(val[0]), float(val[0]))
        try:
            v = float(val)
            return (v, v)
        except Exception:
            return default
    nm_neuron_frac = _get_pair(nm_cfg.get("neuron_fraction", (1.0, 1.0)), (1.0, 1.0))
    nm_param_frac_map = _parse_param_fraction_map(nm_cfg.get("param_fraction", None), default=1.0)
    nm_neuron_frac_enable = bool(nm_cfg.get("neuron_fraction_enable", False))
    nm_param_frac_enable = bool(nm_cfg.get("param_fraction_enable", False))
    hidden_target = hidden_ann_layout.group_count if hidden_ann_layout else nb_hidden
    output_target = output_ann_layout.group_count if output_ann_layout else nb_outputs
    mod_mask_tensors = {}
    if mod_mask_enabled:
        hidden_target = int(mod_mask["hidden_target"])
        output_target = int(mod_mask["output_target"])
        mod_mask_tensors["hidden_neuron_idx"] = mod_mask["hidden_neuron_idx"].to(device=device)
        mod_mask_tensors["output_neuron_idx"] = mod_mask["output_neuron_idx"].to(device=device)
        mod_mask_tensors["hidden_union_idx"] = mod_mask["hidden_union_idx"].to(device=device)
        mod_mask_tensors["output_union_idx"] = mod_mask["output_union_idx"].to(device=device)
        mod_mask_tensors["hidden_param_idx"] = {
            name: idx.to(device=device) for name, idx in mod_mask.get("hidden_param_idx", {}).items()
        }
        mod_mask_tensors["output_param_idx"] = {
            name: idx.to(device=device) for name, idx in mod_mask.get("output_param_idx", {}).items()
        }
        mod_mask_tensors["param_masks_target"] = {
            name: mask.to(device=device, dtype=dtype) for name, mask in mod_mask.get("param_masks_target", {}).items()
        }
        mod_mask_tensors["param_masks_union"] = {
            name: mask.to(device=device, dtype=dtype) for name, mask in mod_mask.get("param_masks_union", {}).items()
        }
        mod_mask_tensors["flat_hidden_idx"] = None
        mod_mask_tensors["flat_output_idx"] = None
        if mod_mask.get("flat_inputs"):
            if mod_mask.get("flat_hidden_idx") is not None:
                mod_mask_tensors["flat_hidden_idx"] = mod_mask["flat_hidden_idx"].to(device=device)
            if mod_mask.get("flat_output_idx") is not None:
                mod_mask_tensors["flat_output_idx"] = mod_mask["flat_output_idx"].to(device=device)

    def _apply_nm_masks(effects: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if mod_mask_enabled or not (nm_neuron_frac_enable or nm_param_frac_enable):
            return effects
        masked = {k: v for k, v in effects.items()}
        if nm_neuron_frac_enable:
            if "alpha_1" in masked:
                mask_h = (torch.rand((batch_size, masked["alpha_1"].size(1)), device=device, dtype=dtype) < float(nm_neuron_frac[0])).float()
                for key in ("alpha_1", "beta_1", "thr", "reset", "rest"):
                    if key in masked:
                        masked[key] = masked[key] * mask_h
            if "alpha_2" in masked:
                mask_o = (torch.rand((batch_size, masked["alpha_2"].size(1)), device=device, dtype=dtype) < float(nm_neuron_frac[1])).float()
                for key in ("alpha_2", "beta_2"):
                    if key in masked:
                        masked[key] = masked[key] * mask_o
        if nm_param_frac_enable:
            for key in ("alpha_1", "beta_1", "thr", "reset", "rest", "alpha_2", "beta_2"):
                if key in masked:
                    frac = float(nm_param_frac_map.get(key, 1.0))
                    masked[key] = masked[key] * (torch.rand_like(masked[key]) < frac).float()
        return masked

    def _expand_with_indices(values: torch.Tensor, indices: Optional[torch.Tensor], target_count: int) -> torch.Tensor:
        if indices is None:
            return values
        out = values.new_zeros((values.size(0), int(target_count)))
        if indices.numel() > 0:
            out[:, indices] = values
        return out

    def _expand_param_effect(name: str, values: torch.Tensor, layout: Optional[GroupLayout]) -> torch.Tensor:
        if not mod_mask_enabled:
            return layout.expand(values) if layout is not None else values
        target_count = hidden_target if name in HIDDEN_PARAM_NAMES else output_target
        idx_map = mod_mask_tensors.get("hidden_param_idx") if name in HIDDEN_PARAM_NAMES else mod_mask_tensors.get("output_param_idx")
        idx = idx_map.get(name) if idx_map else None
        expanded = _expand_with_indices(values, idx, target_count)
        mask = mod_mask_tensors.get("param_masks_target", {}).get(name)
        if mask is not None:
            expanded = expanded * mask
        if layout is not None:
            expanded = layout.expand(expanded)
        return expanded

    def _expand_nm_effect(name: str, values: torch.Tensor, layout: Optional[GroupLayout]) -> torch.Tensor:
        if not mod_mask_enabled:
            return layout.expand(values) if layout is not None else values
        target_count = hidden_target if name in HIDDEN_PARAM_NAMES else output_target
        idx = mod_mask_tensors.get("hidden_neuron_idx") if name in HIDDEN_PARAM_NAMES else mod_mask_tensors.get("output_neuron_idx")
        expanded = _expand_with_indices(values, idx, target_count)
        mask = mod_mask_tensors.get("param_masks_target", {}).get(name)
        if mask is not None:
            expanded = expanded * mask
        if layout is not None:
            expanded = layout.expand(expanded)
        return expanded

    def _expand_union_effect(name: str, values: torch.Tensor, layout: Optional[GroupLayout]) -> torch.Tensor:
        if not mod_mask_enabled:
            return layout.expand(values) if layout is not None else values
        target_count = hidden_target if name in HIDDEN_PARAM_NAMES else output_target
        idx = mod_mask_tensors.get("hidden_union_idx") if name in HIDDEN_PARAM_NAMES else mod_mask_tensors.get("output_union_idx")
        mask = mod_mask_tensors.get("param_masks_union", {}).get(name)
        if mask is not None:
            values = values * mask
        expanded = _expand_with_indices(values, idx, target_count)
        if layout is not None:
            expanded = layout.expand(expanded)
        return expanded

    def _select_param_input(name: str, tensor: torch.Tensor, layout: Optional[GroupLayout]) -> torch.Tensor:
        src = _prepare_grouped_input(tensor, layout)
        if not mod_mask_enabled:
            return src
        idx_map = mod_mask_tensors.get("hidden_param_idx") if name in HIDDEN_PARAM_NAMES else mod_mask_tensors.get("output_param_idx")
        idx = idx_map.get(name) if idx_map else None
        if idx is None:
            return src
        return src.index_select(1, idx)

    def _select_union_input(name: str, tensor: torch.Tensor, layout: Optional[GroupLayout]) -> torch.Tensor:
        src = _prepare_grouped_input(tensor, layout)
        if not mod_mask_enabled:
            return src
        idx = mod_mask_tensors.get("hidden_union_idx") if name in HIDDEN_PARAM_NAMES else mod_mask_tensors.get("output_union_idx")
        if idx is None:
            return src
        return src.index_select(1, idx)

    def _select_flat_input(tensor: torch.Tensor, idx: Optional[torch.Tensor]) -> torch.Tensor:
        if idx is None:
            return tensor
        return tensor.index_select(1, idx)

    def _expand_group_idx_to_neurons(
        layout: Optional['GroupLayout'],
        group_idx: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if layout is None or getattr(layout, "forward", None) is None or group_idx is None:
            return group_idx
        if group_idx.numel() == 0:
            return group_idx
        weights = layout.forward.to(device=group_idx.device, dtype=dtype)
        try:
            selected = weights.index_select(0, group_idx)
        except Exception:
            return group_idx
        mask = selected.sum(dim=0) > 0
        return mask.nonzero(as_tuple=False).flatten().to(dtype=torch.long, device=group_idx.device)

    def _weight_current_map() -> Dict[str, torch.Tensor]:
        if weight_mod_active:
            return {"w1": w1_local, "v1": v1_local, "w2": w2_local}
        return {
            "w1": w1.unsqueeze(0).expand(batch_size, -1, -1),
            "v1": v1.unsqueeze(0).expand(batch_size, -1, -1),
            "w2": w2.unsqueeze(0).expand(batch_size, -1, -1),
        }

    def _select_weight_input(name: str) -> Optional[torch.Tensor]:
        layout = weight_layouts.get(name)
        if layout is None:
            return None
        matrix = _weight_current_map()[name]
        projected = layout.project_matrix(matrix)
        if substitution_mode:
            scale = _weight_reference_scale(name, matrix).view(matrix.size(0), 1)
            norm = torch.clamp(projected / scale, min=-1.0 + 1e-6, max=1.0 - 1e-6)
            if not snn_mode:
                if getattr(mlp, "use_neuromodulators", False):
                    return torch.atanh(norm)
                p = torch.clamp((norm + 1.0) * 0.5, min=1e-6, max=1.0 - 1e-6)
                return torch.log(p / (1.0 - p))
            return norm
        return projected

    def _expand_weight_effect(name: str, values: torch.Tensor) -> Optional[torch.Tensor]:
        layout = weight_layouts.get(name)
        if layout is None:
            return None
        return layout.expand_effect(values)

    def _weight_reference_scale(name: str, current: torch.Tensor) -> torch.Tensor:
        cfg = _weight_mod_cfg(settings)
        std_scale = float(cfg.get("range_std_scale", 3.0))
        max_abs = current.detach().abs().amax(dim=(1, 2), keepdim=True)
        std = current.detach().std(dim=(1, 2), keepdim=True, unbiased=False)
        scale = torch.maximum(max_abs, std * std_scale)
        return scale.clamp_min(float(cfg.get("min_scale", 1e-6)))

    def _weight_update_mode() -> str:
        mode = str(_weight_mod_cfg(settings).get("mode", "auto") or "auto").lower()
        if mode in {"sub", "substitution", "replace"}:
            return "substitution"
        if mode in {"mul", "mult", "multiplicative"}:
            return "multiplicative"
        if mode in {"add", "additive"}:
            return "additive"
        if substitution_mode:
            return "substitution"
        return "additive"

    def _apply_weight_update(name: str, current: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        update_mode = _weight_update_mode()
        cfg = _weight_mod_cfg(settings)
        if update_mode == "substitution":
            activation_kind = getattr(mlp, "output_activation_kind", "sigmoid") if mlp is not None else "tanh"
            scale = _weight_reference_scale(name, current)
            if activation_kind == "sigmoid":
                raw_norm = raw * 2.0 - 1.0
            elif activation_kind == "tanh":
                raw_norm = raw
            else:
                raw_norm = torch.tanh(raw)
            return torch.clamp(raw_norm, -1.0, 1.0) * scale
        if update_mode == "multiplicative":
            raw_norm = torch.clamp(raw, -1.0, 1.0)
            scale = torch.exp(raw_norm * COMBO_MULT_LOG_SPAN)
            return current * scale
        delta_scale = float(cfg.get("delta_scale", 0.05))
        scale = _weight_reference_scale(name, current)
        return current + raw * scale * delta_scale

    def _apply_weight_effects(effects: Dict[str, torch.Tensor]):
        nonlocal w1_local, v1_local, w2_local
        if not weight_mod_active:
            return
        for name in WEIGHT_PARAM_NAMES:
            if name not in effects:
                continue
            expanded = _expand_weight_effect(name, effects[name])
            if expanded is None:
                continue
            if name == "w1":
                w1_local = _apply_weight_update(name, w1_local, expanded)
            elif name == "v1":
                v1_local = _apply_weight_update(name, v1_local, expanded)
            elif name == "w2":
                w2_local = _apply_weight_update(name, w2_local, expanded)

    def _compute_mlp_output(t_step: int) -> torch.Tensor:
        if mlp is None:
            raise RuntimeError("MLP modulator not initialized.")
        if t_step >= base_interval:
            in_flat  = inputs_mod[:, t_step - base_interval + 1:t_step + 1, :].sum(dim=1)
            hid_flat = torch.stack(spk_rec[-base_interval:], dim=1).sum(dim=1)
            out_flat = torch.stack(out_rec[-base_interval:], dim=1).sum(dim=1)
        else:
            in_flat  = inputs_mod[:, :t_step + 1, :].sum(dim=1)
            hid_flat = torch.stack(spk_rec[:t_step + 1], dim=1).sum(dim=1)
            out_flat = torch.stack(out_rec[:t_step + 1], dim=1).sum(dim=1)

        parts = []
        slc = mlp.in_slices
        zero = lambda ref: torch.zeros_like(ref)
        if "alpha_1" in slc:
            src = _select_param_input("alpha_1", alpha_1_local, hidden_ann_layout)
            parts.append(src if in_block_mask.get("alpha_1", True) else zero(src))
        if "beta_1" in slc:
            src = _select_param_input("beta_1", beta_1_local, hidden_ann_layout)
            parts.append(src if in_block_mask.get("beta_1", True) else zero(src))
        if "thr" in slc:
            src = _select_param_input("thr", thr_local, hidden_ann_layout)
            parts.append(src if in_block_mask.get("thr", True) else zero(src))
        if "reset" in slc:
            src = _select_param_input("reset", rst_local, hidden_ann_layout)
            parts.append(src if in_block_mask.get("reset", True) else zero(src))
        if "rest" in slc:
            src = _select_param_input("rest", rpo_local, hidden_ann_layout)
            parts.append(src if in_block_mask.get("rest", True) else zero(src))
        if "alpha_2" in slc:
            src = _select_param_input("alpha_2", alpha_2_local, output_ann_layout)
            parts.append(src if in_block_mask.get("alpha_2", True) else zero(src))
        if "beta_2" in slc:
            src = _select_param_input("beta_2", beta_2_local, output_ann_layout)
            parts.append(src if in_block_mask.get("beta_2", True) else zero(src))
        for name in WEIGHT_PARAM_NAMES:
            if name in slc:
                src = _select_weight_input(name)
                if src is None:
                    src = inputs_mod.new_zeros((batch_size, slc[name].stop - slc[name].start))
                parts.append(src if in_block_mask.get(name, True) else zero(src))
        if "in_flat" in slc:
            parts.append(in_flat if in_block_mask.get("in_flat", True) else zero(in_flat))
        if "hid_flat" in slc:
            src = hid_flat
            if mod_hid_flat_group and hidden_ann_layout is not None:
                src = hidden_ann_layout.project(src)
            if mod_hid_flat_mod_only and mod_mask_enabled:
                if mod_hid_flat_group:
                    idx = mod_mask_tensors.get("hidden_union_idx")
                else:
                    idx = _expand_group_idx_to_neurons(hidden_group_layout, mod_mask_tensors.get("hidden_union_idx"))
                src = _select_flat_input(src, idx)
            elif not (mod_hid_flat_group or mod_hid_flat_mod_only):
                src = _select_flat_input(src, mod_mask_tensors.get("flat_hidden_idx"))
            parts.append(src if in_block_mask.get("hid_flat", True) else zero(src))
        if "out_flat" in slc:
            src = _select_flat_input(out_flat, mod_mask_tensors.get("flat_output_idx"))
            parts.append(src if in_block_mask.get("out_flat", True) else zero(src))

        mlp_input = torch.cat(parts, dim=1)
        return mlp(mlp_input)

    nm_debug_print = bool(settings.get("nm_debug_print", False))
    nm_debug_done = False

    for t in range(nb_steps):
        if weight_mod_active:
            h1_t = torch.einsum("bc,bch->bh", inputs_delayed[:, t, :], w1_local)
            h1_t = h1_t + torch.einsum("bd,bdh->bh", out_h, v1_local)
        else:
            h1_t = h1_in[:, t] + torch.einsum("bd,dc->bc", out_h, v1)  # [B,H]
        if mod_current_enable and use_current_hidden and hidden_current is not None:
            h1_t = h1_t + hidden_current

        mthr = mem - thr_local
        out_h = spike_fn(mthr)
        if hidden_dropout_p > 0.0 and training:
            out_h = F.dropout(out_h, p=hidden_dropout_p, training=True)
        rst_mask = (mthr > 0).float()

        syn = alpha_1_local * syn + h1_t
        syn_term = (1 - beta_1_local) * syn
        if gain_hidden is not None:
            syn_term = gain_hidden * syn_term
        mem = beta_1_local * (mem - rpo_local) + rpo_local + syn_term - rst_mask * (thr_local - rst_local)

        mem_rec.append(mem); spk_rec.append(out_h)

        if weight_mod_active:
            h2_t = torch.einsum("bd,bdo->bo", out_h, w2_local)
        else:
            h2_t = torch.einsum("bd,do->bo", out_h, w2)                # [B,O]
        if mod_current_enable and use_current_output and output_current is not None:
            h2_t = h2_t + output_current

        flt2 = alpha_2_local * flt2 + h2_t
        flt_term = (1 - beta_2_local) * flt2
        if gain_out is not None:
            flt_term = gain_out * flt_term
        out2 = beta_2_local * out2 + flt_term
        out_rec.append(out2)

        mlp_out_cached = None
        if mlp_state_each_step:
            mlp_out_cached = _compute_mlp_output(t)

        if snn_mode:
            alpha_1_in = alpha_1_local
            beta_1_in = beta_1_local
            thr_in = thr_local
            rst_in = rst_local
            rpo_in = rpo_local
            alpha_2_in = alpha_2_local
            beta_2_in = beta_2_local
            hid_spk_in = out_h
            out_spk_in = out2
            if mod_mask_enabled:
                alpha_1_in = _select_union_input("alpha_1", alpha_1_in, hidden_group_layout)
                beta_1_in = _select_union_input("beta_1", beta_1_in, hidden_group_layout)
                thr_in = _select_union_input("thr", thr_in, hidden_group_layout)
                rst_in = _select_union_input("reset", rst_in, hidden_group_layout)
                rpo_in = _select_union_input("rest", rpo_in, hidden_group_layout)
                alpha_2_in = _select_union_input("alpha_2", alpha_2_in, output_group_layout)
                beta_2_in = _select_union_input("beta_2", beta_2_in, output_group_layout)
                hid_spk_in = _select_flat_input(hid_spk_in, mod_mask_tensors.get("flat_hidden_idx"))
                out_spk_in = _select_flat_input(out_spk_in, mod_mask_tensors.get("flat_output_idx"))
            weight_features = []
            if weight_mod_active:
                for name in WEIGHT_PARAM_NAMES:
                    src = _select_weight_input(name)
                    if src is not None:
                        weight_features.append(src)
            features = modulator.build_features(
                alpha_1_in, beta_1_in, thr_in, rst_in, rpo_in,
                alpha_2_in, beta_2_in,
                inputs_mod[:, t, :], hid_spk_in, out_spk_in, t,
                weight_features=weight_features,
            )
            mod_spk, mod_state = modulator.forward_step(features, mod_state)
            if mod_buffer is not None:
                mod_buffer[t % base_interval] = mod_spk
            if mod_spk_rec is not None:
                mod_spk_rec.append(mod_spk)

        if snn_mode:
            do_update = update_every_step or ((t + 1) % base_interval == 0)
        else:
            do_update = update_every_step or (t % base_interval == 0)
        if do_update:
            if snn_mode:
                # Notebook behavior: update every interval with full-strength masks.
                ready_masks = {name: torch.ones_like(update_ticks[name]) for name in param_names}
            else:
                ready_masks = _advance_param_timers()

            if snn_mode:
                if update_every_step or (t + 1 >= base_interval):
                    if snn_add_mode and mod_buffer is not None:
                        span = max(1, int(min(t + 1, base_interval)))
                        recent_spikes = mod_buffer.sum(dim=0) / float(span)
                        effects = modulator.modulation_effects(recent_spikes)
                        expanded = {}
                        for key, vals in effects.items():
                            if key in WEIGHT_PARAM_NAMES:
                                expanded[key] = vals
                                continue
                            layout = hidden_group_layout if key in HIDDEN_PARAM_NAMES else output_group_layout if key in OUTPUT_PARAM_NAMES else None
                            if modulator.use_neuromodulators:
                                expanded[key] = _expand_nm_effect(key, vals, layout)
                            else:
                                expanded[key] = _expand_union_effect(key, vals, layout)
                        effects = expanded
                        if modulator.use_neuromodulators:
                            effects = _scale_headroom_effects_additive(effects)
                        effects = _apply_nm_masks(effects)
                        _apply_weight_effects(effects)
                        if mod_current_enable:
                            _update_mod_currents(effects, ready_masks)
                        else:
                            if "alpha_1" in ready_masks:
                                if smoothing_active:
                                    alpha_1_target = _apply_additive_delta(alpha_1_target, effects["alpha_1"] * ready_masks["alpha_1"], "alpha_1", param_ranges)
                                else:
                                    alpha_1_local = _apply_additive_delta(alpha_1_local, effects["alpha_1"] * ready_masks["alpha_1"], "alpha_1", param_ranges)
                                    if gain_hidden is not None:
                                        gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                            if "beta_1" in ready_masks:
                                if smoothing_active:
                                    beta_1_target = _apply_additive_delta(beta_1_target, effects["beta_1"] * ready_masks["beta_1"], "beta_1", param_ranges)
                                else:
                                    beta_1_local = _apply_additive_delta(beta_1_local, effects["beta_1"] * ready_masks["beta_1"], "beta_1", param_ranges)
                                    if gain_hidden is not None:
                                        gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                            if "alpha_2" in ready_masks:
                                if smoothing_active:
                                    alpha_2_target = _apply_additive_delta(alpha_2_target, effects["alpha_2"] * ready_masks["alpha_2"], "alpha_2", param_ranges)
                                else:
                                    alpha_2_local = _apply_additive_delta(alpha_2_local, effects["alpha_2"] * ready_masks["alpha_2"], "alpha_2", param_ranges)
                                    if gain_out is not None:
                                        gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
                            if "beta_2" in ready_masks:
                                if smoothing_active:
                                    beta_2_target = _apply_additive_delta(beta_2_target, effects["beta_2"] * ready_masks["beta_2"], "beta_2", param_ranges)
                                else:
                                    beta_2_local = _apply_additive_delta(beta_2_local, effects["beta_2"] * ready_masks["beta_2"], "beta_2", param_ranges)
                                    if gain_out is not None:
                                        gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
                            if "thr" in ready_masks:
                                if smoothing_active:
                                    thr_target = _apply_additive_delta(thr_target, effects["thr"] * ready_masks["thr"], "thr", param_ranges)
                                else:
                                    thr_local = _apply_additive_delta(thr_local, effects["thr"] * ready_masks["thr"], "thr", param_ranges)
                            if "reset" in ready_masks:
                                if smoothing_active:
                                    rst_target = _apply_additive_delta(rst_target, effects["reset"] * ready_masks["reset"], "reset", param_ranges)
                                else:
                                    rst_local = _apply_additive_delta(rst_local, effects["reset"] * ready_masks["reset"], "reset", param_ranges)
                            if "rest" in ready_masks:
                                if smoothing_active:
                                    rpo_target = _apply_additive_delta(rpo_target, effects["rest"] * ready_masks["rest"], "rest", param_ranges)
                                else:
                                    rpo_local = _apply_additive_delta(rpo_local, effects["rest"] * ready_masks["rest"], "rest", param_ranges)
                    elif snn_sub_mode:
                        out_mem = mod_state[-1].get("mem")
                        if out_mem is None:
                            continue
                        effects = modulator.modulation_effects(out_mem)
                        expanded = {}
                        for key, vals in effects.items():
                            if key in WEIGHT_PARAM_NAMES:
                                expanded[key] = vals
                                continue
                            layout = hidden_group_layout if key in HIDDEN_PARAM_NAMES else output_group_layout if key in OUTPUT_PARAM_NAMES else None
                            if modulator.use_neuromodulators:
                                expanded[key] = _expand_nm_effect(key, vals, layout)
                            else:
                                expanded[key] = _expand_union_effect(key, vals, layout)
                        effects = _apply_nm_masks(expanded)
                        _apply_weight_effects(effects)
                        if mod_current_enable:
                            _update_mod_currents(effects, ready_masks)
                        else:
                            direct_sub = not getattr(modulator, "use_neuromodulators", False)
                            if "alpha_1" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        alpha_1_target = _direct_substitute("alpha_1", alpha_1_target, effects["alpha_1"], ready_masks["alpha_1"])
                                    else:
                                        alpha_1_target = _apply_modulated_update("alpha_1", alpha_1_target, effects["alpha_1"] * ready_masks["alpha_1"])
                                else:
                                    if direct_sub:
                                        alpha_1_local = _direct_substitute("alpha_1", alpha_1_local, effects["alpha_1"], ready_masks["alpha_1"])
                                    else:
                                        alpha_1_local = _apply_modulated_update("alpha_1", alpha_1_local, effects["alpha_1"] * ready_masks["alpha_1"])
                                    if gain_hidden is not None:
                                        gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                            if "beta_1" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        beta_1_target = _direct_substitute("beta_1", beta_1_target, effects["beta_1"], ready_masks["beta_1"])
                                    else:
                                        beta_1_target = _apply_modulated_update("beta_1", beta_1_target, effects["beta_1"] * ready_masks["beta_1"])
                                else:
                                    if direct_sub:
                                        beta_1_local = _direct_substitute("beta_1", beta_1_local, effects["beta_1"], ready_masks["beta_1"])
                                    else:
                                        beta_1_local = _apply_modulated_update("beta_1", beta_1_local, effects["beta_1"] * ready_masks["beta_1"])
                                    if gain_hidden is not None:
                                        gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                            if "alpha_2" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        alpha_2_target = _direct_substitute("alpha_2", alpha_2_target, effects["alpha_2"], ready_masks["alpha_2"])
                                    else:
                                        alpha_2_target = _apply_modulated_update("alpha_2", alpha_2_target, effects["alpha_2"] * ready_masks["alpha_2"])
                                else:
                                    if direct_sub:
                                        alpha_2_local = _direct_substitute("alpha_2", alpha_2_local, effects["alpha_2"], ready_masks["alpha_2"])
                                    else:
                                        alpha_2_local = _apply_modulated_update("alpha_2", alpha_2_local, effects["alpha_2"] * ready_masks["alpha_2"])
                                    if gain_out is not None:
                                        gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
                            if "beta_2" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        beta_2_target = _direct_substitute("beta_2", beta_2_target, effects["beta_2"], ready_masks["beta_2"])
                                    else:
                                        beta_2_target = _apply_modulated_update("beta_2", beta_2_target, effects["beta_2"] * ready_masks["beta_2"])
                                else:
                                    if direct_sub:
                                        beta_2_local = _direct_substitute("beta_2", beta_2_local, effects["beta_2"], ready_masks["beta_2"])
                                    else:
                                        beta_2_local = _apply_modulated_update("beta_2", beta_2_local, effects["beta_2"] * ready_masks["beta_2"])
                                    if gain_out is not None:
                                        gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
                            if "thr" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        thr_target = _direct_substitute("thr", thr_target, effects["thr"], ready_masks["thr"])
                                    else:
                                        thr_target = _apply_modulated_update("thr", thr_target, effects["thr"] * ready_masks["thr"])
                                else:
                                    if direct_sub:
                                        thr_local = _direct_substitute("thr", thr_local, effects["thr"], ready_masks["thr"])
                                    else:
                                        thr_local = _apply_modulated_update("thr", thr_local, effects["thr"] * ready_masks["thr"])
                            if "reset" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        rst_target = _direct_substitute("reset", rst_target, effects["reset"], ready_masks["reset"])
                                    else:
                                        rst_target = _apply_modulated_update("reset", rst_target, effects["reset"] * ready_masks["reset"])
                                else:
                                    if direct_sub:
                                        rst_local = _direct_substitute("reset", rst_local, effects["reset"], ready_masks["reset"])
                                    else:
                                        rst_local = _apply_modulated_update("reset", rst_local, effects["reset"] * ready_masks["reset"])
                            if "rest" in ready_masks:
                                if smoothing_active:
                                    if direct_sub:
                                        rpo_target = _direct_substitute("rest", rpo_target, effects["rest"], ready_masks["rest"])
                                    else:
                                        rpo_target = _apply_modulated_update("rest", rpo_target, effects["rest"] * ready_masks["rest"])
                                else:
                                    if direct_sub:
                                        rpo_local = _direct_substitute("rest", rpo_local, effects["rest"], ready_masks["rest"])
                                    else:
                                        rpo_local = _apply_modulated_update("rest", rpo_local, effects["rest"] * ready_masks["rest"])
                            if (not smoothing_active) and psp_norm:
                                gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                                gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
        if do_update and (not snn_mode):
            # ANN modes: fall through to MLP application below.
            mlp_out = mlp_out_cached if mlp_out_cached is not None else _compute_mlp_output(t)
            if mod_current_enable and getattr(mlp, "current_mode", False):
                slc_out = mlp.current_out_slices or mlp.out_slices
                hid_cur = out_cur = None
                if "hidden_current" in slc_out:
                    hid_cur = mlp_out[:, slc_out["hidden_current"]]
                if "output_current" in slc_out:
                    out_cur = mlp_out[:, slc_out["output_current"]]
                if mod_mask_enabled:
                    hid_cur = _expand_current_effect(hid_cur, mod_mask_tensors.get("hidden_union_idx"), nb_hidden)
                    out_cur = _expand_current_effect(out_cur, mod_mask_tensors.get("output_union_idx"), nb_outputs)
                if smoothing_active:
                    if use_current_hidden and hid_cur is not None:
                        hidden_current_target = hid_cur
                    if use_current_output and out_cur is not None:
                        output_current_target = out_cur
                else:
                    if use_current_hidden and hid_cur is not None:
                        hidden_current = hid_cur
                    if use_current_output and out_cur is not None:
                        output_current = out_cur
            elif getattr(mlp, "use_neuromodulators", False) and mlp.nm_mapper is not None:
                nm_dim = int(mlp.nm_mapper.total_output_dim)
                effects = mlp.nm_mapper.effects_from_flat(mlp_out[:, :nm_dim])
                expanded = {}
                for key, vals in effects.items():
                    layout = hidden_ann_layout if key in HIDDEN_PARAM_NAMES else output_ann_layout if key in OUTPUT_PARAM_NAMES else None
                    expanded[key] = _expand_nm_effect(key, vals, layout)
                effects = expanded
                for name, s in getattr(mlp, "weight_out_slices", {}).items():
                    if out_block_mask.get(name, True):
                        effects[name] = mlp_out[:, s]
                # Note: additive-mode headroom scaling is handled inside `_apply_modulated_update`.
                # If we scale here as well, deltas get headroom-scaled twice (and become too small).
                effects = _apply_nm_masks(effects)
                _apply_weight_effects(effects)
                if nm_debug_print and (not nm_debug_done):
                    nm_debug_done = True
                    with torch.no_grad():
                        print(
                            f"[NM debug] t={t} mlp_out={tuple(mlp_out.shape)} "
                            f"flat_order={getattr(mlp.nm_mapper, 'flat_order', None)} mapper={getattr(mlp.nm_mapper, 'mapper_type', None)}"
                        )
                        for name in MOD_TARGET_PARAM_NAMES:
                            eff = effects.get(name)
                            if eff is None:
                                continue
                            v = eff.detach()
                            finite = torch.isfinite(v)
                            v_f = v[finite] if finite.any() else v.reshape(-1)[:0]
                            if v_f.numel() == 0:
                                print(f"[NM debug] {name}: all non-finite")
                                continue
                            print(
                                f"[NM debug] {name}: min={float(v_f.min()):.5g} max={float(v_f.max()):.5g} "
                                f"mean={float(v_f.mean()):.5g} std={float(v_f.std(unbiased=False)):.5g}"
                            )
                if mod_current_enable:
                    _update_mod_currents(effects, ready_masks)
                else:
                    if combo_mode:
                        if "alpha_1" in ready_masks:
                            if smoothing_active:
                                alpha_1_target = _apply_combo_update(
                                    "alpha_1", alpha_1_local, effects["alpha_1"], ready_masks["alpha_1"], target=alpha_1_target
                                )
                            else:
                                alpha_1_local = _apply_combo_update("alpha_1", alpha_1_local, effects["alpha_1"], ready_masks["alpha_1"])
                        if "beta_1" in ready_masks:
                            if smoothing_active:
                                beta_1_target = _apply_combo_update(
                                    "beta_1", beta_1_local, effects["beta_1"], ready_masks["beta_1"], target=beta_1_target
                                )
                            else:
                                beta_1_local = _apply_combo_update("beta_1", beta_1_local, effects["beta_1"], ready_masks["beta_1"])
                        if "thr" in ready_masks:
                            if smoothing_active:
                                thr_target = _apply_combo_update(
                                    "thr", thr_local, effects["thr"], ready_masks["thr"], target=thr_target
                                )
                            else:
                                thr_local = _apply_combo_update("thr", thr_local, effects["thr"], ready_masks["thr"])
                        if "reset" in ready_masks:
                            if smoothing_active:
                                rst_target = _apply_combo_update(
                                    "reset", rst_local, effects["reset"], ready_masks["reset"], target=rst_target
                                )
                            else:
                                rst_local = _apply_combo_update("reset", rst_local, effects["reset"], ready_masks["reset"])
                        if "rest" in ready_masks:
                            if smoothing_active:
                                rpo_target = _apply_combo_update(
                                    "rest", rpo_local, effects["rest"], ready_masks["rest"], target=rpo_target
                                )
                            else:
                                rpo_local = _apply_combo_update("rest", rpo_local, effects["rest"], ready_masks["rest"])
                        if "alpha_2" in ready_masks:
                            if smoothing_active:
                                alpha_2_target = _apply_combo_update(
                                    "alpha_2", alpha_2_local, effects["alpha_2"], ready_masks["alpha_2"], target=alpha_2_target
                                )
                            else:
                                alpha_2_local = _apply_combo_update("alpha_2", alpha_2_local, effects["alpha_2"], ready_masks["alpha_2"])
                        if "beta_2" in ready_masks:
                            if smoothing_active:
                                beta_2_target = _apply_combo_update(
                                    "beta_2", beta_2_local, effects["beta_2"], ready_masks["beta_2"], target=beta_2_target
                                )
                            else:
                                beta_2_local = _apply_combo_update("beta_2", beta_2_local, effects["beta_2"], ready_masks["beta_2"])
                        if (not smoothing_active) and psp_norm:
                            gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                            gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
                    else:
                        if "alpha_1" in ready_masks:
                            if smoothing_active:
                                alpha_1_target = _apply_modulated_update("alpha_1", alpha_1_target, effects["alpha_1"] * ready_masks["alpha_1"])
                            else:
                                alpha_1_local = _apply_modulated_update("alpha_1", alpha_1_local, effects["alpha_1"] * ready_masks["alpha_1"])
                        if "beta_1" in ready_masks:
                            if smoothing_active:
                                beta_1_target = _apply_modulated_update("beta_1", beta_1_target, effects["beta_1"] * ready_masks["beta_1"])
                            else:
                                beta_1_local = _apply_modulated_update("beta_1", beta_1_local, effects["beta_1"] * ready_masks["beta_1"])
                        if "thr" in ready_masks:
                            if smoothing_active:
                                thr_target = _apply_modulated_update("thr", thr_target, effects["thr"] * ready_masks["thr"])
                            else:
                                thr_local = _apply_modulated_update("thr", thr_local, effects["thr"] * ready_masks["thr"])
                        if "reset" in ready_masks:
                            if smoothing_active:
                                rst_target = _apply_modulated_update("reset", rst_target, effects["reset"] * ready_masks["reset"])
                            else:
                                rst_local = _apply_modulated_update("reset", rst_local, effects["reset"] * ready_masks["reset"])
                        if "rest" in ready_masks:
                            if smoothing_active:
                                rpo_target = _apply_modulated_update("rest", rpo_target, effects["rest"] * ready_masks["rest"])
                            else:
                                rpo_local = _apply_modulated_update("rest", rpo_local, effects["rest"] * ready_masks["rest"])
                        if "alpha_2" in ready_masks:
                            if smoothing_active:
                                alpha_2_target = _apply_modulated_update("alpha_2", alpha_2_target, effects["alpha_2"] * ready_masks["alpha_2"])
                            else:
                                alpha_2_local = _apply_modulated_update("alpha_2", alpha_2_local, effects["alpha_2"] * ready_masks["alpha_2"])
                        if "beta_2" in ready_masks:
                            if smoothing_active:
                                beta_2_target = _apply_modulated_update("beta_2", beta_2_target, effects["beta_2"] * ready_masks["beta_2"])
                            else:
                                beta_2_local = _apply_modulated_update("beta_2", beta_2_local, effects["beta_2"] * ready_masks["beta_2"])
                        if (not smoothing_active) and psp_norm:
                            gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                            gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
            else:
                slc_out = mlp.out_slices
                effects = {}
                if "alpha_1" in slc_out and out_block_mask.get("alpha_1", True):
                    s = slc_out["alpha_1"]; vals = mlp_out[:, s]
                    effects["alpha_1"] = _expand_param_effect("alpha_1", vals, hidden_ann_layout)
                if "beta_1" in slc_out and out_block_mask.get("beta_1", True):
                    s = slc_out["beta_1"]; vals = mlp_out[:, s]
                    effects["beta_1"] = _expand_param_effect("beta_1", vals, hidden_ann_layout)
                if "thr" in slc_out and out_block_mask.get("thr", True):
                    s = slc_out["thr"]; vals = mlp_out[:, s]
                    effects["thr"] = _expand_param_effect("thr", vals, hidden_ann_layout)
                if "reset" in slc_out and out_block_mask.get("reset", True):
                    s = slc_out["reset"]; vals = mlp_out[:, s]
                    effects["reset"] = _expand_param_effect("reset", vals, hidden_ann_layout)
                if "rest" in slc_out and out_block_mask.get("rest", True):
                    s = slc_out["rest"]; vals = mlp_out[:, s]
                    effects["rest"] = _expand_param_effect("rest", vals, hidden_ann_layout)
                if "alpha_2" in slc_out and out_block_mask.get("alpha_2", True):
                    s = slc_out["alpha_2"]; vals = mlp_out[:, s]
                    effects["alpha_2"] = _expand_param_effect("alpha_2", vals, output_ann_layout)
                if "beta_2" in slc_out and out_block_mask.get("beta_2", True):
                    s = slc_out["beta_2"]; vals = mlp_out[:, s]
                    effects["beta_2"] = _expand_param_effect("beta_2", vals, output_ann_layout)
                for name in WEIGHT_PARAM_NAMES:
                    if name in slc_out and out_block_mask.get(name, True):
                        s = slc_out[name]
                        effects[name] = mlp_out[:, s]
                effects = _apply_nm_masks(effects)
                _apply_weight_effects(effects)
                if mod_current_enable:
                    _update_mod_currents(effects, ready_masks)
                else:
                    if "alpha_1" in effects and "alpha_1" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                alpha_1_target = _apply_combo_update(
                                    "alpha_1", alpha_1_local, effects["alpha_1"], ready_masks["alpha_1"], target=alpha_1_target
                                )
                            else:
                                alpha_1_local = _apply_combo_update("alpha_1", alpha_1_local, effects["alpha_1"], ready_masks["alpha_1"])
                        else:
                            if smoothing_active:
                                alpha_1_target = _apply_modulated_update("alpha_1", alpha_1_target, effects["alpha_1"] * ready_masks["alpha_1"])
                            else:
                                alpha_1_local = _apply_modulated_update("alpha_1", alpha_1_local, effects["alpha_1"] * ready_masks["alpha_1"])
                    if "beta_1" in effects and "beta_1" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                beta_1_target = _apply_combo_update(
                                    "beta_1", beta_1_local, effects["beta_1"], ready_masks["beta_1"], target=beta_1_target
                                )
                            else:
                                beta_1_local = _apply_combo_update("beta_1", beta_1_local, effects["beta_1"], ready_masks["beta_1"])
                        else:
                            if smoothing_active:
                                beta_1_target = _apply_modulated_update("beta_1", beta_1_target, effects["beta_1"] * ready_masks["beta_1"])
                            else:
                                beta_1_local = _apply_modulated_update("beta_1", beta_1_local, effects["beta_1"] * ready_masks["beta_1"])
                    if "thr" in effects and "thr" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                thr_target = _apply_combo_update("thr", thr_local, effects["thr"], ready_masks["thr"], target=thr_target)
                            else:
                                thr_local = _apply_combo_update("thr", thr_local, effects["thr"], ready_masks["thr"])
                        else:
                            if smoothing_active:
                                thr_target = _apply_modulated_update("thr", thr_target, effects["thr"] * ready_masks["thr"])
                            else:
                                thr_local = _apply_modulated_update("thr", thr_local, effects["thr"] * ready_masks["thr"])
                    if "reset" in effects and "reset" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                rst_target = _apply_combo_update(
                                    "reset", rst_local, effects["reset"], ready_masks["reset"], target=rst_target
                                )
                            else:
                                rst_local = _apply_combo_update("reset", rst_local, effects["reset"], ready_masks["reset"])
                        else:
                            if smoothing_active:
                                rst_target = _apply_modulated_update("reset", rst_target, effects["reset"] * ready_masks["reset"])
                            else:
                                rst_local = _apply_modulated_update("reset", rst_local, effects["reset"] * ready_masks["reset"])
                    if "rest" in effects and "rest" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                rpo_target = _apply_combo_update("rest", rpo_local, effects["rest"], ready_masks["rest"], target=rpo_target)
                            else:
                                rpo_local = _apply_combo_update("rest", rpo_local, effects["rest"], ready_masks["rest"])
                        else:
                            if smoothing_active:
                                rpo_target = _apply_modulated_update("rest", rpo_target, effects["rest"] * ready_masks["rest"])
                            else:
                                rpo_local = _apply_modulated_update("rest", rpo_local, effects["rest"] * ready_masks["rest"])
                    if "alpha_2" in effects and "alpha_2" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                alpha_2_target = _apply_combo_update(
                                    "alpha_2", alpha_2_local, effects["alpha_2"], ready_masks["alpha_2"], target=alpha_2_target
                                )
                            else:
                                alpha_2_local = _apply_combo_update("alpha_2", alpha_2_local, effects["alpha_2"], ready_masks["alpha_2"])
                        else:
                            if smoothing_active:
                                alpha_2_target = _apply_modulated_update("alpha_2", alpha_2_target, effects["alpha_2"] * ready_masks["alpha_2"])
                            else:
                                alpha_2_local = _apply_modulated_update("alpha_2", alpha_2_local, effects["alpha_2"] * ready_masks["alpha_2"])
                    if "beta_2" in effects and "beta_2" in ready_masks:
                        if combo_mode:
                            if smoothing_active:
                                beta_2_target = _apply_combo_update(
                                    "beta_2", beta_2_local, effects["beta_2"], ready_masks["beta_2"], target=beta_2_target
                                )
                            else:
                                beta_2_local = _apply_combo_update("beta_2", beta_2_local, effects["beta_2"], ready_masks["beta_2"])
                        else:
                            if smoothing_active:
                                beta_2_target = _apply_modulated_update("beta_2", beta_2_target, effects["beta_2"] * ready_masks["beta_2"])
                            else:
                                beta_2_local = _apply_modulated_update("beta_2", beta_2_local, effects["beta_2"] * ready_masks["beta_2"])

        if smoothing_active:
            alpha_1_local = _smooth_param(alpha_1_local, alpha_1_target, "alpha_1")
            beta_1_local = _smooth_param(beta_1_local, beta_1_target, "beta_1")
            thr_local = _smooth_param(thr_local, thr_target, "thr")
            rst_local = _smooth_param(rst_local, rst_target, "reset")
            rpo_local = _smooth_param(rpo_local, rpo_target, "rest")
            alpha_2_local = _smooth_param(alpha_2_local, alpha_2_target, "alpha_2")
            beta_2_local = _smooth_param(beta_2_local, beta_2_target, "beta_2")
            if psp_norm:
                gain_hidden = _psp_peak_gain(alpha_1_local, beta_1_local, dt=decay_dt)
                gain_out = _psp_peak_gain(alpha_2_local, beta_2_local, dt=decay_dt)
            if mod_current_enable:
                if use_current_hidden and hidden_current_target is not None:
                    if hidden_current_mix is None:
                        hidden_current = hidden_current_target
                    else:
                        hidden_current = hidden_current + hidden_current_mix * (hidden_current_target - hidden_current)
                if use_current_output and output_current_target is not None:
                    if output_current_mix is None:
                        output_current = output_current_target
                    else:
                        output_current = output_current + output_current_mix * (output_current_target - output_current)

        if trace_fn is not None:
            payload = {
                "alpha_1": alpha_1_local.detach(),
                "beta_1": beta_1_local.detach(),
                "thr": thr_local.detach(),
                "reset": rst_local.detach(),
                "rest": rpo_local.detach(),
                "alpha_2": alpha_2_local.detach(),
                "beta_2": beta_2_local.detach(),
            }
            if smoothing_active:
                payload.update({
                    "alpha_1_target": alpha_1_target.detach(),
                    "beta_1_target": beta_1_target.detach(),
                    "thr_target": thr_target.detach(),
                    "reset_target": rst_target.detach(),
                    "rest_target": rpo_target.detach(),
                    "alpha_2_target": alpha_2_target.detach(),
                    "beta_2_target": beta_2_target.detach(),
                })
            trace_fn(t, payload)

    mem_rec = torch.stack(mem_rec, dim=1)      # [B,T,H]
    spk_rec = torch.stack(spk_rec, dim=1)      # [B,T,H]
    out_rec = torch.stack(out_rec, dim=1)      # [B,T,O]
    if snn_mode and training and mod_spk_rec is not None:
        setattr(modulator, "_last_spk_rec", torch.stack(mod_spk_rec, dim=1))  # [B,T,M]
    return out_rec, (mem_rec, spk_rec)

# -------------------------
# Eval helpers (Modulated)
# -------------------------
@torch.no_grad()
def eval_val_metrics_mod(state, settings, x_data, y_data,
                         modulator: nn.Module, mlp_interval: int,
                         indices: Optional[np.ndarray] = None,
                         mlp_input_block_mask: Optional[Dict[str, bool]] = None,
                         mlp_output_block_mask: Optional[Dict[str, bool]] = None):
    loss_fn = nn.NLLLoss(reduction="mean"); log_softmax = nn.LogSoftmax(dim=1)
    nlls, accs = [], []
    gen = sparse_data_generator_from_hdf5_spikes(
        x_data, y_data, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False, indices=indices,
        augment_cfg=settings.get("augment_eval"), postbin_time_mask=settings.get("postbin_mask_eval", 0.0)
    )
    for x_local, y_local in gen:
        out, _ = run_snn_modulated(
            x_local.to_dense(), state, settings, modulator, mlp_interval,
            mlp_input_block_mask=mlp_input_block_mask,
            mlp_output_block_mask=mlp_output_block_mask
        )
        m, _ = torch.max(out, 1)
        log_p = log_softmax(m)
        nlls.append(loss_fn(log_p, y_local).item())
        _, am = torch.max(m, 1)
        accs.append(torch.mean((y_local == am).float()).item())
    return float(np.mean(nlls)) if nlls else float("nan"), float(np.mean(accs)) if accs else 0.0

@torch.no_grad()
def evaluate_testset_mod(state, settings, x_test, y_test, modulator: nn.Module, mlp_interval: int,
                         mlp_input_block_mask: Optional[Dict[str, bool]] = None,
                         mlp_output_block_mask: Optional[Dict[str, bool]] = None):
    loss_fn = nn.NLLLoss(reduction="mean"); log_softmax = nn.LogSoftmax(dim=1)
    nlls, accs = [], []
    gen = sparse_data_generator_from_hdf5_spikes(
        x_test, y_test, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False,
        augment_cfg=settings.get("augment_test"),
        postbin_time_mask=settings.get("postbin_mask_test", 0.0)
    )
    for x_local, y_local in gen:
        out, _ = run_snn_modulated(
            x_local.to_dense(), state, settings, modulator, mlp_interval,
            mlp_input_block_mask=mlp_input_block_mask,
            mlp_output_block_mask=mlp_output_block_mask
        )
        m, _ = torch.max(out, 1)
        log_p = log_softmax(m)
        nlls.append(loss_fn(log_p, y_local).item())
        _, am = torch.max(m, 1)
        accs.append(torch.mean((y_local == am).float()).item())
    test_nll = float(np.mean(nlls)) if nlls else float("nan")
    test_acc = float(np.mean(accs)) if accs else 0.0
    print(f"Final Test (Mod) — NLL: {test_nll:.5f}, Accuracy: {test_acc:.5f}")
    return test_nll, test_acc

def _summarize_spike_activity(total_spikes: float, num_samples: int, nb_hidden: int) -> Dict[str, float]:
    num_samples = max(1, int(num_samples))
    nb_hidden = max(1, int(nb_hidden or 1))
    avg_per_sample = float(total_spikes) / num_samples
    avg_per_neuron = avg_per_sample / nb_hidden
    return {
        "total_spikes": float(total_spikes),
        "num_samples": num_samples,
        "avg_per_sample": avg_per_sample,
        "avg_per_neuron": avg_per_neuron,
    }

def _format_spike_stats(stats: Dict[str, float]) -> str:
    return (
        f"avg/sample={stats['avg_per_sample']:.4f}, avg/neuron={stats['avg_per_neuron']:.6f}, "
        f"samples={stats['num_samples']}, total_spikes={stats['total_spikes']:.2f}"
    )

@torch.no_grad()
def compute_snn_test_spike_stats(state, settings, x_test, y_test) -> Dict[str, float]:
    total_spikes = 0.0
    num_samples = 0
    gen = sparse_data_generator_from_hdf5_spikes(
        x_test, y_test, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False,
        augment_cfg=settings.get("augment_test"),
        postbin_time_mask=settings.get("postbin_mask_test", 0.0)
    )
    for x_local, _ in gen:
        batch_inputs = x_local.to_dense()
        num_samples += batch_inputs.size(0)
        _, (_, spk_rec) = run_snn_hetero(batch_inputs, state, settings)
        total_spikes += float(spk_rec.sum().item())
    return _summarize_spike_activity(total_spikes, num_samples, settings.get("nb_hidden", 1))

@torch.no_grad()
def compute_mod_test_spike_stats(
    state,
    settings,
    modulator: nn.Module,
    mlp_interval: int,
    x_test,
    y_test,
    mlp_input_block_mask: Optional[Dict[str, bool]] = None,
    mlp_output_block_mask: Optional[Dict[str, bool]] = None,
) -> Dict[str, float]:
    total_spikes = 0.0
    num_samples = 0
    modulator.eval()
    gen = sparse_data_generator_from_hdf5_spikes(
        x_test, y_test, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False,
        augment_cfg=settings.get("augment_test"),
        postbin_time_mask=settings.get("postbin_mask_test", 0.0)
    )
    for x_local, _ in gen:
        batch_inputs = x_local.to_dense()
        num_samples += batch_inputs.size(0)
        _, (_, spk_rec) = run_snn_modulated(
            batch_inputs, state, settings, modulator, mlp_interval,
            mlp_input_block_mask=mlp_input_block_mask,
            mlp_output_block_mask=mlp_output_block_mask,
        )
        total_spikes += float(spk_rec.sum().item())
    return _summarize_spike_activity(total_spikes, num_samples, settings.get("nb_hidden", 1))

def _collect_eval_batches(x_data, y_data, settings, batch_limit: int,
                          subset: Optional[np.ndarray] = None,
                          augment_cfg: Optional[dict] = None,
                          postbin_time_mask: float = 0.0):
    """
    Materialize a capped list of dense batches (CPU) for repeated evaluation sweeps.
    """
    batches = []
    gen = sparse_data_generator_from_hdf5_spikes(
        x_data, y_data, settings["batch_size"], settings["nb_steps"],
        settings["nb_inputs"], settings["max_time"], shuffle=False, indices=subset,
        augment_cfg=augment_cfg, postbin_time_mask=postbin_time_mask
    )
    for i, (x_sp, y) in enumerate(gen):
        batches.append((x_sp.to_dense().cpu(), y.cpu()))
        if batch_limit and (i + 1) >= batch_limit:
            break
    return batches

@torch.no_grad()
def _compute_mod_metric_from_batches(
    state,
    settings,
    modulator: nn.Module,
    mlp_interval: int,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    metric: str = "acc",
    mlp_input_block_mask: Optional[Dict[str, bool]] = None,
    mlp_output_block_mask: Optional[Dict[str, bool]] = None,
):
    loss_fn = nn.NLLLoss(reduction="mean")
    log_softmax = nn.LogSoftmax(dim=1)
    total, correct = 0, 0
    losses = []
    for x_cpu, y_cpu in batches:
        x = x_cpu.to(device)
        y = y_cpu.to(device)
        out, _ = run_snn_modulated(
            x, state, settings, modulator, mlp_interval,
            mlp_input_block_mask=mlp_input_block_mask,
            mlp_output_block_mask=mlp_output_block_mask
        )
        m, _ = torch.max(out, 1)
        log_p = log_softmax(m)
        loss = loss_fn(log_p, y)
        losses.append(loss.item())
        pred = torch.argmax(m, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    if metric == "nll":
        return float(np.mean(losses)) if losses else float("nan")
    return correct / max(1, total)

def _approximate_block_shapley(block_names: List[str], metric_fn, nb_samples: int):
    """
    Monte Carlo permutation-based Shapley approximation for a small feature set.
    """
    if not block_names:
        return {}, None
    baseline_mask = {b: False for b in block_names}
    baseline_metric = metric_fn(dict(baseline_mask))
    contribs = {b: 0.0 for b in block_names}
    for _ in range(max(1, nb_samples)):
        perm = block_names[:]
        random.shuffle(perm)
        mask = dict(baseline_mask)
        prev_metric = baseline_metric
        for blk in perm:
            mask[blk] = True
            metric_val = metric_fn(dict(mask))
            contribs[blk] += metric_val - prev_metric
            prev_metric = metric_val
    scale = 1.0 / max(1, nb_samples)
    for blk in contribs:
        contribs[blk] *= scale
    return contribs, baseline_metric

def run_mlp_block_shap_analysis(
    state,
    settings,
    mlp: ModulatingMLP,
    x_train, y_train, x_test, y_test,
    dataset: str = "test",
    nb_samples: int = 16,
    batch_limit: int = 4,
    metric: str = "acc",
    run_dir: Optional[Union[str, Path]] = None,
):
    """
    Approximate Shapley contributions for enabled MLP input/output blocks by
    toggling them and measuring the chosen metric over a fixed mini-dataset.
    """
    dataset = (dataset or "test").lower()
    if dataset not in {"train", "test"}:
        raise ValueError(f"Unsupported SHAP dataset '{dataset}'. Use 'train' or 'test'.")

    if dataset == "train":
        src_x, src_y = x_train, y_train
        augment_cfg = settings.get("augment_eval") if settings.get("augment_eval") else None
        postbin_mask = settings.get("postbin_mask_eval", 0.0)
    else:
        src_x, src_y = x_test, y_test
        augment_cfg = settings.get("augment_test") if settings.get("augment_test") else None
        postbin_mask = settings.get("postbin_mask_test", 0.0)

    batches = _collect_eval_batches(
        src_x, src_y, settings, batch_limit=batch_limit,
        subset=None, augment_cfg=augment_cfg, postbin_time_mask=postbin_mask
    )
    if not batches:
        print("SHAP analysis skipped: no evaluation batches could be collected.")
        return None

    metric = metric.lower()
    metric_sign = -1.0 if metric == "nll" else 1.0  # convert to maximization score
    def _score_to_metric(val: float) -> float:
        return val / metric_sign if metric_sign != 0 else val

    full_metric = _compute_mod_metric_from_batches(
        state, settings, mlp, settings["mlp_interval"], batches,
        metric=metric
    )

    reports = {"dataset": dataset, "metric": metric,
               "full_metric": full_metric, "input": None, "output": None}

    in_blocks = list(mlp.in_slices.keys())
    if in_blocks:
        def score_fn(mask_dict):
            return metric_sign * _compute_mod_metric_from_batches(
                state, settings, mlp, settings["mlp_interval"], batches,
                metric=metric,
                mlp_input_block_mask=mask_dict, mlp_output_block_mask=None
            )
        contribs_score, baseline_score = _approximate_block_shapley(in_blocks, score_fn, nb_samples)
        contribs_metric = {k: _score_to_metric(v) for k, v in contribs_score.items()}
        baseline_metric = _score_to_metric(baseline_score)
        reports["input"] = {
            "blocks": sorted(
                [{"name": k, "delta": contribs_metric[k]} for k in contribs_metric],
                key=lambda itm: abs(itm["delta"]), reverse=True
            ),
            "all_disabled_metric": baseline_metric
        }

    out_blocks = list(mlp.out_slices.keys())
    if out_blocks:
        def score_fn_out(mask_dict):
            return metric_sign * _compute_mod_metric_from_batches(
                state, settings, mlp, settings["mlp_interval"], batches,
                metric=metric,
                mlp_input_block_mask=None, mlp_output_block_mask=mask_dict
            )
        contribs_score, baseline_score = _approximate_block_shapley(out_blocks, score_fn_out, nb_samples)
        contribs_metric = {k: _score_to_metric(v) for k, v in contribs_score.items()}
        baseline_metric = _score_to_metric(baseline_score)
        reports["output"] = {
            "blocks": sorted(
                [{"name": k, "delta": contribs_metric[k]} for k in contribs_metric],
                key=lambda itm: abs(itm["delta"]), reverse=True
            ),
            "all_disabled_metric": baseline_metric
        }

    if run_dir:
        out_path = Path(run_dir) / "mlp_block_shap.json"
        with open(out_path, "w") as fh:
            json.dump(reports, fh, indent=2)
        print(f"Saved MLP block SHAP report to {out_path}")

    return reports

def _pretty_print_shap_report(report: Optional[dict]):
    if not report:
        return
    metric = report["metric"]
    print("\n===== MLP Block SHAP =====")
    print(f"Dataset: {report['dataset']} | Metric: {metric} | All blocks metric: {report['full_metric']:.5f}")
    def _log_section(name, payload):
        if not payload:
            return
        print(f"Top {name} blocks (Δ{metric} relative to disabled baseline {payload['all_disabled_metric']:.5f}):")
        for entry in payload["blocks"]:
            print(f"  - {entry['name']}: {entry['delta']:+.5f}")
    _log_section("input", report.get("input"))
    _log_section("output", report.get("output"))

def maybe_run_ann_shap(
    training_result: dict,
    settings_mod: dict,
    shap_cfg: dict,
    x_train, y_train, x_test, y_test
):
    if not shap_cfg.get("enable"):
        return
    if _is_snn_mode(_select_mlp_mode(settings_mod)):
        print("SHAP skipped: snn_add/snn_sub modes do not expose MLP blocks.")
        return
    if training_result is None or "run_dir" not in training_result:
        print("SHAP skipped: training result missing run_dir.")
        return
    fold_summary = None
    fold_summaries = training_result.get("fold_summaries")
    if isinstance(fold_summaries, list) and fold_summaries:
        fold_summary = fold_summaries[0]
    try:
        ckpt_path = _select_mod_ckpt(training_result["run_dir"], fold_summary)
    except Exception as exc:
        print(f"SHAP skipped: {exc}")
        return

    try:
        state_eval, mlp_eval = load_modulated_checkpoint(ckpt_path, settings_mod)
    except Exception as exc:
        print(f"SHAP skipped (failed to load checkpoint): {exc}")
        return

    report = run_mlp_block_shap_analysis(
        state_eval, settings_mod, mlp_eval,
        x_train, y_train, x_test, y_test,
        dataset=shap_cfg.get("dataset", "test"),
        nb_samples=shap_cfg.get("samples", 16),
        batch_limit=shap_cfg.get("batch_limit", 4),
        metric=shap_cfg.get("metric", "acc"),
        run_dir=training_result["run_dir"]
    )
    _pretty_print_shap_report(report)

def _select_mod_ckpt(run_dir: Union[str, Path], fold_summary: Optional[dict] = None):
    """
    Locate a reasonable checkpoint artifact inside a mod run directory.
    """
    run_dir = Path(run_dir)
    fold_id = fold_summary["fold"] if fold_summary and "fold" in fold_summary else 1
    fold_name = f"Fold_{fold_id}" if (run_dir / f"Fold_{fold_id}").exists() else "Fold_1"
    fold_dir = run_dir / fold_name
    candidates = [
        fold_dir / "mod_best.pth",
        fold_dir / "best_mod.pth",
        fold_dir / "chosen.pth",
        fold_dir / "chosen",
    ]
    best_epoch = None
    if fold_summary:
        best_epoch = fold_summary.get("best_epoch")
        if best_epoch is not None and best_epoch > 0:
            candidates.append(fold_dir / f"mod_{best_epoch}.pth")
    mod_ckpts = sorted(fold_dir.glob("mod_*.pth"), reverse=True)
    candidates.extend(mod_ckpts)
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No modulation checkpoint found under {fold_dir}")


# -------------------------
# Train Modulated SNN
# -------------------------
def train_modulated_snn(
    state: Dict, settings: Dict, x_train, y_train, x_test, y_test,
    modulator: Optional[nn.Module], mlp_interval: int,
    x_val=None, y_val=None,
    save_every_epoch: bool = True,
    use_validation: bool = True, val_fraction: float = 0.1,
    k_folds: int = 1, patience: int = 10, eval_test_each_epoch: bool = False, seed: int = 0,
    fixed_split_path: Optional[str] = None,
    reinit_per_fold: bool = True,
    start_locked: bool = False, unlock_metric: Optional[str] = None, unlock_threshold: Optional[float] = None,
    base_ckpt_path: Optional[Union[str, Path]] = None,
    run_index: Optional[int] = None,
):
    if modulator is not None:
        settings["mlp_init_state_dict"] = copy.deepcopy(modulator.state_dict())

    run_dir = ensure_run_dir(settings["save_dir"], run_index=run_index)
    print(f"Saving to: {run_dir}")
    log_path = run_dir / "training_log.txt"
    log_file = open(log_path, "a", buffering=1)
    print(f"Logging to: {log_path}")
    def log(msg): print(msg) or print(msg, file=log_file)

    snn_disabled = sorted([k for k, v in (settings.get("snn_train_flags") or {}).items() if not v])

    external_val_available = (x_val is not None) and (y_val is not None)
    val_dataset_name = settings.get("val_file")
    use_validation_flag = use_validation or external_val_available
    if external_val_available:
        try:
            val_size = len(y_val)
        except TypeError:
            val_size = "unknown"
        log(f"[info] External validation split detected ({val_dataset_name or 'custom'}), size={val_size}.")

    log_run_header(
        log, run_dir, settings,
        mode="mod",
        use_validation=use_validation_flag,
        val_fraction=val_fraction,
        k_folds=k_folds,
        patience=patience,
        eval_test_each_epoch=eval_test_each_epoch,
        seed=seed,
        fixed_split_path=fixed_split_path,
        reinit_per_fold=reinit_per_fold,
        start_locked=start_locked,
        unlock_metric=unlock_metric,
        unlock_threshold=unlock_threshold,
        mlp_interval=mlp_interval,
        use_input_delay=settings.get("use_input_delay"),
        input_delay_steps=settings.get("input_delay_steps"),
        mod_reg_enabled=settings.get("use_mod_reg"),
        snn_mod_reg_enabled=settings.get("use_snn_mod_reg"),
        snn_train_disabled=snn_disabled,
        external_val=external_val_available,
        external_val_file=val_dataset_name,
    )
    log(format_param_stats(state, modulator, prefix="Initial:", settings=settings))

    # NEW: print base SNN ckpt metrics if provided (unchanged)
    if base_ckpt_path is not None:
        try:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            epoch = ckpt.get("epoch", None)
            metrics = ckpt.get("metrics", None)
            history = ckpt.get("history", None)
            log("---- Base SNN checkpoint info ----")
            log(f"Path: {base_ckpt_path}")
            if epoch is not None:
                log(f"Epoch: {epoch}")
            if metrics:
                for k in ["train_loss","train_acc","val_nll","val_acc","test_nll","test_acc"]:
                    if k in metrics:
                        v = metrics[k]
                        log(f"{k}: {v:.5f}" if isinstance(v, (float,int)) else f"{k}: {v}")
            elif history:
                def _last(lst): 
                    return lst[-1] if isinstance(lst, list) and len(lst)>0 else None
                tl = _last(history.get("train_loss", []))
                ta = _last(history.get("train_acc", []))
                vl = _last(history.get("val_nll", []))
                va = _last(history.get("val_acc", []))
                tsl = _last(history.get("test_nll", []))
                tsa = _last(history.get("test_acc", []))
                log(f"train_loss(last): {tl}" if tl is not None else "train_loss(last): n/a")
                log(f"train_acc(last):  {ta}" if ta is not None else "train_acc(last): n/a")
                log(f"val_nll(last):   {vl}" if vl is not None else "val_nll(last): n/a")
                log(f"val_acc(last):   {va}" if va is not None else "val_acc(last): n/a")
                log(f"test_nll(last):  {tsl}" if tsl is not None else "test_nll(last): n/a")
                log(f"test_acc(last):  {tsa}" if tsa is not None else "test_acc(last): n/a")
            else:
                log("(no metrics/history found in checkpoint)")
            log("-----------------------------------")
        except Exception as e:
            log(f"[warn] Could not read base SNN checkpoint metrics: {e}")
    else:
        log("[info] No base SNN checkpoint provided; using random initialization for mod training.")

    # ----- build splits (unchanged) -----
    folds: List[Tuple[np.ndarray, Optional[np.ndarray], dict]] = []
    if external_val_available:
        tr_idx = np.arange(len(y_train))
        val_idx = None
        idx_path = run_dir / "Fold_1" / "split_indices.npz"
        save_indices(idx_path, tr_idx, val_idx, {"external_validation": True})
        folds = [(tr_idx, val_idx, {"source": "external_val", "path": str(idx_path)})]
        if k_folds and k_folds > 1:
            log("[info] External validation provided; ignoring k-fold configuration.")
        if fixed_split_path:
            log("[info] External validation provided; ignoring --fixed_split_path.")
    elif k_folds and k_folds > 1:
        if fixed_split_path:
            for k in range(1, k_folds+1):
                p = Path(fixed_split_path) if str(fixed_split_path).endswith(".npz") else Path(fixed_split_path)/f"fold_{k}_indices.npz"
                tr_idx, val_idx, meta = load_indices(p); folds.append((tr_idx, val_idx, {"source":"loaded","path":str(p)}))
            log(f"Loaded {k_folds} folds from {fixed_split_path}")
        else:
            for k,(tr_idx,val_idx) in enumerate(make_kfold_splits(y_train, k_folds, seed), start=1):
                idx_path = run_dir / f"Fold_{k}" / "fold_indices.npz"
                save_indices(idx_path, tr_idx, val_idx, {"k_folds":k_folds, "fold":k, "seed":seed})
                folds.append((tr_idx, val_idx, {"source":"created","path":str(idx_path)}))
            log(f"Created & saved {k_folds} stratified folds.")
    else:
        if use_validation_flag:
            if fixed_split_path:
                p = Path(fixed_split_path); tr_idx, val_idx, meta = load_indices(p)
                folds = [(tr_idx, val_idx, {"source":"loaded","path":str(p)})]; log(f"Loaded fixed split from {p}")
            else:
                tr_idx, val_idx = stratified_split_indices(y_train, val_fraction, seed)
                idx_path = run_dir / "Fold_1" / "split_indices.npz"
                save_indices(idx_path, tr_idx, val_idx, {"val_fraction":val_fraction, "seed":seed})
                folds = [(tr_idx, val_idx, {"source":"created","path":str(idx_path)})]; log(f"Saved split to {idx_path}")
        else:
            tr_idx = np.arange(len(y_train)); val_idx = None
            idx_path = run_dir / "Fold_1" / "split_indices.npz"
            save_indices(idx_path, tr_idx, val_idx, {"no_validation":True})
            folds = [(tr_idx, val_idx, {"source":"created","path":str(idx_path)})]; log("No validation: saved train indices.")

    manifest = []

    # ----- train per fold (unchanged control flow) -----
    for fold_id, (tr_idx, val_idx, meta) in enumerate(folds, start=1):
        multi_fold = (k_folds and k_folds > 1) and not external_val_available
        fold_dir = run_dir / (f"Fold_{fold_id}" if multi_fold else "Fold_1"); fold_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== Fold {fold_id}/{len(folds)} ({meta['source']}) — indices: {meta['path']} ===")
        if external_val_available and y_val is not None:
            log(f"Train size: {len(tr_idx)} / External Val size: {len(y_val)}")
        elif val_idx is None:
            log(f"Train size (no validation): {len(tr_idx)}")
        else:
            log(f"Train/Val sizes: {len(tr_idx)} / {len(val_idx)}")

        state_fold = (
            {k: torch.nn.Parameter(v.detach().clone(), requires_grad=v.requires_grad) for k,v in state.items()}
            if reinit_per_fold else state
        )
        fold_mode = _select_mlp_mode(settings)
        fold_hidden_sizes = _resolve_mlp_hidden_sizes(settings)
        fold_mod_hidden_sizes = settings.get("mod_hidden_sizes")
        hidden_cfg = fold_mod_hidden_sizes if _is_snn_mode(fold_mode) else fold_hidden_sizes
        modulator_fold = build_modulator(
            settings,
            override_mode=fold_mode,
            hidden_sizes=hidden_cfg,
        )
        log(format_param_stats(state_fold, modulator_fold, prefix=f"Fold {fold_id}:", settings=settings))

        if settings.get("mlp_init_state_dict") is not None:
            modulator_fold.load_state_dict(settings["mlp_init_state_dict"])

        mod_optimizer = torch.optim.Adam(
            [{"params": modulator_fold.parameters(), "lr": settings["lr"]}],
            betas=(0.9, 0.999),
        )
        snn_optimizer = None
        snn_unlocked = False
        snn_trainables = _trainable_params_from(state_fold)
        if (not start_locked) and snn_trainables:
            snn_optimizer = torch.optim.Adam([{"params": snn_trainables, "lr": settings["lr"]}], betas=(0.9, 0.999))
            snn_unlocked = True

        loss_fn = nn.NLLLoss(); log_softmax = nn.LogSoftmax(dim=1)
        snn_mode_flag = _is_snn_mode(fold_mode)

        def _pack_params(): return {k: v.clone() for k, v in state_fold.items()}
        def _save_ckpt(path: Path, epoch: int, metrics: dict, history: dict):
            torch.save({
                "epoch": epoch,
                "metrics": metrics,
                "snn_params": _pack_params(),
                "mlp_state": modulator_fold.state_dict(),
                "history": history,
                "mlp_mode": settings.get("mlp_mode", DEFAULT_MLP_MODE),
                "ann_mode": settings.get("ann_mode", settings.get("mlp_mode", DEFAULT_MLP_MODE)),
                "mlp_interval": settings.get("mlp_interval"),
                "ann_interval": settings.get("ann_interval", settings.get("mlp_interval")),
                "mod_update_every_step": settings.get("mod_update_every_step", False),
                "mlp_hidden_sizes": settings.get("mlp_hidden_sizes"),
                "mlp_arch": settings.get("mlp_arch", DEFAULT_MLP_ARCH),
                "mod_hidden_sizes": settings.get("mod_hidden_sizes"),
                "snn_mod_hidden": settings.get("snn_mod_hidden"),
                "snn_mod_hidden_recurrent": settings.get("snn_mod_hidden_recurrent", False),
                "run_config": _config_snapshot(settings),
            }, path)

        history = {"train_loss": [], "val_nll": [], "val_acc": [], "train_acc": [], "test_nll": [], "test_acc": []}
        best_val_nll, best_val_acc, best_val_epoch = float("inf"), -1.0, -1
        best_ckpt_epoch = -1
        val_data_x = x_val if external_val_available else x_train
        val_data_y = y_val if external_val_available else y_train
        val_indices = None if external_val_available else val_idx
        val_count = len(val_data_y) if (external_val_available and val_data_y is not None) else (len(val_idx) if val_idx is not None else 0)
        has_validation = use_validation_flag and (val_count > 0)
        epochs_no_improve = 0

        for e in range(settings["nb_epochs"]):
            local_loss, accs = [], []
            train_gen = sparse_data_generator_from_hdf5_spikes(
                x_train, y_train, settings["batch_size"], settings["nb_steps"],
                settings["nb_inputs"], settings["max_time"], shuffle=True, indices=tr_idx,
                augment_cfg=settings.get("augment_train"), postbin_time_mask=settings.get("postbin_mask_train", 0.0)
            )
            for x_local, y_local in train_gen:
                out, (mem_rec, spk_rec) = run_snn_modulated(
                    x_local.to_dense(), state_fold, settings, modulator_fold, settings["mlp_interval"],
                    training=True
                )
                m,_ = torch.max(out, 1)
                _, am = torch.max(m, 1)
                accs.append(torch.mean((y_local == am).float()).item())
                log_p = log_softmax(m); ground_loss = loss_fn(log_p, y_local)

                loss_val = ground_loss

                if settings.get("use_mod_reg", True):
                    nb_hidden = settings["nb_hidden"]; nb_outputs = settings["nb_outputs"]
                    N_samp, T = 8128, settings["nb_steps"]
                    sl, thetal = 1.0, 0.01; su, thetau = 0.06, 100.0
                    tmp  = (torch.clamp((1/T)*torch.sum(spk_rec, 1) - thetal, min=0.0))**2
                    L1   = torch.sum(tmp, (0,1))
                    tmp2 = (torch.clamp((1/nb_hidden)*torch.sum(spk_rec, (1,2)) - thetau, min=0.0))**2
                    L2   = torch.sum(tmp2)
                    reg_loss = (sl/(N_samp+nb_hidden+nb_outputs))*L1 + (su/N_samp)*L2
                    reg_loss = settings.get("mod_reg_scale", 1.0) * reg_loss
                    loss_val = loss_val + reg_loss

                if snn_mode_flag and settings.get("use_snn_mod_reg", False):
                    mod_spk_trace = getattr(modulator_fold, "_last_spk_rec", None)
                    if mod_spk_trace is not None:
                        mod_units = int(mod_spk_trace.size(-1))
                        N_samp, T = 8128, settings["nb_steps"]
                        sl, thetal = 1.0, 0.01; su, thetau = 0.06, 100.0
                        tmp  = (torch.clamp((1/T)*torch.sum(mod_spk_trace, 1) - thetal, min=0.0))**2
                        L1   = torch.sum(tmp, (0,1))
                        tmp2 = (torch.clamp((1/max(1, mod_units))*torch.sum(mod_spk_trace, (1,2)) - thetau, min=0.0))**2
                        L2   = torch.sum(tmp2)
                        mod_reg_loss = (sl/(N_samp+mod_units))*L1 + (su/N_samp)*L2
                        mod_reg_loss = settings.get("snn_mod_reg_scale", 1.0) * mod_reg_loss
                        loss_val = loss_val + mod_reg_loss
                mod_optimizer.zero_grad()
                if snn_optimizer is not None:
                    snn_optimizer.zero_grad()
                loss_val.backward()
                mod_optimizer.step()
                if snn_optimizer is not None:
                    snn_optimizer.step()
                if snn_mode_flag and isinstance(modulator_fold, SNNAdditiveModulator):
                    modulator_fold.clamp_mod_factors_()

                with torch.no_grad():
                    _clamp_state_params(state_fold, _resolve_param_ranges(settings))

                local_loss.append(float(loss_val.item()))

            train_loss = float(np.mean(local_loss)) if local_loss else float("nan")
            train_acc = float(np.mean(accs)) if accs else 0.0
            history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)

            test_nll = test_acc = None
            if eval_test_each_epoch:
                test_nll, test_acc = evaluate_testset_mod(
                    state_fold, settings, x_test, y_test, modulator_fold, settings["mlp_interval"]
                )
                history["test_nll"].append(test_nll)
                history["test_acc"].append(test_acc)

            if has_validation and val_data_x is not None and val_data_y is not None:
                val_nll, val_acc = eval_val_metrics_mod(
                    state_fold, settings, val_data_x, val_data_y,
                    indices=val_indices, modulator=modulator_fold, mlp_interval=settings["mlp_interval"]
                )
                history["val_nll"].append(val_nll); history["val_acc"].append(val_acc)
                log_msg = (
                    f"[Fold {fold_id}] Epoch {e+1}: train_loss={train_loss:.5f} "
                    f"train_acc={train_acc:.5f} val_nll={val_nll:.5f} val_acc={val_acc:.5f}"
                )
                if eval_test_each_epoch:
                    log_msg += f" test_nll={test_nll:.5f} test_acc={test_acc:.5f}"
                log(log_msg)

                metrics_payload = {
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_nll": val_nll, "val_acc": val_acc,
                }
                if eval_test_each_epoch:
                    metrics_payload.update({"test_nll": test_nll, "test_acc": test_acc})
                if save_every_epoch:
                    _save_ckpt(fold_dir / f"mod_{e+1}.pth", e+1, metrics_payload, history)

                # unlock logic (unchanged)
                if (unlock_metric is not None) and (unlock_threshold is not None):
                    cond = (
                        (unlock_metric == "val_acc" and val_acc  >= unlock_threshold) or
                        (unlock_metric == "train_acc" and train_acc >= unlock_threshold)
                    )
                    if cond and not snn_unlocked:
                        log(f"Unlocking SNN ({unlock_metric} ≥ {unlock_threshold:.5f})")
                        if snn_trainables:
                            snn_optimizer = torch.optim.Adam([{"params": snn_trainables, "lr": settings["lr"]}], betas=(0.9, 0.999))
                        snn_unlocked = True

                improved = val_nll < best_val_nll - 1e-6
                if improved:
                    best_val_nll, best_val_acc, best_val_epoch = val_nll, val_acc, e + 1
                    best_ckpt_epoch = best_val_epoch
                    epochs_no_improve = 0
                    _save_ckpt(fold_dir / "mod_best.pth", best_ckpt_epoch, metrics_payload, history)
                    log(f"[Fold {fold_id}] New best validation loss at epoch {best_val_epoch} "
                        f"(val_nll={best_val_nll:.5f}, val_acc={best_val_acc:.5f})")
                else:
                    epochs_no_improve += 1
                    log(f"[Fold {fold_id}] No validation-loss improvement "
                        f"({epochs_no_improve}/{patience}) — best val_nll={best_val_nll:.5f} "
                        f"at epoch {best_val_epoch}")

                if epochs_no_improve >= patience:
                    log(f"[Fold {fold_id}] Early stopping."); break
            else:
                log_msg = (
                    f"[Fold {fold_id}] Epoch {e+1}: train_loss={train_loss:.5f} "
                    f"train_acc={train_acc:.5f} (no validation)"
                )
                if eval_test_each_epoch:
                    log_msg += f" test_nll={test_nll:.5f} test_acc={test_acc:.5f}"
                log(log_msg)

                metrics_payload = {"train_loss": train_loss, "train_acc": train_acc}
                if eval_test_each_epoch:
                    metrics_payload.update({"test_nll": test_nll, "test_acc": test_acc})
                if save_every_epoch:
                    _save_ckpt(fold_dir / f"mod_{e+1}.pth", e+1, metrics_payload, history)

                if train_loss < best_val_nll - 1e-6:
                    best_val_nll, best_val_epoch = train_loss, e + 1
                    best_ckpt_epoch = best_val_epoch
                    _save_ckpt(fold_dir / "mod_best.pth", best_ckpt_epoch, metrics_payload, history)

        manifest.append({"fold": fold_id, "indices_path": meta["path"], "best_epoch": best_ckpt_epoch})

        best_mod_ckpt = fold_dir / "mod_best.pth"
        if best_mod_ckpt.exists():
            try:
                state_eval, mlp_eval = load_modulated_checkpoint(best_mod_ckpt, settings, quiet=True)
                final_test_nll, final_test_acc = evaluate_testset_mod(
                    state_eval, settings, x_test, y_test, mlp_eval, settings["mlp_interval"]
                )
                log(f"[Fold {fold_id}] Selected-checkpoint test (Mod) — "
                    f"NLL: {final_test_nll:.5f}, Accuracy: {final_test_acc:.5f}")
                ckpt = torch.load(best_mod_ckpt, map_location="cpu")
                ckpt["final_test"] = {"nll": final_test_nll, "accuracy": final_test_acc}
                torch.save(ckpt, best_mod_ckpt)

                spike_stats = compute_mod_test_spike_stats(
                    state_eval,
                    settings,
                    mlp_eval,
                    settings["mlp_interval"],
                    x_test,
                    y_test,
                )
                log(f"[Fold {fold_id}] Test spike stats (Mod) — {_format_spike_stats(spike_stats)}")
            except Exception as exc:
                log(f"[Fold {fold_id}] Final test/spike-stats computation failed (Mod): {exc}")
        else:
            log(f"[Fold {fold_id}] Spike stats (Mod) skipped: {best_mod_ckpt} not found.")

    log_file.close()
    return {"fold_summaries": manifest, "run_dir": str(run_dir)}

# ===== Block 2: Adjustables + Run (CLI) =====
import argparse

def _parse_mask(disable_csv: str, all_keys: list) -> dict:
    disabled = set([k.strip() for k in disable_csv.split(",") if k.strip()]) if disable_csv else set()
    return {k: (k not in disabled) for k in all_keys}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unified SNN / Modulated SNN runner")

    # --- core model/dataset/sim ---
    p.add_argument("--run_mode", type=str, default="mod", choices=["snn", "mod", "staged"])
    p.add_argument("--nb_inputs", type=int, default=700)
    p.add_argument("--nb_hidden", type=int, default=256)
    p.add_argument("--nb_outputs", type=int, default=20)
    p.add_argument("--time_step", type=float, default=1e-3)
    p.add_argument("--nb_steps", type=int, default=100)
    p.add_argument("--use_input_delay", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable learnable per-input delay kernel between inputs and hidden layer.")
    p.add_argument("--input_delay_steps", type=int, default=1,
                   help="Max causal delay window (steps) per input; used only when --use_input_delay is True.")
    p.add_argument("--input_delay_init", type=int, default=None,
                   help="Optional fixed initial delay index; if omitted, random per-input init within cap.")
    p.add_argument("--input_delay_init_cap", type=int, default=None,
                   help="Cap for random initial delay (0..cap); defaults to input_delay_steps when omitted.")
    p.add_argument("--input_delay_init_bias", type=float, default=0.1,
                   help="Small bias added to the chosen initial delay logit.")
    p.add_argument("--input_delay_init_noise", type=float, default=0.01,
                   help="Stddev of Gaussian noise added to delay logits at init.")
    p.add_argument("--input_delay_temp", type=float, default=1.0,
                   help="Softmax temperature for delay selection (straight-through argmax uses this temp).")
    p.add_argument("--max_time", type=float, default=1.4)
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--cache_dir", type=str, default="~/data")
    p.add_argument("--cache_subdir", type=str, default="hdspikes")
    p.add_argument("--train_file", type=str, default="shd_train.h5")
    p.add_argument("--test_file", type=str, default="shd_test.h5")
    p.add_argument("--val_file", type=str, default=None,
                   help="Optional validation H5 file; if provided, skips splitting train set.")

    # --- training schedule/hparams ---
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--tau_syn", type=float, default=10e-3)
    p.add_argument("--tau_mem", type=float, default=20e-3)
    p.add_argument("--tau_match_clip", type=str2bool, nargs="?", const=True, default=False,
                   help="If True, convert tau_syn/tau_mem using max_time/nb_steps so they follow clip duration.")
    p.add_argument("--weight_scale", type=float, default=0.2)
    p.add_argument("--psp_norm_peak", type=str2bool, nargs="?", const=True, default=False,
                   help="Normalize bi-exponential PSP peaks to 1 (scales syn->mem and output filters).")

    # single-phase default epochs (used for run_mode snn/mod, and as fallback)
    p.add_argument("--nb_epochs", type=int, default=30)

    # staged-specific epoch counts
    p.add_argument("--nb_epochs_snn", type=int, default=30, help="Epochs for Stage-1 SNN in staged mode.")
    p.add_argument("--nb_epochs_mod", type=int, default=70, help="Epochs for Stage-2 Mod in staged mode.")


    # --- save dirs ---
    p.add_argument("--save_dir_root", type=str, default=None,
                   help="Shared base directory; SNN and Mod runs will be stored under SNN/ and NM/ subfolders.")
    p.add_argument("--save_dir_snn", type=str, default="Runs_SNN_EXP")
    p.add_argument("--save_dir_mod", type=str, default="Runs_Mod_EXP")

    # --- loss regularization toggles ---
    p.add_argument("--snn_reg_enable", type=str2bool, nargs="?", const=True, default=True,
                   help="Enable spike regularization term during SNN training. Default=True.")
    p.add_argument("--mod_reg_enable", type=str2bool, nargs="?", const=True, default=True,
                   help="Enable spike regularization term during modulated training. Default=True.")
    p.add_argument("--snn_mod_reg_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable spike regularization term for the modulator SNN (snn_add/snn_sub). Default=False.")
    p.add_argument("--snn_reg_scale", type=float, default=1.0,
                   help="Scalar multiplier applied to the SNN regularization term. Default=1.0.")
    p.add_argument("--mod_reg_scale", type=float, default=1.0,
                   help="Scalar multiplier applied to the Mod regularization term. Default=1.0.")
    p.add_argument("--snn_mod_reg_scale", type=float, default=1.0,
                   help="Scalar multiplier applied to the modulator-SNN regularization term. Default=1.0.")

    # --- augmentation (train) ---
    p.add_argument("--aug_jitter_ms", type=float, default=3.0)
    p.add_argument("--aug_shift_ms", type=float, default=15.0)
    p.add_argument("--aug_scale_low", type=float, default=0.95)
    p.add_argument("--aug_scale_high", type=float, default=1.05)
    p.add_argument("--aug_drop_p", type=float, default=0.10)
    p.add_argument("--aug_insert_rate", type=float, default=0.005)
    p.add_argument("--aug_band_frac", type=float, default=0.05)
    p.add_argument("--aug_channel_jitter_std", type=float, default=20.0,
                   help="Std dev (channels) for Gaussian jitter of input unit index (sigma_u in paper).")
    p.add_argument("--aug_noise_rate_hz", type=float, default=0.0,
                   help="Poisson background noise rate (spikes/s) injected per sample, as in Heidelberg paper.")
    p.add_argument("--aug_noise_per_input", type=str2bool, nargs="?", const=True, default=False,
                   help="Interpret aug_noise_rate_hz as a per-input rate (default: global rate).")
    p.add_argument("--channel_compress_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable channel compression (coarse-grain neighboring input channels).")
    p.add_argument("--channel_compress_target", type=int, default=70,
                   help="Target number of input channels when compression is enabled (paper used 70 from 700 via groups of 10).")
    p.add_argument("--channel_compress_mode", type=str, default="all",
                   choices=["all", "mod_only", "mod_mlp"],
                   help="Where to apply compression: all inputs (default), mod_only (modulator only), mod_mlp (learned MLP compression for modulator).")
    p.add_argument("--channel_compress_mlp_hidden_sizes", type=str, default="[]",
                   help="Hidden sizes for modulator input compression MLP when channel_compress_mode=mod_mlp (e.g., \"[256]\").")
    p.add_argument("--postbin_time_mask_train", type=float, default=0.05)
    p.add_argument("--hidden_dropout_p", type=float, default=0.0,
                   help="Dropout prob on hidden spikes during training (0 disables).")
    
    # --- augmentation toggles ---
    p.add_argument("--train_aug_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable augmentation during TRAIN. Default=False.")
    p.add_argument("--eval_aug_enable",  type=str2bool, nargs="?", const=True, default=False,
                   help="Enable augmentation during VALIDATION. Default=False.")
    p.add_argument("--test_aug_enable",  type=str2bool, nargs="?", const=True, default=False,
                   help="Enable augmentation during TEST. Default=False.")
    p.add_argument("--paper_aug_train", type=str2bool, nargs="?", const=True, default=False,
                   help="Use only paper-style aug/noise (channel jitter + optional Poisson) for TRAIN; disables legacy aug knobs.")
    p.add_argument("--paper_aug_eval", type=str2bool, nargs="?", const=True, default=False,
                   help="Use only paper-style aug/noise for VALIDATION; disables legacy aug knobs.")
    p.add_argument("--paper_aug_test", type=str2bool, nargs="?", const=True, default=False,
                   help="Use only paper-style aug/noise for TEST; disables legacy aug knobs.")
    p.add_argument("--paper_tau_scale_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable time-constant scaling factor (paper best: 4x).")
    p.add_argument("--paper_tau_scale", type=float, default=4.0,
                   help="Scaling factor applied to tau_mem and tau_syn when enabled.")
    p.add_argument("--train_noise_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable Poisson noise injection during TRAIN split.")
    p.add_argument("--eval_noise_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable Poisson noise injection during VALIDATION split.")
    p.add_argument("--test_noise_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable Poisson noise injection during TEST split.")

    # --- test-time postbin mask (independent from eval) ---
    p.add_argument("--postbin_time_mask_test", type=float, default=0.0)


    # --- eval augmentation (default: clean/no aug) ---
    # p.add_argument("--eval_aug_enable", action="store_true", help="If set, use same aug knobs for eval; default off")
    p.add_argument("--postbin_time_mask_eval", type=float, default=0.0)

    # --- split / training controls ---
    p.add_argument("--use_validation", type=str2bool, nargs="?", const=True, default=True,
                    help="Whether to use validation (True/False). Default=True.")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--k_folds", type=int, default=1)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument(
        "--eval_test_each_epoch", type=str2bool, nargs="?", const=True, default=False,
        help="Evaluate the test set after every epoch for diagnostics only. "
             "Default=False; checkpoint selection never uses test metrics.",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--fixed_split_path", type=str, default=None)
    p.add_argument("--reinit_per_fold", type=str2bool, nargs="?", const=True, default=True,
                    help="Reinitialize model per fold (True/False). Default=True.")

    # --- staged controls ---
    p.add_argument("--staged_unlock_metric", type=str, default="train_acc", choices=["train_acc", "val_acc"])
    p.add_argument("--staged_unlock_threshold", type=float, default=0.85)
    p.add_argument("--staged_min_epochs", type=int, default=1)

    # --- mod-only / MLP ---
    p.add_argument("--start_locked", type=str2bool, nargs="?", const=True, default=False,
                    help="Lock SNN params at start of mod training (True/False). Default=False.")
    p.add_argument("--unlock_metric", type=str, default=None, choices=[None, "val_acc", "train_acc"])
    p.add_argument("--unlock_threshold", type=float, default=None)

    # --- Modulator (ANN) configuration ---
    p.add_argument("--ann_hidden_sizes", type=str, default="[2048]",
                   help="List of hidden sizes for the ANN/RNN/LSTM modulator (e.g. \"[2048, 1024]\").")
    p.add_argument("--ann_interval", type=int, default=3,
                   help="Cadence (in steps) between modulation updates.")
    p.add_argument("--ann_mode", type=str, default="ann_sub",
                   choices=["ann_sub", "ann_add", "ann_combo", "snn_add", "snn_sub"],
                   help="Modulation strategy: ann_sub (~mlp_sub), ann_add (~mlp_add), ann_combo (~mlp_combo), snn_add, or snn_sub.")
    p.add_argument("--ann_combo_additive", type=str, nargs="+", default=None,
                   help="Comma-separated or JSON list of params treated additively in ann_combo (default: thr, reset, rest).")
    p.add_argument("--ann_combo_multiplicative", type=str, nargs="+", default=None,
                   help="Comma-separated or JSON list of params treated multiplicatively in ann_combo (default: remaining params).")
    p.add_argument("--ann_arch", type=str, default="mlp", choices=["mlp", "rnn", "lstm"],
                   help="Backbone used for ann_sub/ann_add modulators.")
    p.add_argument("--ann_rnn_state_every_step", type=str2bool, nargs="?", const=True, default=False,
                   help="If True and ann_arch is rnn/lstm, advance recurrent state every timestep even when ann_interval>1 (updates still gated by interval).")
    p.add_argument("--mod_update_every_step", type=str2bool, nargs="?", const=True, default=False,
                   help="If True, apply modulation updates every timestep instead of only on ann_interval.")
    p.add_argument("--mod_current_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Use modulator outputs as injected currents instead of parameter updates.")
    p.add_argument("--mod_current_target", type=str, default="both",
                   choices=["hidden", "output", "both"],
                   help="Where to inject modulator currents when mod_current_enable is true.")
    p.add_argument("--mod_current_activation", type=str, default="tanh",
                   choices=["sigmoid", "tanh"],
                   help="Activation on modulator current outputs when mod_current_enable is true.")
    p.add_argument("--mod_hidden_sizes", type=str, default=None,
                   help="Hidden layer sizes for the modulator (falls back to --ann_hidden_sizes when omitted).")
    p.add_argument("--snn_mod_hidden", type=int, default=None,
                   help="(Deprecated) Number of modulator neurons. In snn_add mode this is derived automatically (10*nb_hidden + 4*nb_outputs).")
    p.add_argument("--snn_mod_hidden_recurrent", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable recurrent connections inside secondary SNN hidden layers.")
    p.add_argument("--snn_mod_rec_init_zero", type=str2bool, nargs="?", const=True, default=True,
                   help="If True and snn_mod_hidden_recurrent is enabled, initialize modulator SNN recurrent weights to 0. Default=True.")
    p.add_argument("--snn_mod_gain_init", type=float, default=SNN_ADD_GAIN_INIT_DEFAULT,
                   help="Initial gain per modulator neuron for snn_add mode (small near-zero start).")
    p.add_argument("--snn_mod_weight_scale", type=float, default=SNN_ADD_WEIGHT_SCALE_DEFAULT,
                   help="Scale used when initializing snn_add input weights (small near-zero default).")
    p.add_argument("--snn_add_balanced_init", type=str2bool, nargs="?", const=True, default=False,
                   help="If True (snn_add), initialize paired +/- modulator neurons to cancel initially (pass-through start). Default=False.")
    p.add_argument("--snn_add_init_effect_frac", type=float, default=0.05,
                   help="When snn_add_balanced_init is True, set per-side gain to this fraction of each parameter's range width (net still cancels). Default=0.05.")
    p.add_argument("--snn_sub_scale_init", type=float, default=SNN_SUB_SCALE_INIT_DEFAULT,
                   help="Initial multiplicative factor applied to snn_sub membrane potentials before squashing to parameter ranges.")
    p.add_argument("--snn_sub_bias_init", type=float, default=SNN_SUB_BIAS_INIT_DEFAULT,
                   help="Initial additive bias applied to snn_sub membrane potentials before squashing to parameter ranges.")
    # Neuromodulator-inspired modulation knobs.
    p.add_argument("--nm_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable neuromodulator-based modulation (snn_add/ann_sub/ann_add/ann_combo).")
    p.add_argument("--nm_counts", type=str, default="[0,0]",
                   help="List of neuromodulator type counts per layer (e.g. \"[6,5]\" for hidden/output).")
    p.add_argument("--nm_init_scale", type=float, default=SOFT_MOD_GAIN_INIT,
                   help="Scale for initializing shared neuromodulator MLP weights (small near-zero default).")
    p.add_argument("--nm_debug_print", type=str2bool, nargs="?", const=True, default=False,
                   help="Print NM mapper effect stats at t=0 (debug).")
    p.add_argument("--nm_neuron_frac_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable random neuron masking for NM effects (fractions per layer).")
    p.add_argument("--nm_neuron_frac", type=str, default="[1.0,1.0]",
                   help="Fractions of neurons modulated in [hidden, output] (e.g., \"[0.8,0.6]\").")
    p.add_argument("--nm_param_frac_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable random parameter masking per neuron for NM effects.")
    p.add_argument(
        "--nm_param_frac",
        type=str,
        default='{"alpha_1":1.0,"beta_1":1.0,"thr":1.0,"reset":1.0,"rest":1.0,"alpha_2":1.0,"beta_2":1.0}',
                   help="Parameter fractions per neuron. Accepts [hidden, output] or a dict like "
                        "'{\"alpha_1\":0.5,\"beta_1\":0.5,\"thr\":0.5,\"reset\":0.5,"
                        "\"rest\":0.5,\"alpha_2\":1.0,\"beta_2\":1.0}'.")
    p.add_argument("--nm_mapper_type", type=str, default="mlp",
                   choices=["mlp", "linear"],
                   help="Neuromodulator mapper type: full MLP (default) or single linear layer.")
    p.add_argument("--nm_mapper_activation", type=str, default="auto",
                   choices=["auto", "linear", "sigmoid", "tanh", "none"],
                   help="Final activation for NM mapper. 'auto' matches mode (sigmoid for sub, tanh for add).")
    p.add_argument("--nm_mapper_hidden_activation", type=str, default="silu",
                   choices=["silu", "gelu", "leakyrelu", "tanh", "relu", "none"],
                   help="Hidden activation for NM mapper MLP (ignored for linear mapper).")
    p.add_argument("--nm_mapper_hidden_size", type=int, default=None,
                   help="Override NM mapper hidden width (applies to both hidden/output mapper MLPs).")
    p.add_argument("--nm_flat_order", type=str, default="type_major",
                   choices=["type_major", "target_major"],
                   help="How NM levels are packed in the flat modulator output. "
                        "type_major packs by modulator type then target index; target_major packs by target then type.")
    p.add_argument("--nm_warm_start", type=str2bool, nargs="?", const=True, default=True,
                   help="Warm-start NM to behave like normal mode (best with --nm_counts [5,2]). "
                        "Enables identity init for the NM mapper and sets flat order to type_major unless overridden.")
    p.add_argument("--nm_warm_start_force_linear", type=str2bool, nargs="?", const=True, default=False,
                   help="When nm_warm_start is true and nm_counts is [5,2], force nm_mapper_type=linear for the closest "
                        "match to normal mode. Set false to keep an MLP mapper (optionally with nm_mapper_hidden_activation=none).")
    p.add_argument("--mod_fixed_mask_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Use fixed neuron/param masks for the whole run and shrink modulator IO (applies to NM and non-NM).")
    p.add_argument("--mod_fixed_mask_seed", type=int, default=None,
                   help="Seed for fixed modulation mask sampling (defaults to --seed when unset).")
    p.add_argument("--mod_fixed_mask_flat_inputs", type=str2bool, nargs="?", const=True, default=False,
                   help="Also reduce hid_flat/out_flat inputs using the fixed neuron mask.")
    p.add_argument("--mod_fixed_mask_zero_fallback", type=str2bool, nargs="?", const=True, default=False,
                   help="Legacy behavior: if True, 0.0 fractions fall back to full selection instead of empty.")
    p.add_argument("--mod_hid_flat_group", type=str2bool, nargs="?", const=True, default=False,
                   help="Group-compress hid_flat inputs using the hidden group layout.")
    p.add_argument("--mod_hid_flat_modulated_only", type=str2bool, nargs="?", const=True, default=False,
                   help="Limit hid_flat inputs to neurons/groups with any modulated parameter (requires fixed mask).")

    # Block masking helpers (comma-separated lists of ANN inputs/outputs to disable).
    p.add_argument("--ann_in_disable", type=str, default="",
                   help="Comma-separated ANN input blocks to disable.")
    p.add_argument("--ann_out_disable", type=str, default="",
                   help="Comma-separated ANN output blocks to disable.")
    p.add_argument("--ann_output_activation", type=str.lower, default=None,
                   choices=list(ANN_OUTPUT_ACTIVATIONS.keys()) + ["none", "default"],
                   help="Optional ANN/RNN/LSTM output activation override (linear/sigmoid/tanh); use 'default' or 'none' to keep the mode-specific default.")

    # Grouping controls: share modulation deltas across overlapping neurons (format [hidden, output]).
    p.add_argument("--group_size", type=str, nargs="+", default=["1", "1"],
                   help="Grouping window size for [hidden, output] layers.")
    p.add_argument("--group_overlap", type=str, nargs="+", default=["0", "0"],
                   help="Number of neurons of overlap for [hidden, output] groups (uniform: overlapping contributions are averaged).")
    p.add_argument("--group_distribution", type=str, nargs="+", default=["uniform", "uniform"],
                   help="Distribution per layer (\"uniform\" or \"normal\"). Normal uses soft Gaussian falloff; overlap is ignored.")
    p.add_argument("--group_normal_std", type=str, nargs="+", default=["1.0", "1.0"],
                   help="Stddev when distribution is 'normal' for [hidden, output] (space-separated or JSON list).")

    # Weight modulation controls. Weight outputs are grouped because full dense weight modulation is usually too large.
    p.add_argument("--weight_mod_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable grouped modulation of SNN weights.")
    p.add_argument("--weight_mod_targets", type=str, default="all",
                   help="Comma/JSON list of weight matrices to modulate: w1,v1,w2 or all.")
    p.add_argument("--weight_mod_scope", type=str, default="block",
                   choices=sorted(WEIGHT_SCOPE_CHOICES),
                   help="Weight grouping scope: source rows, target columns, or source-target blocks.")
    p.add_argument("--weight_group_dim", type=str, default="1d",
                   choices=sorted(WEIGHT_GRID_CHOICES),
                   help="Treat each source/target axis as a 1D line or nearest rectangular 2D grid.")
    p.add_argument("--weight_group_size", type=int, default=1,
                   help="Weight group width. In 2D this is the square side length, so 2 covers 4 neurons.")
    p.add_argument("--weight_group_overlap", type=int, default=0,
                   help="Overlap between adjacent weight groups. Uniform overlapping contributions are averaged.")
    p.add_argument("--weight_group_distribution", type=str, default="uniform",
                   choices=sorted(GROUP_DISTRIBUTIONS),
                   help="Weight group distribution: uniform hard groups or normal soft falloff.")
    p.add_argument("--weight_group_normal_std", type=float, default=1.0,
                   help="Stddev for weight groups when distribution is normal.")
    p.add_argument("--weight_mod_mode", type=str, default="auto",
                   choices=["auto", "additive", "substitution", "multiplicative"],
                   help="How grouped weight effects are applied. Auto follows ANN sub/add mode; multiplicative is optional.")
    p.add_argument("--weight_mod_delta_scale", type=float, default=0.05,
                   help="Additive weight delta scale as a fraction of each matrix reference scale.")
    p.add_argument("--weight_mod_range_std_scale", type=float, default=3.0,
                   help="Substitution/multiplicative reference range uses max(abs(weight), std*this scale).")

    # Per-parameter modulation cadence (learned offsets).
    p.add_argument("--param_timescales_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable learnable per-parameter modulation intervals.")
    p.add_argument("--param_timescales_distribution", type=str, default="fixed",
                   choices=["fixed", "uniform", "normal"])
    p.add_argument("--param_timescales_scale", type=float, default=0.0,
                   help="Scale for initializing learnable intervals when distribution != fixed.")
    p.add_argument("--param_timescales_std", type=float, default=0.0,
                   help="Stddev for timescale offsets (used when distribution=normal; falls back to scale).")
    p.add_argument("--param_timescales_seed", type=int, default=None,
                   help="Optional RNG seed for initializing learnable intervals.")
    p.add_argument("--param_timescales_trainable", type=str2bool, nargs="?", const=True, default=True,
                   help="Toggle requires_grad for param timescale offsets.")
    # Per-parameter modulation smoothing (learned time constants).
    p.add_argument("--param_smoothing_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Enable learned time-constant smoothing of modulation updates.")
    p.add_argument("--param_smoothing_tau_init", type=float, default=PARAM_SMOOTH_TAU_INIT_DEFAULT,
                   help="Initial per-step mixing fraction (0-1) for smoothing modulation outputs (e.g., 0.2 -> 20%% toward target per step).")
    p.add_argument("--param_smoothing_tau_min", type=float, default=PARAM_SMOOTH_TAU_MIN_DEFAULT,
                   help="Minimum per-step mixing fraction (keeps smoothing from becoming zero).")
    p.add_argument("--param_smoothing_trainable", type=str2bool, nargs="?", const=True, default=True,
                   help="Toggle requires_grad for smoothing time constants.")
    p.add_argument("--param_smoothing_modes", type=str, nargs="+",
                   default=None,
                   choices=["ann_sub", "ann_add", "ann_combo", "snn_add", "snn_sub", "mlp_sub", "mlp_add", "mlp_combo", "all"],
                   help="Optional subset of modes to smooth; omit to smooth all when enabled.")

    # Parameter range overrides (used for clamps and initialisation).
    p.add_argument("--alpha_1_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for alpha_1 (default ~[1/e,0.995]).")
    p.add_argument("--beta_1_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for beta_1 (default ~[1/e,0.995]).")
    p.add_argument("--thr_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for thresholds (default [0.5,1.5]).")
    p.add_argument("--reset_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for reset potentials (default [-0.5,0.5]).")
    p.add_argument("--rest_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for rest potentials (default [-0.5,0.5]).")
    p.add_argument("--alpha_2_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for alpha_2 (default ~[1/e,0.995]).")
    p.add_argument("--beta_2_range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="Allowed range for beta_2 (default ~[1/e,0.995]).")

    # --- SHAP analysis toggles ---
    p.add_argument("--ann_shap_enable", type=str2bool, nargs="?", const=True, default=False,
                   help="Approximate Shapley importance for ANN input/output blocks after mod/staged runs.")
    p.add_argument("--ann_shap_samples", type=int, default=16,
                   help="Number of random permutations for SHAP approximation (per input/output group).")
    p.add_argument("--ann_shap_batch_limit", type=int, default=4,
                   help="How many evaluation batches to cache for SHAP metric sweeps.")
    p.add_argument("--ann_shap_dataset", type=str, default="test", choices=["train", "test"],
                   help="Dataset split to score during SHAP analysis.")
    p.add_argument("--ann_shap_metric", type=str, default="acc", choices=["acc", "nll"],
                   help="Metric to explain (accuracy or NLL).")

    # --- base ckpt for --run_mode=mod ---
    p.add_argument("--base_snn_ckpt", type=str, default=None, help="Path to base SNN checkpoint for 'mod' mode")
    p.add_argument("--base_snn_from_stockpile", type=str2bool, nargs="?", const=True, default=False,
                   help="If True (mod mode), ignore --base_snn_ckpt and sample a base SNN from a stockpile directory.")
    p.add_argument("--base_snn_stockpile_dir", type=str, default=None,
                   help="Directory containing Run_* folders with base SNN checkpoints (used with --base_snn_from_stockpile).")
    p.add_argument("--base_snn_stockpile_seed", type=int, default=None,
                   help="Optional seed to make stockpile checkpoint selection reproducible (default: None = random).")

    p.add_argument("--save_every_epoch", type=str2bool, nargs="?", const=True, default=True,
                    help="Save checkpoints every epoch (True/False). Default=True.")

    p.add_argument("--homo_init", type=str2bool, nargs="?", const=True, default=False,
                   help="If True, initialize neuron params homogeneously (thr=1, reset=rest=0, taus from base).")

    # --- SNN parameter training toggles ---
    p.add_argument("--snn_train_disable", type=str, default="",
                   help="Comma-separated list of SNN params to freeze (options: w1,w2,v1,alpha_hetero_1,beta_hetero_1,"
                        "alpha_hetero_2,beta_hetero_2,thresholds_1,reset_1,rest_1,input_delay_logits).")
    
    args = p.parse_args()

    # Normalize save directories
    if args.save_dir_root is not None:
        root_str = str(args.save_dir_root).strip()
        if root_str and root_str.lower() != "none":
            root_path = Path(os.path.expanduser(root_str))
            args.save_dir_root = str(root_path)
            args.save_dir_snn = str(root_path / "SNN")
            args.save_dir_mod = str(root_path / "NM")
        else:
            args.save_dir_root = None

    shap_cfg = dict(
        enable=args.ann_shap_enable,
        samples=args.ann_shap_samples,
        batch_limit=args.ann_shap_batch_limit,
        dataset=args.ann_shap_dataset,
        metric=args.ann_shap_metric
    )

    # --- settings dict assembly (kept same names) ---
    # effective taus (paper scaling)
    tau_scale = args.paper_tau_scale if args.paper_tau_scale_enable else 1.0
    tau_syn_eff = args.tau_syn * tau_scale
    tau_mem_eff = args.tau_mem * tau_scale

    # channel compression
    nb_inputs_raw = args.nb_inputs
    compress_factor = 1
    nb_inputs_comp = nb_inputs_raw
    if args.channel_compress_enable:
        target = max(1, min(int(args.channel_compress_target), nb_inputs_raw))
        compress_factor = max(1, int(math.ceil(nb_inputs_raw / target)))
        nb_inputs_comp = int(math.ceil(nb_inputs_raw / compress_factor))
    compress_mode = str(args.channel_compress_mode).lower()
    if not args.channel_compress_enable:
        compress_mode = "none"
    nb_inputs_snn = nb_inputs_raw
    nb_inputs_mod = nb_inputs_raw
    compress_factor_snn = 1
    compress_factor_mod = 1
    if compress_mode == "all":
        nb_inputs_snn = nb_inputs_comp
        nb_inputs_mod = nb_inputs_comp
        compress_factor_snn = compress_factor
        compress_factor_mod = compress_factor
    elif compress_mode == "mod_only":
        nb_inputs_snn = nb_inputs_raw
        nb_inputs_mod = nb_inputs_comp
        compress_factor_snn = 1
        compress_factor_mod = compress_factor
    elif compress_mode == "mod_mlp":
        nb_inputs_snn = nb_inputs_raw
        nb_inputs_mod = nb_inputs_comp
        compress_factor_snn = 1
        compress_factor_mod = 1

    compress_mlp_hidden_sizes = parse_int_list(args.channel_compress_mlp_hidden_sizes)
    if compress_mlp_hidden_sizes is None:
        compress_mlp_hidden_sizes = []

    settings = dict(
        nb_inputs=nb_inputs_snn,
        nb_inputs_raw=nb_inputs_raw,
        nb_inputs_mod=nb_inputs_mod,
        channel_compress_enable=bool(args.channel_compress_enable),
        channel_compress_mode=compress_mode,
        channel_compress_target=int(args.channel_compress_target),
        channel_compress_factor=int(compress_factor),
        channel_compress_factor_snn=int(compress_factor_snn),
        channel_compress_factor_mod=int(compress_factor_mod),
        channel_compress_mlp_hidden_sizes=compress_mlp_hidden_sizes,

        nb_hidden=args.nb_hidden,
        nb_outputs=args.nb_outputs,

        time_step=args.time_step,
        nb_steps=args.nb_steps,
        max_time=args.max_time,

        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        cache_subdir=args.cache_subdir,
        train_file=args.train_file,
        test_file=args.test_file,

        lr=args.lr,
        nb_epochs=args.nb_epochs,
        eval_test_each_epoch=bool(args.eval_test_each_epoch),

        tau_syn=tau_syn_eff,
        tau_mem=tau_mem_eff,
        tau_match_clip=args.tau_match_clip,

        psp_norm_peak=args.psp_norm_peak,
        weight_scale=args.weight_scale,

        save_dir_snn=args.save_dir_snn,
        save_dir_mod=args.save_dir_mod,
        save_dir_root=args.save_dir_root,
        paper_tau_scale=tau_scale,
        paper_tau_scale_enable=bool(args.paper_tau_scale_enable),
        homo_init=bool(args.homo_init),
        hidden_dropout_p=float(args.hidden_dropout_p),
        use_input_delay=bool(args.use_input_delay),
        input_delay_steps=max(1, int(args.input_delay_steps)) if args.use_input_delay else 0,
        input_delay_init=args.input_delay_init,
        input_delay_init_cap=int(args.input_delay_init_cap) if args.input_delay_init_cap is not None else None,
        input_delay_init_bias=float(args.input_delay_init_bias),
        input_delay_init_noise=float(args.input_delay_init_noise),
        input_delay_temp=float(args.input_delay_temp),
    )

    disabled_params = set()
    if args.snn_train_disable:
        disabled_params = {p.strip() for p in args.snn_train_disable.split(",") if p.strip()}
    invalid = sorted(disabled_params - set(SNN_PARAM_NAMES))
    if invalid:
        raise ValueError(f"Unknown SNN parameters in --snn_train_disable: {', '.join(invalid)}")
    settings["snn_train_flags"] = {name: (name not in disabled_params) for name in SNN_PARAM_NAMES}

    # --- settings dict assembly for augmentation ---
    def _build_aug_cfg(enable_aug: bool, enable_noise: bool, paper_only: bool = False):
        compress_factor = settings.get("channel_compress_factor_snn", settings.get("channel_compress_factor", 1))
        if paper_only:
            if not (enable_aug or enable_noise or compress_factor > 1):
                return None
            eff_factor = max(1, compress_factor)
            channel_jitter_std = args.aug_channel_jitter_std if (enable_aug or enable_noise) else 0.0
            noise_rate_hz = args.aug_noise_rate_hz if enable_noise else 0.0
            return dict(
                jitter_ms=0.0, shift_ms=0.0, scale_low=1.0, scale_high=1.0,
                drop_p=0.0, insert_rate=0.0, band_frac=0.0,
                compress_factor=compress_factor,
                channel_jitter_std=channel_jitter_std,
                channel_jitter_std_eff=channel_jitter_std / float(eff_factor),
                noise_rate_hz=noise_rate_hz,
                noise_rate_hz_eff=noise_rate_hz * (eff_factor if args.aug_noise_per_input and enable_noise else 1.0),
                noise_per_input=args.aug_noise_per_input if enable_noise else False,
            )
        if not (enable_aug or enable_noise or compress_factor > 1):
            return None
        eff_factor = max(1, compress_factor)
        channel_jitter_std = args.aug_channel_jitter_std if enable_noise else 0.0
        noise_rate_hz = args.aug_noise_rate_hz if enable_noise else 0.0
        return dict(
            jitter_ms=args.aug_jitter_ms if enable_aug else 0.0,
            shift_ms=args.aug_shift_ms if enable_aug else 0.0,
            scale_low=args.aug_scale_low if enable_aug else 1.0,
            scale_high=args.aug_scale_high if enable_aug else 1.0,
            drop_p=args.aug_drop_p if enable_aug else 0.0,
            insert_rate=args.aug_insert_rate if enable_aug else 0.0,
            band_frac=args.aug_band_frac if enable_aug else 0.0,
            compress_factor=compress_factor,
            channel_jitter_std=channel_jitter_std,
            channel_jitter_std_eff=channel_jitter_std / float(eff_factor),
            noise_rate_hz=noise_rate_hz,
            noise_rate_hz_eff=noise_rate_hz * (eff_factor if args.aug_noise_per_input and enable_noise else 1.0),
            noise_per_input=args.aug_noise_per_input if enable_noise else False,
        )

    AUGMENT_TRAIN = _build_aug_cfg(args.train_aug_enable, args.train_noise_enable, paper_only=args.paper_aug_train)
    POSTBIN_TIME_MASK_TRAIN = args.postbin_time_mask_train if (args.train_aug_enable or args.train_noise_enable) else 0.0

    AUGMENT_EVAL = _build_aug_cfg(args.eval_aug_enable, args.eval_noise_enable, paper_only=args.paper_aug_eval)
    POSTBIN_TIME_MASK_EVAL = args.postbin_time_mask_eval

    AUGMENT_TEST = _build_aug_cfg(args.test_aug_enable, args.test_noise_enable, paper_only=args.paper_aug_test)
    POSTBIN_TIME_MASK_TEST = args.postbin_time_mask_test if (args.test_aug_enable or args.test_noise_enable) else 0.0

    settings.update(dict(
        augment_train=AUGMENT_TRAIN,
        postbin_mask_train=POSTBIN_TIME_MASK_TRAIN,
        augment_eval=AUGMENT_EVAL,
        postbin_mask_eval=POSTBIN_TIME_MASK_EVAL,
        augment_test=AUGMENT_TEST,
        postbin_mask_test=POSTBIN_TIME_MASK_TEST
    ))

    settings.update({
        "use_snn_reg": args.snn_reg_enable,
        "use_mod_reg": args.mod_reg_enable,
        "use_snn_mod_reg": args.snn_mod_reg_enable,
        "snn_reg_scale": args.snn_reg_scale,
        "mod_reg_scale": args.mod_reg_scale,
        "snn_mod_reg_scale": args.snn_mod_reg_scale,
    })


    # run mode
    run_mode = args.run_mode

    # MLP masks
    in_all  = ["alpha_1","beta_1","thr","reset","rest","alpha_2","beta_2","in_flat","hid_flat","out_flat"]
    out_all = ["alpha_1","beta_1","thr","reset","rest","alpha_2","beta_2"]
    mlp_in_mask  = _parse_mask(args.ann_in_disable, in_all)
    mlp_out_mask = _parse_mask(args.ann_out_disable, out_all)
    weight_targets = _normalize_weight_targets(args.weight_mod_targets)
    if args.weight_mod_enable:
        for name in weight_targets:
            mlp_in_mask[name] = True
            mlp_out_mask[name] = True
    mlp_hidden_sizes = parse_int_list(args.ann_hidden_sizes)
    if mlp_hidden_sizes is None:
        mlp_hidden_sizes = [2048]
    mod_hidden_sizes = parse_int_list(args.mod_hidden_sizes)
    if mod_hidden_sizes is None:
        mod_hidden_sizes = mlp_hidden_sizes
    mlp_interval = args.ann_interval
    mlp_mode = args.ann_mode
    mlp_arch = args.ann_arch
    ann_combo_additive = parse_str_list(args.ann_combo_additive)
    ann_combo_multiplicative = parse_str_list(args.ann_combo_multiplicative)
    ann_combo_additive, ann_combo_multiplicative = _resolve_ann_combo_lists(
        ann_combo_additive, ann_combo_multiplicative
    )

    ann_activation_raw = args.ann_output_activation
    ann_activation = None
    if isinstance(ann_activation_raw, str):
        ann_activation_raw = ann_activation_raw.strip()
        lowered = ann_activation_raw.lower()
        if ann_activation_raw and lowered not in {"none", "default"}:
            ann_activation = lowered

    normalized_mlp_mode = _normalize_mlp_mode(mlp_mode)
    resolved_activation = _final_activation_for_mode(normalized_mlp_mode, ann_activation)[1]

    # MLP config
    settings.update({
        "mlp_hidden_sizes": mlp_hidden_sizes,
        "mod_hidden_sizes": mod_hidden_sizes,
        "mlp_in_mask": mlp_in_mask,
        "mlp_out_mask": mlp_out_mask,
        "mlp_interval": mlp_interval,
        "ann_interval": mlp_interval,
        "mlp_mode": mlp_mode,
        "ann_mode": args.ann_mode,
        "mlp_arch": mlp_arch,
        "snn_mod_hidden": args.snn_mod_hidden,
        "snn_mod_gain_init": args.snn_mod_gain_init,
        "snn_mod_weight_scale": args.snn_mod_weight_scale,
        "snn_mod_hidden_recurrent": args.snn_mod_hidden_recurrent,
        "snn_mod_rec_init_zero": bool(args.snn_mod_rec_init_zero),
        "snn_add_balanced_init": bool(args.snn_add_balanced_init),
        "snn_add_init_effect_frac": float(args.snn_add_init_effect_frac),
        "ann_output_activation": ann_activation,
        "ann_output_activation_effective": resolved_activation,
        "ann_rnn_state_every_step": bool(args.ann_rnn_state_every_step),
        "mod_update_every_step": bool(args.mod_update_every_step),
        "mod_current_enable": bool(args.mod_current_enable),
        "mod_current_target": str(args.mod_current_target),
        "mod_current_activation": str(args.mod_current_activation),
        "ann_combo_additive": ann_combo_additive,
        "ann_combo_multiplicative": ann_combo_multiplicative,
    })
    settings["mlp_mode"] = normalized_mlp_mode

    # Parameter ranges
    param_ranges = {}
    for name in ("alpha_1", "beta_1", "thr", "reset", "rest", "alpha_2", "beta_2"):
        values = getattr(args, f"{name}_range")
        if values:
            param_ranges[name] = (float(values[0]), float(values[1]))
    if param_ranges:
        settings["param_ranges"] = param_ranges

    group_size = _parse_int_pair(args.group_size) or (1, 1)
    group_overlap = _parse_int_pair(args.group_overlap) or (0, 0)
    group_distribution = _parse_str_pair(args.group_distribution) or ("uniform", "uniform")
    group_normal_std = _parse_float_pair(args.group_normal_std) or (1.0, 1.0)
    hidden_size, output_size = group_size
    hidden_overlap, output_overlap = group_overlap
    group_enable = (hidden_size > 1 or hidden_overlap > 0,
                    output_size > 1 or output_overlap > 0)
    settings["group_cfg"] = {
        "enable": group_enable,
        "size": group_size,
        "overlap": group_overlap,
        "distribution": tuple(d.lower() for d in group_distribution),
        "normal_std": group_normal_std,
    }
    settings["weight_mod_cfg"] = {
        "enable": bool(args.weight_mod_enable),
        "targets": weight_targets if args.weight_mod_enable else [],
        "scope": str(args.weight_mod_scope).lower(),
        "grid_dim": str(args.weight_group_dim).lower(),
        "size": max(1, int(args.weight_group_size)),
        "overlap": max(0, int(args.weight_group_overlap)),
        "distribution": str(args.weight_group_distribution).lower(),
        "normal_std": float(args.weight_group_normal_std),
        "mode": str(args.weight_mod_mode).lower(),
        "delta_scale": float(args.weight_mod_delta_scale),
        "range_std_scale": float(args.weight_mod_range_std_scale),
    }

    nm_counts = parse_int_list(args.nm_counts)
    if not nm_counts:
        nm_counts = [0, 0]
    hidden_nm = int(nm_counts[0]) if len(nm_counts) > 0 else 0
    output_nm = int(nm_counts[1]) if len(nm_counts) > 1 else 0
    nm_activation_kind = None
    nm_mapper_activation = str(getattr(args, "nm_mapper_activation", "auto") or "auto").lower()
    if nm_mapper_activation == "auto":
        if normalized_mlp_mode in {"mlp_add", "mlp_combo", "snn_add"}:
            nm_activation_kind = "tanh"
        elif normalized_mlp_mode in {"mlp_sub", "snn_sub"}:
            nm_activation_kind = "sigmoid"
    else:
        if nm_mapper_activation not in {"none", "linear"}:
            nm_activation_kind = nm_mapper_activation
    nm_neuron_frac = parse_float_list(args.nm_neuron_frac) or [1.0, 1.0]
    if len(nm_neuron_frac) == 1:
        nm_neuron_frac = [nm_neuron_frac[0], nm_neuron_frac[0]]
    nm_param_frac_map = _parse_param_fraction_map(args.nm_param_frac, default=1.0)
    nm_cfg: Dict[str, object] = {
        "enable": args.nm_enable,
        "hidden_per_neuron": max(0, hidden_nm),
        "output_per_neuron": max(0, output_nm),
        "init_scale": args.nm_init_scale,
        "activation_kind": nm_activation_kind,
        "mapper_type": str(getattr(args, "nm_mapper_type", "mlp") or "mlp").lower(),
        "hidden_activation": str(getattr(args, "nm_mapper_hidden_activation", "silu") or "silu").lower(),
        "flat_order": str(getattr(args, "nm_flat_order", "type_major") or "type_major").lower(),
        "warm_start": bool(getattr(args, "nm_warm_start", False)),
        "warm_start_force_linear": bool(getattr(args, "nm_warm_start_force_linear", False)),
        "neuron_fraction_enable": args.nm_neuron_frac_enable,
        "neuron_fraction": (float(nm_neuron_frac[0]), float(nm_neuron_frac[1])) if len(nm_neuron_frac) >= 2 else (1.0, 1.0),
        "param_fraction_enable": args.nm_param_frac_enable,
        "param_fraction": nm_param_frac_map,
    }
    if nm_cfg.get("warm_start"):
        nm_cfg["init_identity"] = True
        # If counts match the parameter set sizes, a linear identity mapper is the closest analogue to normal mode.
        if bool(nm_cfg.get("warm_start_force_linear", True)) and hidden_nm == 5 and output_nm == 2:
            nm_cfg["mapper_type"] = "linear"
    nm_mapper_hidden_size = getattr(args, "nm_mapper_hidden_size", None)
    if nm_mapper_hidden_size is not None:
        nm_cfg["hidden_mlp_hidden"] = int(nm_mapper_hidden_size)
        nm_cfg["output_mlp_hidden"] = int(nm_mapper_hidden_size)
        nm_cfg["mlp_hidden"] = int(nm_mapper_hidden_size)
        nm_cfg["mapper_hidden_exact"] = True
    settings["nm_cfg"] = nm_cfg
    settings["nm_debug_print"] = bool(args.nm_debug_print)
    settings["mod_mask_cfg"] = {
        "fixed_enable": bool(args.mod_fixed_mask_enable),
        "fixed_seed": args.mod_fixed_mask_seed,
        "seed": args.seed,
        "fixed_flat_inputs": bool(args.mod_fixed_mask_flat_inputs),
        "zero_fallback": bool(args.mod_fixed_mask_zero_fallback),
    }
    settings["mod_hid_flat_group"] = bool(args.mod_hid_flat_group)
    settings["mod_hid_flat_modulated_only"] = bool(args.mod_hid_flat_modulated_only)

    settings["param_timescales"] = {
        "enable": args.param_timescales_enable,
        "distribution": args.param_timescales_distribution,
        "scale": args.param_timescales_scale,
        "std": args.param_timescales_std,
        "seed": args.param_timescales_seed,
        "trainable": args.param_timescales_trainable,
    }

    smoothing_modes_raw = args.param_smoothing_modes or []
    smoothing_modes: List[str] = []
    for mode in smoothing_modes_raw:
        lowered = str(mode).lower()
        if lowered == "all":
            smoothing_modes = ["mlp_sub", "mlp_add", "mlp_combo", "snn_add", "snn_sub"]
            break
        smoothing_modes.append(_normalize_mlp_mode(lowered))
    settings["param_smoothing"] = {
        "enable": args.param_smoothing_enable,
        "tau_init": args.param_smoothing_tau_init,
        "tau_min": args.param_smoothing_tau_min,
        "trainable": args.param_smoothing_trainable,
        "modes": smoothing_modes,
    }

    # split / training controls
    use_validation   = args.use_validation
    val_fraction     = args.val_fraction
    k_folds          = args.k_folds
    patience         = args.patience
    seed             = args.seed

    def _normalize_optional_path(value):
        if value is None:
            return None
        value_str = str(value).strip()
        if not value_str or value_str.lower() == "none":
            return None
        return value_str

    val_file = _normalize_optional_path(args.val_file)
    settings["val_file"] = val_file

    fixed_split_path = _normalize_optional_path(args.fixed_split_path)
    reinit_per_fold  = args.reinit_per_fold

    # staged controls
    staged_unlock_metric    = args.staged_unlock_metric
    staged_unlock_threshold = args.staged_unlock_threshold
    staged_min_epochs       = args.staged_min_epochs  # currently not used explicitly; reserved

    # mod-only controls
    start_locked   = args.start_locked
    unlock_metric  = args.unlock_metric
    unlock_threshold = args.unlock_threshold

    # base ckpt
    base_snn_ckpt = _normalize_optional_path(args.base_snn_ckpt)
    base_snn_stockpile_dir = _normalize_optional_path(args.base_snn_stockpile_dir)
    use_stockpile_base = bool(args.base_snn_from_stockpile)
    if use_stockpile_base and base_snn_ckpt is not None:
        print("[info] --base_snn_from_stockpile enabled; ignoring --base_snn_ckpt.")
        base_snn_ckpt = None
    if use_stockpile_base and run_mode != "mod":
        print("[info] --base_snn_from_stockpile applies only to run_mode='mod'; ignoring for this run.")
        use_stockpile_base = False
    snn_save_every_epoch = args.save_every_epoch
    mod_save_every_epoch = False
    if args.save_every_epoch and run_mode in ("mod", "staged"):
        print("[info] --save_every_epoch applies only to the SNN phase; "
              "mod checkpoints are saved only when they improve (mod_best.pth).")

    # determine global run index (align SNN and Mod directories)
    aligned_run_index = next_aligned_run_index(settings["save_dir_snn"], settings["save_dir_mod"])

    # choose save dir by mode
    if run_mode == "snn":
        settings["save_dir"] = settings["save_dir_snn"]
    elif run_mode in ("mod", "staged"):
        settings["save_dir"] = settings["save_dir_mod"]

    # load data
    x_train, y_train, x_test, y_test = load_h5(
        cache_dir=settings["cache_dir"], cache_subdir=settings["cache_subdir"],
        train_file=settings["train_file"], test_file=settings["test_file"]
    )
    x_val, y_val = (None, None)
    if val_file:
        x_val, y_val = load_validation_split(
            cache_dir=settings["cache_dir"], cache_subdir=settings["cache_subdir"], val_file=val_file
        )

    # build base state
    state = setup_model(settings)

    # === Execute ===
    if run_mode == "snn":
        print(format_param_stats(state, None, prefix="Run mode=snn:"))
        # use settings["nb_epochs"] directly
        result = train_snn_hetero(
            state, settings, x_train, y_train, x_test, y_test, x_val=x_val, y_val=y_val,
            save_every_epoch=snn_save_every_epoch,
            use_validation=args.use_validation, val_fraction=args.val_fraction,
            k_folds=args.k_folds, patience=args.patience,
            eval_test_each_epoch=args.eval_test_each_epoch, seed=args.seed,
            fixed_split_path=fixed_split_path, reinit_per_fold=args.reinit_per_fold,
            chosen_metric=args.staged_unlock_metric,
            chosen_threshold=args.staged_unlock_threshold,
            run_index=aligned_run_index,
        )

        print("\nSNN training complete.")
        print(result)

    elif run_mode == "mod":
        # override epochs for mod if provided
        settings_mod = dict(settings)
        settings_mod["nb_epochs"] = args.nb_epochs_mod if args.nb_epochs_mod is not None else args.nb_epochs
        stockpile_ckpt = None
        stockpile_split_path = None
        if use_stockpile_base:
            stockpile_source = base_snn_stockpile_dir or "SNN_Stockpile/Base_SNN/SNN"
            stockpile_ckpt = pick_random_stockpile_ckpt(stockpile_source, seed=args.base_snn_stockpile_seed)
            candidate_split = stockpile_ckpt.parent / "split_indices.npz"
            if candidate_split.exists():
                stockpile_split_path = str(candidate_split)
                print(f"[info] Using stockpile split indices from {candidate_split}")
            else:
                print(f"[warn] Stockpile split indices not found next to {stockpile_ckpt}; using default split settings.")
        effective_base_ckpt = stockpile_ckpt or base_snn_ckpt
        effective_fixed_split = stockpile_split_path or fixed_split_path
        use_validation_mod = args.use_validation or (stockpile_split_path is not None)
        if (stockpile_split_path is not None) and (not args.use_validation):
            print("[info] Enabling validation to reuse stockpile split indices.")
        if effective_base_ckpt is not None:
            state_base = load_base_snn_state(effective_base_ckpt, settings_mod.get("snn_train_flags"), settings=settings_mod)
        else:
            state_base = setup_model(settings_mod)

        mod_mode = _select_mlp_mode(settings_mod)
        mlp_hidden_sizes_mod = _resolve_mlp_hidden_sizes(settings_mod)
        snn_hidden_sizes_mod = settings_mod.get("mod_hidden_sizes")
        hidden_cfg = snn_hidden_sizes_mod if _is_snn_mode(mod_mode) else mlp_hidden_sizes_mod
        modulator = build_modulator(
            settings_mod,
            override_mode=mod_mode,
            hidden_sizes=hidden_cfg,
        )
        print(format_param_stats(state_base, modulator, prefix="Run mode=mod:", settings=settings_mod))
        result = train_modulated_snn(
            state_base, settings_mod, x_train, y_train, x_test, y_test, x_val=x_val, y_val=y_val,
            modulator=modulator, mlp_interval=settings_mod["mlp_interval"],
            save_every_epoch=mod_save_every_epoch, use_validation=use_validation_mod, val_fraction=args.val_fraction,
            k_folds=args.k_folds, patience=args.patience,
            eval_test_each_epoch=args.eval_test_each_epoch, seed=args.seed,
            fixed_split_path=effective_fixed_split, reinit_per_fold=args.reinit_per_fold,
            start_locked=args.start_locked, unlock_metric=args.unlock_metric, unlock_threshold=args.unlock_threshold,
            base_ckpt_path=effective_base_ckpt,  # <— NEW
            run_index=aligned_run_index,
        )


        print("\nModulated training complete.")
        print(result)
        maybe_run_ann_shap(result, settings_mod, shap_cfg, x_train, y_train, x_test, y_test)

    elif run_mode == "staged":
        # ---- Stage 1: SNN ----
        settings_stage1 = dict(settings)
        settings_stage1["save_dir"] = settings["save_dir_snn"]
        settings_stage1["nb_epochs"] = args.nb_epochs_snn

        print(format_param_stats(state, None, prefix="Stage-1 snn:"))
        result = train_snn_hetero(
            state, settings_stage1, x_train, y_train, x_test, y_test, x_val=x_val, y_val=y_val,
            save_every_epoch=snn_save_every_epoch, use_validation=args.use_validation, val_fraction=args.val_fraction,
            k_folds=args.k_folds, patience=args.patience,
            eval_test_each_epoch=args.eval_test_each_epoch, seed=args.seed,
            fixed_split_path=fixed_split_path, reinit_per_fold=args.reinit_per_fold,
            chosen_metric=args.staged_unlock_metric,
            chosen_threshold=args.staged_unlock_threshold,
            run_index=aligned_run_index,
        )

        print("\nStage-1 (SNN) complete.")
        print(result)

        stage_run_index = _parse_run_index(Path(result["run_dir"]).name)
        if stage_run_index is None:
            stage_run_index = aligned_run_index

        # pick 'chosen' for Stage-2; fallback to snn_best.pth if needed
        fold1_dir = Path(result["run_dir"]) / "Fold_1"
        chosen_candidates = [fold1_dir / "chosen.pth", fold1_dir / "chosen"]
        chosen_ckpt = next((p for p in chosen_candidates if p.exists()), None)
        if chosen_ckpt is None:
            fallback_candidates = [fold1_dir / "snn_best.pth", fold1_dir / "best_snn.pth"]
            for cand in fallback_candidates:
                if cand.exists():
                    chosen_ckpt = cand
                    print(f"⚠️ 'chosen' not found; using fallback checkpoint {cand}")
                    break
        if chosen_ckpt is None:
            raise FileNotFoundError(f"No base SNN checkpoint found in {fold1_dir}")
        print(f"[Stage-2] Using base SNN checkpoint: {chosen_ckpt}")

        # ---- Stage 2: Modulated ----
        settings_mod = dict(settings)
        settings_mod["nb_epochs"] = args.nb_epochs_mod
        state_base = load_base_snn_state(chosen_ckpt, settings_mod.get("snn_train_flags"), settings=settings_mod)
        mod_mode = _select_mlp_mode(settings_mod)
        mlp_hidden_sizes_stage2 = _resolve_mlp_hidden_sizes(settings_mod)
        snn_hidden_sizes_stage2 = settings_mod.get("mod_hidden_sizes")
        hidden_cfg = snn_hidden_sizes_stage2 if _is_snn_mode(mod_mode) else mlp_hidden_sizes_stage2
        modulator = build_modulator(
            settings_mod,
            override_mode=mod_mode,
            hidden_sizes=hidden_cfg,
        )
        print(format_param_stats(state_base, modulator, prefix="Stage-2 mod:", settings=settings_mod))

        result_mod = train_modulated_snn(
            state_base, settings_mod, x_train, y_train, x_test, y_test, x_val=x_val, y_val=y_val,
            modulator=modulator, mlp_interval=settings_mod["mlp_interval"],
            save_every_epoch=mod_save_every_epoch, use_validation=args.use_validation, val_fraction=args.val_fraction,
            k_folds=args.k_folds, patience=args.patience,
            eval_test_each_epoch=args.eval_test_each_epoch, seed=args.seed,
            fixed_split_path=fixed_split_path, reinit_per_fold=args.reinit_per_fold,
            start_locked=args.start_locked,                     
            unlock_metric=args.unlock_metric,                    
            unlock_threshold=args.unlock_threshold,              
            base_ckpt_path=chosen_ckpt,
            run_index=stage_run_index,
        )

        print("\nStage-2 (Modulated) complete.")
        print(result_mod)
        maybe_run_ann_shap(result_mod, settings_mod, shap_cfg, x_train, y_train, x_test, y_test)
    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")

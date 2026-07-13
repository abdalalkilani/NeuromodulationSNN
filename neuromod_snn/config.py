import ast
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_PARAM_NAMES = ["alpha_1", "beta_1", "thr", "reset", "rest"]
OUTPUT_PARAM_NAMES = ["alpha_2", "beta_2"]
MOD_PARAM_NAMES = HIDDEN_PARAM_NAMES + OUTPUT_PARAM_NAMES

SNN_PARAM_NAMES = [
    "w1",
    "v1",
    "w2",
    "alpha_1",
    "beta_1",
    "thr",
    "reset",
    "rest",
    "alpha_2",
    "beta_2",
]

PARAMETER_RANGES = {
    "alpha_1": (1.0 / math.e, 0.995),
    "beta_1": (1.0 / math.e, 0.995),
    "alpha_2": (1.0 / math.e, 0.995),
    "beta_2": (1.0 / math.e, 0.995),
    "thr": (0.5, 1.5),
    "reset": (-0.5, 0.5),
    "rest": (-0.5, 0.5),
}

DEFAULT_INPUT_BLOCKS = {
    "alpha_1": True,
    "beta_1": True,
    "thr": True,
    "reset": True,
    "rest": True,
    "alpha_2": True,
    "beta_2": True,
    "in_flat": True,
    "hid_flat": True,
    "out_flat": True,
}

DEFAULT_OUTPUT_BLOCKS = {name: True for name in MOD_PARAM_NAMES}


@dataclass
class ExperimentConfig:
    nb_inputs: int = 700
    nb_hidden: int = 256
    nb_outputs: int = 20
    nb_steps: int = 100
    max_time: float = 1.4
    time_step: float = 1e-3
    tau_syn: float = 10e-3
    tau_mem: float = 20e-3
    tau_match_clip: bool = False
    weight_scale: float = 0.2
    batch_size: int = 64
    lr: float = 2e-4
    nb_epochs: int = 30
    nb_epochs_snn: int = 30
    nb_epochs_mod: int = 70
    cache_dir: str = "~/data"
    cache_subdir: str = "hdspikes"
    train_file: str = "shd_train.h5"
    test_file: str = "shd_test.h5"
    save_dir: str = "runs"
    seed: int = 123
    ann_mode: str = "ann_sub"
    ann_hidden_sizes: List[int] = field(default_factory=lambda: [2048])
    ann_interval: int = 3
    ann_in_disable: str = ""
    ann_out_disable: str = ""
    group_size: Tuple[int, int] = (1, 1)
    channel_compress_enable: bool = False
    channel_compress_target: int = 70
    channel_compress_mode: str = "mod_only"
    nm_enable: bool = False
    nm_counts: Tuple[int, int] = (0, 0)
    nm_mapper_hidden_size: Optional[int] = None
    param_smoothing_enable: bool = False
    param_smoothing_tau: float = 1.0
    snn_reg_enable: bool = True
    snn_reg_scale: float = 1.0
    train_aug_enable: bool = False
    aug_channel_jitter_std: float = 20.0
    train_noise_enable: bool = False
    aug_noise_rate_hz: float = 0.0
    hidden_dropout_p: float = 0.0
    base_snn_ckpt: Optional[str] = None

    @property
    def train_path(self) -> Path:
        return resolve_h5_path(self.cache_dir, self.cache_subdir, self.train_file)

    @property
    def test_path(self) -> Path:
        return resolve_h5_path(self.cache_dir, self.cache_subdir, self.test_file)

    @property
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        return dict(PARAMETER_RANGES)


def resolve_h5_path(cache_dir: str, cache_subdir: str, file_name: str) -> Path:
    path = Path(file_name).expanduser()
    if path.is_absolute():
        return path
    base = Path(cache_dir).expanduser()
    if cache_subdir:
        base = base / cache_subdir
    return base / file_name


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def parse_int_list(value, default=None) -> List[int]:
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return list(default or [])
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [int(v) for v in parsed]
    except (SyntaxError, ValueError):
        pass
    return [int(tok) for tok in text.replace(",", " ").split()]


def parse_pair(value, default=(1, 1)) -> Tuple[int, int]:
    vals = parse_int_list(value, default=default)
    if not vals:
        return default
    if len(vals) == 1:
        return int(vals[0]), int(vals[0])
    return int(vals[0]), int(vals[1])


def parse_mask(disabled_csv: str, allowed: Dict[str, bool]) -> Dict[str, bool]:
    mask = dict(allowed)
    disabled = [item.strip() for item in str(disabled_csv or "").split(",") if item.strip()]
    for name in disabled:
        if name not in mask:
            raise ValueError(f"Unknown block {name!r}. Valid blocks: {sorted(mask)}")
        mask[name] = False
    return mask


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

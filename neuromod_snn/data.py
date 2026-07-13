import math
from pathlib import Path
from typing import Iterator, Optional, Tuple

import h5py
import numpy as np
import torch

from .config import ExperimentConfig, device, dtype


def open_h5_pair(cfg: ExperimentConfig):
    train_path = cfg.train_path
    test_path = cfg.test_path
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    train_h5 = h5py.File(train_path, "r")
    test_h5 = h5py.File(test_path, "r")
    return train_h5["spikes"], train_h5["labels"], test_h5["spikes"], test_h5["labels"]


def compress_dense_inputs(inputs: torch.Tensor, factor: int, nb_units_new: Optional[int] = None) -> torch.Tensor:
    if factor <= 1:
        return inputs
    batch, steps, channels = inputs.shape
    nb_units_new = int(nb_units_new or int(np.ceil(channels / factor)))
    padded_channels = nb_units_new * factor
    if padded_channels < channels:
        nb_units_new = int(math.ceil(channels / factor))
        padded_channels = nb_units_new * factor
    if padded_channels > channels:
        pad = inputs.new_zeros((batch, steps, padded_channels - channels))
        inputs = torch.cat([inputs, pad], dim=2)
    return inputs.reshape(batch, steps, nb_units_new, factor).sum(dim=3)


def channel_jitter(units: np.ndarray, nb_units: int, sigma_units: float) -> np.ndarray:
    if sigma_units <= 0:
        return units
    jitter = np.random.normal(0.0, sigma_units, size=units.shape)
    return np.clip(np.rint(units + jitter), 0, nb_units - 1).astype(np.int64)


def inject_poisson_noise(times: np.ndarray, units: np.ndarray, nb_units: int, rate_hz: float, max_time: float):
    if rate_hz <= 0:
        return times, units
    count = np.random.poisson(rate_hz * max_time)
    if count <= 0:
        return times, units
    noise_times = np.random.uniform(0.0, max_time, size=count)
    noise_units = np.random.randint(0, nb_units, size=count)
    return np.concatenate([times, noise_times]), np.concatenate([units, noise_units])


def dense_batches_from_hdf5(
    spikes,
    labels,
    cfg: ExperimentConfig,
    *,
    shuffle: bool,
    max_samples: Optional[int] = None,
    augment: bool = False,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    labels_np = np.asarray(labels, dtype=np.int64)
    sample_index = np.arange(len(labels_np))
    if max_samples is not None:
        sample_index = sample_index[: min(int(max_samples), len(sample_index))]
    if shuffle:
        np.random.shuffle(sample_index)

    firing_times = spikes["times"]
    units_fired = spikes["units"]
    time_bins = np.linspace(0.0, cfg.max_time, num=cfg.nb_steps + 1)

    for start in range(0, len(sample_index), cfg.batch_size):
        batch_index = sample_index[start : start + cfg.batch_size]
        if len(batch_index) == 0:
            continue
        dense = torch.zeros((len(batch_index), cfg.nb_steps, cfg.nb_inputs), dtype=dtype)
        target = torch.tensor(labels_np[batch_index], dtype=torch.long)
        for b, idx in enumerate(batch_index):
            times = np.asarray(firing_times[idx], dtype=np.float64)
            units = np.asarray(units_fired[idx], dtype=np.int64)
            if augment and cfg.train_aug_enable:
                units = channel_jitter(units, cfg.nb_inputs, cfg.aug_channel_jitter_std)
            if augment and cfg.train_noise_enable:
                times, units = inject_poisson_noise(times, units, cfg.nb_inputs, cfg.aug_noise_rate_hz, cfg.max_time)
            bins = np.digitize(times, time_bins) - 1
            bins = np.clip(bins, 0, cfg.nb_steps - 1).astype(np.int64)
            keep = (units >= 0) & (units < cfg.nb_inputs)
            dense[b, bins[keep], units[keep]] = 1.0
        yield dense.to(device), target.to(device)

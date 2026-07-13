import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig, MOD_PARAM_NAMES, device, dtype
from .data import compress_dense_inputs, dense_batches_from_hdf5
from .modulation import ANNModulator, SNNAdditiveModulator, apply_additive, map_substitution
from .snn import (
    clamp_state_,
    clone_state,
    load_snn_checkpoint,
    spike_fn,
    spike_regularizer,
    trainable_state_params,
)


def _recent_sum(records, start, end, fallback):
    if not records:
        return fallback
    return torch.stack(records[start:end], dim=1).sum(dim=1)


def _smooth(old, target, cfg: ExperimentConfig):
    if not cfg.param_smoothing_enable:
        return target
    lam = float(cfg.param_smoothing_tau)
    return old + lam * (target - old)


def _apply_effects(params, effects: Dict[str, torch.Tensor], cfg: ExperimentConfig) -> Dict[str, torch.Tensor]:
    updated = dict(params)
    for name, effect in effects.items():
        if cfg.ann_mode == "ann_sub":
            target = map_substitution(effect, name, cfg)
        else:
            target = apply_additive(params[name], effect, name, cfg)
        updated[name] = _smooth(params[name], target, cfg)
    return updated


def run_modulated_snn(inputs: torch.Tensor, state: Dict[str, torch.Tensor], modulator, cfg: ExperimentConfig, return_records: bool = False):
    batch = inputs.size(0)
    x_raw = inputs.to(device=device, dtype=dtype)
    x_snn = x_raw
    if cfg.channel_compress_enable and cfg.channel_compress_mode == "mod_only":
        target = min(cfg.nb_inputs, cfg.channel_compress_target)
        factor = int(math.ceil(cfg.nb_inputs / max(1, target)))
        x_mod = compress_dense_inputs(x_raw, factor, target)
    else:
        x_mod = x_raw

    w1, v1, w2 = state["w1"], state["v1"], state["w2"]
    params = {name: state[name].expand(batch, -1).clone() for name in MOD_PARAM_NAMES}

    syn = torch.zeros((batch, cfg.nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros_like(syn)
    spk = torch.zeros_like(syn)
    flt = torch.zeros((batch, cfg.nb_outputs), device=device, dtype=dtype)
    out = torch.zeros_like(flt)
    spk_rec, out_rec, mod_feature_rec = [], [], []
    h1_from_input = torch.einsum("btc,ch->bth", x_snn, w1)

    for step in range(cfg.nb_steps):
        h1 = h1_from_input[:, step] + torch.einsum("bh,hk->bk", spk, v1)
        mthr = mem - params["thr"]
        spk = spike_fn(mthr)
        rst = (mthr > 0).to(dtype)
        syn = params["alpha_1"] * syn + h1
        mem = params["beta_1"] * (mem - params["rest"]) + params["rest"] + (1.0 - params["beta_1"]) * syn - rst * (params["thr"] - params["reset"])
        h2 = torch.einsum("bh,ho->bo", spk, w2)
        flt = params["alpha_2"] * flt + h2
        out = params["beta_2"] * out + (1.0 - params["beta_2"]) * flt

        spk_rec.append(spk)
        out_rec.append(out)
        interval = max(1, int(cfg.ann_interval))
        start = max(0, step - interval + 1)
        in_flat = x_mod[:, start : step + 1].sum(dim=1)
        hid_flat = _recent_sum(spk_rec, start, step + 1, spk)
        out_flat = _recent_sum(out_rec, start, step + 1, out)

        if isinstance(modulator, SNNAdditiveModulator):
            mod_feature_rec.append(modulator.make_input(params, in_flat, hid_flat, out_flat))
            if (step + 1) % interval == 0:
                features = torch.stack(mod_feature_rec[-interval:], dim=1)
                params = _apply_effects(params, modulator(features), cfg)
        elif isinstance(modulator, ANNModulator):
            if step % interval == 0:
                params = _apply_effects(params, modulator(params, in_flat, hid_flat, out_flat), cfg)
        else:
            raise TypeError("Unsupported modulator type")

    out_rec_tensor = torch.stack(out_rec, dim=1)
    logits = out_rec_tensor.max(dim=1).values
    if return_records:
        return logits, {"hidden_spikes": torch.stack(spk_rec, dim=1), "outputs": out_rec_tensor}
    return logits


@torch.no_grad()
def evaluate_modulated(state, cfg: ExperimentConfig, x_data, y_data, modulator, max_samples: Optional[int] = None) -> Tuple[float, float]:
    losses, accs, counts = [], [], []
    modulator.eval()
    for inputs, target in dense_batches_from_hdf5(x_data, y_data, cfg, shuffle=False, max_samples=max_samples):
        logits = run_modulated_snn(inputs, state, modulator, cfg)
        loss = F.cross_entropy(logits, target)
        acc = (logits.argmax(dim=1) == target).float().mean()
        losses.append(float(loss.item()) * len(target))
        accs.append(float(acc.item()) * len(target))
        counts.append(len(target))
    modulator.train()
    total = max(1, sum(counts))
    return sum(losses) / total, sum(accs) / total


def save_mod_checkpoint(path: Path, state, modulator, cfg: ExperimentConfig, epoch: int, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "snn_state": {name: value.detach().cpu() for name, value in state.items()},
            "modulator_state": modulator.state_dict(),
            "config": cfg.__dict__,
        },
        path,
    )


def train_modulated(cfg: ExperimentConfig, x_train, y_train, x_test=None, y_test=None, base_state=None, modulator=None):
    if base_state is None:
        if not cfg.base_snn_ckpt:
            raise ValueError("modulated training needs base_state or cfg.base_snn_ckpt")
        base_state = load_snn_checkpoint(cfg.base_snn_ckpt, requires_grad=True)
    state = clone_state(base_state, requires_grad=True)
    if modulator is None:
        from .modulation import build_modulator

        modulator = build_modulator(cfg)
    optimizer = torch.optim.Adam(list(trainable_state_params(state)) + list(modulator.parameters()), lr=cfg.lr)
    run_dir = Path(cfg.save_dir)
    best_acc = -1.0
    for epoch in range(1, cfg.nb_epochs_mod + 1):
        losses = []
        for inputs, target in dense_batches_from_hdf5(x_train, y_train, cfg, shuffle=True, augment=True):
            logits, rec = run_modulated_snn(inputs, state, modulator, cfg, return_records=True)
            loss = F.cross_entropy(logits, target) + spike_regularizer(rec["hidden_spikes"], cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            clamp_state_(state, cfg)
            losses.append(float(loss.item()))
        metrics = {"train_loss": float(np.mean(losses)) if losses else float("nan")}
        if x_test is not None and y_test is not None:
            test_loss, test_acc = evaluate_modulated(state, cfg, x_test, y_test, modulator)
            metrics.update(test_loss=test_loss, test_acc=test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                save_mod_checkpoint(run_dir / "mod_best.pt", state, modulator, cfg, epoch, metrics)
        print(f"[mod] epoch={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    save_mod_checkpoint(run_dir / "mod_last.pt", state, modulator, cfg, cfg.nb_epochs_mod, metrics)
    return state, modulator

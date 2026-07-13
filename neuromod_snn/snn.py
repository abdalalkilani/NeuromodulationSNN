import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig, MOD_PARAM_NAMES, SNN_PARAM_NAMES, device, dtype
from .data import dense_batches_from_hdf5


class SurrogateSpike(torch.autograd.Function):
    scale = 100.0

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output / (SurrogateSpike.scale * x.abs() + 1.0).pow(2)


spike_fn = SurrogateSpike.apply


def effective_decay_dt(cfg: ExperimentConfig) -> float:
    if cfg.tau_match_clip and cfg.max_time > 0:
        return float(cfg.max_time) / max(1, int(cfg.nb_steps))
    return float(cfg.time_step)


def setup_model(cfg: ExperimentConfig) -> Dict[str, nn.Parameter]:
    dt = effective_decay_dt(cfg)
    alpha = math.exp(-dt / cfg.tau_syn)
    beta = math.exp(-dt / cfg.tau_mem)
    state = {
        "w1": nn.Parameter(torch.randn(cfg.nb_inputs, cfg.nb_hidden, device=device, dtype=dtype) * cfg.weight_scale / math.sqrt(cfg.nb_inputs)),
        "v1": nn.Parameter(torch.randn(cfg.nb_hidden, cfg.nb_hidden, device=device, dtype=dtype) * cfg.weight_scale / math.sqrt(cfg.nb_hidden)),
        "w2": nn.Parameter(torch.randn(cfg.nb_hidden, cfg.nb_outputs, device=device, dtype=dtype) * cfg.weight_scale / math.sqrt(cfg.nb_hidden)),
        "alpha_1": nn.Parameter(torch.full((1, cfg.nb_hidden), alpha, device=device, dtype=dtype)),
        "beta_1": nn.Parameter(torch.full((1, cfg.nb_hidden), beta, device=device, dtype=dtype)),
        "thr": nn.Parameter(torch.ones(1, cfg.nb_hidden, device=device, dtype=dtype)),
        "reset": nn.Parameter(torch.zeros(1, cfg.nb_hidden, device=device, dtype=dtype)),
        "rest": nn.Parameter(torch.zeros(1, cfg.nb_hidden, device=device, dtype=dtype)),
        "alpha_2": nn.Parameter(torch.full((1, cfg.nb_outputs), alpha, device=device, dtype=dtype)),
        "beta_2": nn.Parameter(torch.full((1, cfg.nb_outputs), beta, device=device, dtype=dtype)),
    }
    return state


def trainable_state_params(state: Dict[str, nn.Parameter]) -> Iterable[nn.Parameter]:
    return [p for p in state.values() if p.requires_grad]


def clone_state(state: Dict[str, torch.Tensor], requires_grad: bool = True) -> Dict[str, nn.Parameter]:
    return {name: nn.Parameter(value.detach().clone().to(device), requires_grad=requires_grad) for name, value in state.items()}


def clamp_state_(state: Dict[str, torch.Tensor], cfg: ExperimentConfig):
    with torch.no_grad():
        for name in MOD_PARAM_NAMES:
            lo, hi = cfg.param_ranges[name]
            state[name].clamp_(float(lo), float(hi))


def run_snn(inputs: torch.Tensor, state: Dict[str, torch.Tensor], cfg: ExperimentConfig, return_records: bool = False):
    batch = inputs.size(0)
    x = inputs.to(device=device, dtype=dtype)
    w1, v1, w2 = state["w1"], state["v1"], state["w2"]
    alpha_1 = state["alpha_1"].expand(batch, cfg.nb_hidden)
    beta_1 = state["beta_1"].expand(batch, cfg.nb_hidden)
    thr = state["thr"].expand(batch, cfg.nb_hidden)
    reset = state["reset"].expand(batch, cfg.nb_hidden)
    rest = state["rest"].expand(batch, cfg.nb_hidden)
    alpha_2 = state["alpha_2"].expand(batch, cfg.nb_outputs)
    beta_2 = state["beta_2"].expand(batch, cfg.nb_outputs)

    syn = torch.zeros((batch, cfg.nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros_like(syn)
    spk = torch.zeros_like(syn)
    flt = torch.zeros((batch, cfg.nb_outputs), device=device, dtype=dtype)
    out = torch.zeros_like(flt)
    spk_rec, out_rec = [], []
    h1_from_input = torch.einsum("btc,ch->bth", x, w1)

    for step in range(cfg.nb_steps):
        h1 = h1_from_input[:, step] + torch.einsum("bh,hk->bk", spk, v1)
        mthr = mem - thr
        spk = spike_fn(mthr)
        if cfg.hidden_dropout_p > 0:
            spk = F.dropout(spk, p=cfg.hidden_dropout_p, training=True)
        rst = (mthr > 0).to(dtype)
        syn = alpha_1 * syn + h1
        mem = beta_1 * (mem - rest) + rest + (1.0 - beta_1) * syn - rst * (thr - reset)
        h2 = torch.einsum("bh,ho->bo", spk, w2)
        flt = alpha_2 * flt + h2
        out = beta_2 * out + (1.0 - beta_2) * flt
        spk_rec.append(spk)
        out_rec.append(out)

    out_rec = torch.stack(out_rec, dim=1)
    logits = out_rec.max(dim=1).values
    if return_records:
        return logits, {"hidden_spikes": torch.stack(spk_rec, dim=1), "outputs": out_rec}
    return logits


def spike_regularizer(spikes: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    if not cfg.snn_reg_enable:
        return spikes.new_tensor(0.0)
    low = torch.clamp(spikes.mean(dim=1) - 0.01, min=0.0).pow(2).sum()
    high = torch.clamp(spikes.sum(dim=(1, 2)) / max(1, cfg.nb_hidden) - 100.0, min=0.0).pow(2).sum()
    return cfg.snn_reg_scale * (low / max(1, spikes.numel()) + 0.06 * high / max(1, spikes.size(0)))


@torch.no_grad()
def evaluate_snn(state, cfg: ExperimentConfig, x_data, y_data, max_samples: Optional[int] = None) -> Tuple[float, float]:
    losses, accs, counts = [], [], []
    for inputs, target in dense_batches_from_hdf5(x_data, y_data, cfg, shuffle=False, max_samples=max_samples):
        logits = run_snn(inputs, state, cfg)
        loss = F.cross_entropy(logits, target)
        acc = (logits.argmax(dim=1) == target).float().mean()
        losses.append(float(loss.item()) * len(target))
        accs.append(float(acc.item()) * len(target))
        counts.append(len(target))
    total = max(1, sum(counts))
    return sum(losses) / total, sum(accs) / total


def save_snn_checkpoint(path: Path, state, cfg: ExperimentConfig, epoch: int, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "state": {name: value.detach().cpu() for name, value in state.items()},
            "config": cfg.__dict__,
        },
        path,
    )


def load_snn_checkpoint(path: str, requires_grad: bool = True) -> Dict[str, nn.Parameter]:
    ckpt = torch.load(path, map_location=device)
    params = ckpt.get("state", ckpt.get("params", ckpt))
    aliases = {
        "alpha_hetero_1": "alpha_1",
        "beta_hetero_1": "beta_1",
        "thresholds_1": "thr",
        "reset_1": "reset",
        "rest_1": "rest",
        "alpha_hetero_2": "alpha_2",
        "beta_hetero_2": "beta_2",
    }
    out = {}
    for name, value in params.items():
        key = aliases.get(name, name)
        if key in SNN_PARAM_NAMES:
            out[key] = nn.Parameter(value.to(device), requires_grad=requires_grad)
    missing = [name for name in SNN_PARAM_NAMES if name not in out]
    if missing:
        raise KeyError(f"Checkpoint is missing SNN parameters: {missing}")
    return out


def train_snn(cfg: ExperimentConfig, x_train, y_train, x_test=None, y_test=None):
    state = setup_model(cfg)
    optimizer = torch.optim.Adam(trainable_state_params(state), lr=cfg.lr)
    run_dir = Path(cfg.save_dir)
    best_acc = -1.0
    for epoch in range(1, cfg.nb_epochs + 1):
        losses = []
        for inputs, target in dense_batches_from_hdf5(x_train, y_train, cfg, shuffle=True, augment=True):
            logits, rec = run_snn(inputs, state, cfg, return_records=True)
            loss = F.cross_entropy(logits, target) + spike_regularizer(rec["hidden_spikes"], cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            clamp_state_(state, cfg)
            losses.append(float(loss.item()))
        metrics = {"train_loss": float(np.mean(losses)) if losses else float("nan")}
        if x_test is not None and y_test is not None:
            test_loss, test_acc = evaluate_snn(state, cfg, x_test, y_test)
            metrics.update(test_loss=test_loss, test_acc=test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                save_snn_checkpoint(run_dir / "snn_best.pt", state, cfg, epoch, metrics)
        print(f"[snn] epoch={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    save_snn_checkpoint(run_dir / "snn_last.pt", state, cfg, cfg.nb_epochs, metrics)
    return state

import argparse

from .config import ExperimentConfig, parse_int_list, parse_pair, set_seed, str2bool
from .data import open_h5_pair
from .modulated import train_modulated
from .modulation import build_modulator
from .snn import train_snn


def build_parser():
    p = argparse.ArgumentParser(description="Paper-facing neuromodulated SNN runner")
    p.add_argument("--run_mode", choices=["snn", "mod", "staged"], default="snn")
    p.add_argument("--nb_inputs", type=int, default=700)
    p.add_argument("--nb_hidden", type=int, default=256)
    p.add_argument("--nb_outputs", type=int, default=20)
    p.add_argument("--nb_steps", type=int, default=100)
    p.add_argument("--max_time", type=float, default=1.4)
    p.add_argument("--time_step", type=float, default=1e-3)
    p.add_argument("--tau_syn", type=float, default=10e-3)
    p.add_argument("--tau_mem", type=float, default=20e-3)
    p.add_argument("--tau_match_clip", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--weight_scale", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--nb_epochs", type=int, default=30)
    p.add_argument("--nb_epochs_snn", type=int, default=30)
    p.add_argument("--nb_epochs_mod", type=int, default=70)
    p.add_argument("--cache_dir", type=str, default="~/data")
    p.add_argument("--cache_subdir", type=str, default="hdspikes")
    p.add_argument("--train_file", type=str, default="shd_train.h5")
    p.add_argument("--test_file", type=str, default="shd_test.h5")
    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--ann_mode", choices=["ann_sub", "ann_add", "snn_add"], default="ann_sub")
    p.add_argument("--ann_hidden_sizes", type=str, default="[2048]")
    p.add_argument("--ann_interval", type=int, default=3)
    p.add_argument("--ann_in_disable", type=str, default="")
    p.add_argument("--ann_out_disable", type=str, default="")
    p.add_argument("--group_size", type=str, nargs="+", default=["1", "1"])
    p.add_argument("--nm_enable", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--nm_counts", type=str, default="[0,0]")
    p.add_argument("--nm_mapper_hidden_size", type=int, default=None)
    p.add_argument("--param_smoothing_enable", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--param_smoothing_tau", type=float, default=1.0)

    p.add_argument("--channel_compress_enable", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--channel_compress_target", type=int, default=70)
    p.add_argument("--channel_compress_mode", choices=["mod_only", "none"], default="mod_only")
    p.add_argument("--train_aug_enable", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--aug_channel_jitter_std", type=float, default=20.0)
    p.add_argument("--train_noise_enable", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--aug_noise_rate_hz", type=float, default=0.0)
    p.add_argument("--hidden_dropout_p", type=float, default=0.0)
    p.add_argument("--snn_reg_enable", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--snn_reg_scale", type=float, default=1.0)
    p.add_argument("--base_snn_ckpt", type=str, default=None)
    return p


def config_from_args(args) -> ExperimentConfig:
    mode = "mod_only" if args.channel_compress_mode == "none" else args.channel_compress_mode
    return ExperimentConfig(
        nb_inputs=args.nb_inputs,
        nb_hidden=args.nb_hidden,
        nb_outputs=args.nb_outputs,
        nb_steps=args.nb_steps,
        max_time=args.max_time,
        time_step=args.time_step,
        tau_syn=args.tau_syn,
        tau_mem=args.tau_mem,
        tau_match_clip=args.tau_match_clip,
        weight_scale=args.weight_scale,
        batch_size=args.batch_size,
        lr=args.lr,
        nb_epochs=args.nb_epochs,
        nb_epochs_snn=args.nb_epochs_snn,
        nb_epochs_mod=args.nb_epochs_mod,
        cache_dir=args.cache_dir,
        cache_subdir=args.cache_subdir,
        train_file=args.train_file,
        test_file=args.test_file,
        save_dir=args.save_dir,
        seed=args.seed,
        ann_mode=args.ann_mode,
        ann_hidden_sizes=parse_int_list(args.ann_hidden_sizes, default=[2048]),
        ann_interval=args.ann_interval,
        ann_in_disable=args.ann_in_disable,
        ann_out_disable=args.ann_out_disable,
        group_size=parse_pair(args.group_size, default=(1, 1)),
        channel_compress_enable=args.channel_compress_enable and args.channel_compress_mode != "none",
        channel_compress_target=args.channel_compress_target,
        channel_compress_mode=mode,
        nm_enable=args.nm_enable,
        nm_counts=parse_pair(args.nm_counts, default=(0, 0)),
        nm_mapper_hidden_size=args.nm_mapper_hidden_size,
        param_smoothing_enable=args.param_smoothing_enable,
        param_smoothing_tau=args.param_smoothing_tau,
        snn_reg_enable=args.snn_reg_enable,
        snn_reg_scale=args.snn_reg_scale,
        train_aug_enable=args.train_aug_enable,
        aug_channel_jitter_std=args.aug_channel_jitter_std,
        train_noise_enable=args.train_noise_enable,
        aug_noise_rate_hz=args.aug_noise_rate_hz,
        hidden_dropout_p=args.hidden_dropout_p,
        base_snn_ckpt=args.base_snn_ckpt,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    set_seed(cfg.seed)
    x_train, y_train, x_test, y_test = open_h5_pair(cfg)
    if args.run_mode == "snn":
        cfg.nb_epochs = args.nb_epochs
        train_snn(cfg, x_train, y_train, x_test, y_test)
    elif args.run_mode == "mod":
        train_modulated(cfg, x_train, y_train, x_test, y_test)
    elif args.run_mode == "staged":
        cfg.nb_epochs = cfg.nb_epochs_snn
        base_state = train_snn(cfg, x_train, y_train, x_test, y_test)
        cfg.base_snn_ckpt = None
        train_modulated(cfg, x_train, y_train, x_test, y_test, base_state=base_state, modulator=build_modulator(cfg))
    else:
        raise ValueError(args.run_mode)


if __name__ == "__main__":
    main()

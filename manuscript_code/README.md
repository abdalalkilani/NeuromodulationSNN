# Manuscript Code

Use `snn_allinone_clean.py` for paper-scale runs. The compact package in `../neuromod_snn/` is for demos and code inspection.

## Data

Expected HDF5 layout:

```text
spikes/times
spikes/units
labels
```

Default SHD paths:

```text
~/data/hdspikes/shd_train.h5
~/data/hdspikes/shd_test.h5
```

Full SHD data:

```text
https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
```

Use `--cache_dir`, `--cache_subdir`, `--train_file`, `--test_file`, and `--val_file` to point at other datasets.

## Base Command

Run from the repository root:

```bash
python manuscript_code/snn_allinone_clean.py \
  --run_mode staged \
  --cache_dir ~/data \
  --cache_subdir hdspikes \
  --train_file shd_train.h5 \
  --test_file shd_test.h5 \
  --save_dir_root Runs/example \
  --nb_hidden 256 \
  --ann_mode ann_sub \
  --ann_interval 3
```

Use `--run_mode snn` for unmodulated baselines and `--run_mode mod` for modulated training from a saved SNN checkpoint.

## Main Options

Baseline SNN:

```bash
--run_mode snn
```

ANN substitution:

```bash
--ann_mode ann_sub
```

ANN addition:

```bash
--ann_mode ann_add
```

Spiking controller:

```bash
--ann_mode snn_add
```

Modulation interval:

```bash
--ann_interval 1
--ann_interval 3
--ann_interval 10
```

Spatial grouping:

```bash
--group_size 5 1
--group_size 10 1
```

Overlapping/Gaussian spread:

```bash
--group_overlap 2 0
--group_distribution normal uniform
--group_normal_std 1.5 1.0
```

Neuromodulator-channel bottleneck:

```bash
--nm_enable true
--nm_counts "[4,2]"
--nm_mapper_type mlp
--nm_mapper_hidden_size 64
```

Parameter smoothing/diffusion:

```bash
--param_smoothing_enable true
--param_smoothing_tau_init 0.2
--param_smoothing_modes all
```

Input-channel compression:

```bash
--channel_compress_enable true
--channel_compress_target 10
--channel_compress_mode mod_only
```

Input/output ablations:

```bash
--ann_in_disable "reset,rest,beta_2"
--ann_out_disable "reset,rest,beta_2"
```

Neuron/parameter modulation fractions:

```bash
--nm_neuron_frac_enable true
--nm_neuron_frac "[0.2,1.0]"
--nm_param_frac_enable true
--nm_param_frac '{"alpha_1":0.2,"beta_1":0.2,"thr":0.2,"reset":0.2,"rest":0.2,"alpha_2":1.0,"beta_2":1.0}'
```

Fixed ablation masks:

```bash
--mod_fixed_mask_enable true
--mod_fixed_mask_seed 123
--mod_fixed_mask_flat_inputs true
```

Spike regularisation:

```bash
--snn_reg_enable true
--mod_reg_enable true
--snn_reg_scale 20
--mod_reg_scale 20
```

Paper-style jitter/noise:

```bash
--train_aug_enable true
--paper_aug_train true
--train_noise_enable true
--aug_channel_jitter_std 20
--aug_noise_rate_hz 5
```

Shapley analysis:

```bash
--ann_shap_enable true
--ann_shap_samples 16
--ann_shap_dataset test
--ann_shap_metric acc
```

Run:

```bash
python manuscript_code/snn_allinone_clean.py --help
```

for all options.

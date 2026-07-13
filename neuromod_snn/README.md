# `neuromod_snn`

This is the compact paper-facing runner. It is based on `snn_allinone_clean.py`, but keeps only the code needed to understand and run the main paper models.

For conceptual explanation, start with:

```text
../notebooks/neuromodulated_snn_tutorial.ipynb
```

## Files

- `config.py`: configuration dataclass, parameter names, valid ranges, parsing helpers.
- `data.py`: SHD-style HDF5 loading, event binning, channel compression, channel jitter, Poisson noise.
- `snn.py`: baseline recurrent SNN, surrogate gradient, training, evaluation, checkpointing.
- `modulation.py`: ANN substitution/additive modulators, SNN additive modulator, grouping, NM bottleneck mapper.
- `modulated.py`: modulated SNN forward pass, modulated training, checkpointing.
- `cli.py`: command-line runner for `snn`, `mod`, and `staged`.

## Supported Run Modes

- `--run_mode snn`: train baseline SNN.
- `--run_mode mod`: train a modulator from `--base_snn_ckpt`.
- `--run_mode staged`: train SNN first, then train a modulator.

## Supported Modulators

- `--ann_mode ann_sub`: ANN substitution. MLP outputs full parameter values.
- `--ann_mode ann_add`: ANN addition. MLP outputs bounded parameter deltas.
- `--ann_mode snn_add`: spiking additive controller.

The release intentionally omits `ann_combo`, RNN/LSTM modulators, SHAP analysis, input-delay experiments, and weight-matrix modulation.

## Core Options

Data and model:

- `--nb_inputs`, `--nb_hidden`, `--nb_outputs`
- `--nb_steps`, `--max_time`, `--time_step`
- `--cache_dir`, `--cache_subdir`, `--train_file`, `--test_file`

Training:

- `--lr`
- `--batch_size`
- `--nb_epochs`
- `--nb_epochs_snn`
- `--nb_epochs_mod`
- `--save_dir`
- `--base_snn_ckpt`

Modulation:

- `--ann_mode`
- `--ann_hidden_sizes`
- `--ann_interval`
- `--ann_in_disable`
- `--ann_out_disable`
- `--group_size`

Neuromodulator bottleneck:

- `--nm_enable`
- `--nm_counts`
- `--nm_mapper_hidden_size`

Biological/efficiency constraints:

- `--param_smoothing_enable`
- `--param_smoothing_tau`
- `--channel_compress_enable`
- `--channel_compress_target`
- `--channel_compress_mode` (`mod_only` or `none` in this compact release)

Regularisation/noise:

- `--snn_reg_enable`
- `--snn_reg_scale`
- `--train_aug_enable`
- `--aug_channel_jitter_std`
- `--train_noise_enable`
- `--aug_noise_rate_hz`
- `--hidden_dropout_p`

## Examples

Baseline SHD:

```bash
python -m neuromod_snn.cli \
  --run_mode snn \
  --nb_hidden 256 \
  --nb_epochs 30 \
  --save_dir runs_shd_snn
```

ANN substitution:

```bash
python -m neuromod_snn.cli \
  --run_mode mod \
  --base_snn_ckpt runs_shd_snn/snn_best.pt \
  --ann_mode ann_sub \
  --ann_hidden_sizes "[2048]" \
  --ann_interval 3 \
  --save_dir runs_shd_ann_sub
```

ANN addition:

```bash
python -m neuromod_snn.cli \
  --run_mode mod \
  --base_snn_ckpt runs_shd_snn/snn_best.pt \
  --ann_mode ann_add \
  --ann_hidden_sizes "[2048]" \
  --ann_interval 3 \
  --save_dir runs_shd_ann_add
```

Grouped modulation:

```bash
python -m neuromod_snn.cli \
  --run_mode mod \
  --base_snn_ckpt runs_shd_snn/snn_best.pt \
  --ann_mode ann_sub \
  --group_size 5 1
```

Neuromodulator-channel bottleneck:

```bash
python -m neuromod_snn.cli \
  --run_mode mod \
  --base_snn_ckpt runs_shd_snn/snn_best.pt \
  --ann_mode ann_add \
  --nm_enable true \
  --nm_counts "[4,2]"
```

SNN additive controller:

```bash
python -m neuromod_snn.cli \
  --run_mode mod \
  --base_snn_ckpt runs_shd_snn/snn_best.pt \
  --ann_mode snn_add \
  --ann_hidden_sizes "[512]"
```

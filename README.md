# Neuromodulated SNNs

Code and tutorial material for the paper:

**Neuromodulation enhances dynamic sensory processing in spiking neural network models**

Start with the notebook if you want to understand the idea:

- [`notebooks/neuromodulated_snn_tutorial.ipynb`](notebooks/neuromodulated_snn_tutorial.ipynb)

The notebook walks through the minimal implementation: a baseline SNN, then the small set of additions needed for an ANN neuromodulator.

Use the package if you want to run paper-style experiments:

- [`neuromod_snn/`](neuromod_snn/)

This release code is based on `snn_allinone_clean.py`, but is intentionally smaller. It keeps the paper-relevant components and omits exploratory analysis code such as SHAP.

## Install

```bash
pip install -r requirements.txt
```

The code expects SHD-style HDF5 event files by default:

```text
~/data/hdspikes/shd_train.h5
~/data/hdspikes/shd_test.h5
```

## Quick Start

Train a small baseline SNN:

```bash
python -m neuromod_snn.cli --run_mode snn --nb_epochs 2
```

Train an ANN substitution modulator from a saved SNN checkpoint:

```bash
python -m neuromod_snn.cli \
  --run_mode mod \
  --base_snn_ckpt runs/snn_best.pt \
  --ann_mode ann_sub \
  --ann_hidden_sizes "[1024]" \
  --ann_interval 3
```

Train staged SNN then modulator:

```bash
python -m neuromod_snn.cli \
  --run_mode staged \
  --nb_epochs_snn 30 \
  --nb_epochs_mod 30 \
  --ann_mode ann_sub
```

## What Is Included

- Baseline recurrent SNN with trainable time constants, thresholds, reset, rest, and weights.
- ANN substitution modulation.
- ANN additive modulation.
- Spiking additive modulator.
- Temporal update intervals.
- Spatial grouping of modulation outputs.
- Neuromodulator-channel bottleneck mapper.
- Input/output block selection for ANN modulators.
- Channel compression for input spike channels.
- Paper-style channel jitter, Poisson noise, and spike regularisation.
- Checkpointing and staged training.

## What Is Not Included

- SHAP / Shapley analysis.
- Legacy exploratory modes not used by the paper-facing experiments.
- RNN/LSTM modulators.
- Weight-matrix modulation.
- Input-delay experiments.

Those features remain in older research scripts, but are intentionally left out here to keep the release readable.


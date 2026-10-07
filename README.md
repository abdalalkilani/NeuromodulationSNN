# Neuromodulated SNNs

Code for:

**Neuromodulation enhances the capability and efficiency of spiking neural networks**

This release contains:

- `neuromod_snn/`: compact readable implementation for demos and inspection.
- `manuscript_code/snn_allinone_clean.py`: full script used for manuscript experiment runs.
- `demo_data/`: small real-SHD subset for checking that the code runs.
- `notebooks/`: tutorial notebook.

For paper-scale reproduction commands and option mappings, see:

```text
manuscript_code/README.md
```

## System Requirements

Required software:

- Python >=3.9
- NumPy >=1.20
- h5py >=3.0
- PyTorch >=1.12

Release smoke test:

- Linux 4.18 x86_64
- Python 3.9.7
- PyTorch 2.3.0+cu118
- NumPy 1.26.4
- h5py 3.6.0

Hardware:

- Demo: no special hardware; CPU is sufficient.
- Manuscript-scale experiments: Imperial College London RCS CX3 Linux HPC.
- CX3 hardware used/available for these jobs included Intel Xeon Platinum 8358 CPU nodes, AMD EPYC 7742 CPU nodes, and NVIDIA GPU nodes including Quadro RTX 6000, L40S, A100, and A40 GPUs.

## Installation

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Typical install time: a few minutes, mostly depending on PyTorch download time.

## Demo Data

The demo files are a small subset of the real Spiking Heidelberg Digits dataset:

```text
demo_data/hdspikes/shd_train.h5
demo_data/hdspikes/shd_test.h5
```

They use the same HDF5 layout as SHD:

```text
spikes/times
spikes/units
labels
```

The demo set has 100 training samples and 40 test samples: 5 train and 2 test examples per class. It is small enough for a quick run, but it is not meant to reproduce manuscript accuracy.

Full SHD data are available from Zenke Lab:

```text
https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
```

## Run Demo

```bash
python -m neuromod_snn.cli \
  --cache_dir demo_data \
  --cache_subdir hdspikes \
  --run_mode staged \
  --nb_inputs 700 \
  --nb_outputs 20 \
  --nb_hidden 16 \
  --nb_steps 10 \
  --batch_size 8 \
  --nb_epochs_snn 1 \
  --nb_epochs_mod 1 \
  --ann_hidden_sizes "[16]" \
  --save_dir demo_runs
```

Expected output:

- one `[snn] epoch=1 ...` line
- one `[mod] epoch=1 ...` line
- checkpoints in `demo_runs/`

Expected runtime: under about 30 seconds on the smoke-test CPU.

## Run on Real Data

By default the compact runner expects:

```text
~/data/hdspikes/shd_train.h5
~/data/hdspikes/shd_test.h5
```

The full local SHD files used here are about 257 MB for training and 76 MB for testing. SHD is publicly available from Zenke Lab under CC BY 4.0:

```text
https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
```

Use another dataset location with:

```bash
python -m neuromod_snn.cli \
  --cache_dir /path/to/data_root \
  --cache_subdir hdspikes \
  --train_file shd_train.h5 \
  --test_file shd_test.h5 \
  --run_mode staged
```

Useful options:

- `--run_mode snn|mod|staged`
- `--ann_mode ann_sub|ann_add|snn_add`
- `--ann_interval K`
- `--group_size H O`
- `--channel_compress_enable true`
- `--channel_compress_mode mod_only`
- `--nm_enable true --nm_counts "[H,O]"`
- `--param_smoothing_enable true`
- `--ann_in_disable ...`
- `--ann_out_disable ...`

## License

MIT License. See `LICENSE`.

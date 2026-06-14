# AGENTS.md

## What this is

Research code for Remaining Useful Life (RUL) prediction on NASA C-MAPSS turbofan engine datasets (FD001–FD004). Two models:

1. **STAR baseline** — standard spatio-temporal attention transformer
2. **STAR FiLM** — student-teacher transformer with Feature-wise Linear Modulation (future sensors, regime/fault embeddings, knowledge distillation)

Accompanies an academic article (Overleaf link in `paper/overleaf_link.md`).

## Stack

Python 3.10 · PyTorch · Hydra (config) · scikit-learn · pandas · numpy

Install: `pip install -r requirements.txt`
Download data: `python scripts/download_data.py`

## Entry points

| Command | Purpose |
|---------|---------|
| `python src/train.py` | Train baseline model |
| `python src/inference.py` | Evaluate baseline (prompts for `.pth` path) |
| `python src/train_film.py` | Train FiLM student-teacher model |
| `python src/inference_film.py` | Evaluate FiLM student (prompts for `.pth` path) |

### Baseline examples

```bash
python src/train.py data=fd001
python src/train.py data=fd002 trainer.epochs=80
python src/inference.py data=fd001
```

### FiLM examples

```bash
# Use per-dataset defaults from get_fd_config
python src/train_film.py data=fd001 fd_num=1
python src/train_film.py data=fd002 fd_num=2

# Override specific params
python src/train_film.py data=fd001 fd_num=1 model.d_model=128 model.nhead=1 trainer.lr=0.00025 trainer.max_lambda=300
```

FiLM config root is `configs/config_film.yaml` (separate from baseline's `config.yaml`).

## Data

C-MAPSS dataset files are **not in the repo**. Expected structure:

```
data/
  FD001/
    train.txt
    test.txt
    RUL.txt
  FD002/ ...
  FD003/ ...
  FD004/ ...
```

Each `configs/data/fd00X.yaml` defines paths, window_size, stride, n_clusters, and num_faults.

Key dataset differences:
- FD001/FD003: `n_clusters=1` (single operating regime, global normalization)
- FD002/FD004: `n_clusters=6` (multiple regimes, regime-specific normalization)
- FD003/FD004: `num_faults=2` (HPC vs Fan fault clustering); FD001/FD002: `num_faults=1`

## FiLM model architecture

The FiLM model (`src/models/star_film.py`) extends STAR with:
- **Teacher** sees future sensor windows + regime/fault embeddings (injected via FiLM bias)
- **Student** only sees current window (no future access)
- **Distillation**: student projectors match teacher encoder features via smooth L1 loss
- Training uses dynamic lambda schedule: ramps up teacher influence, then freezes teacher

The `regime_idx` column is preserved in normalized DataFrames (unlike baseline which drops it).

## Outputs

Hydra writes run artifacts to `artifacts/outputs/<timestamp>/`. Model checkpoints:
- Baseline: `best_model.pth` in working directory
- FiLM: `checkpoints/best_student_FD00X.pth` and `checkpoints/last_student_FD00X.pth`

## Directory structure

- `src/models/star_baseline.py` — baseline STAR model
- `src/models/star_film.py` — FiLM student-teacher model
- `src/data/cmapss_loader.py` — data loading + normalization (shared)
- `src/data/windowing.py` — baseline windowing
- `src/data/windowing_film.py` — FiLM windowing (future windows + regime/fault metadata)
- `src/train.py`, `src/inference.py` — baseline scripts
- `src/train_film.py`, `src/inference_film.py` — FiLM scripts
- `configs/` — Hydra YAML configs (composable: model, data, trainer)
- `baselines/` — reference Jupyter notebooks (STAR baseline + FiLM)

## No CI, tests, or linting

This is a research repo. Verify changes by running training and checking metrics.

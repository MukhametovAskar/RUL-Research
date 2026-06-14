# RUL Prediction — STAR + FiLM

Research code for Remaining Useful Life (RUL) prediction on NASA C-MAPSS turbofan engine datasets (FD001–FD004). Implements the STAR baseline and a custom student-teacher FiLM model.

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

## Dataset

C-MAPSS dataset is **not in the repo** (too large). You can either download it or point to an existing copy.

### Download automatically

```bash
python scripts/download_data.py
```

### Expected structure (flat)

All 12 files in a single folder:

```
data/
  train_FD001.txt
  test_FD001.txt
  RUL_FD001.txt
  train_FD002.txt
  test_FD002.txt
  RUL_FD002.txt
  train_FD003.txt
  test_FD003.txt
  RUL_FD003.txt
  train_FD004.txt
  test_FD004.txt
  RUL_FD004.txt
```

### Custom data path

Use `data.data_root` to point to any folder with the 12 files:

```bash
python src/train.py data=fd001 data.data_root=/path/to/dataset
python src/train_film.py data=fd001 fd_num=1 data.data_root=/path/to/dataset
```

On **Kaggle**, add the dataset and pass its mount path:

```bash
python src/train.py data=fd001 data.data_root=/kaggle/input/cmapss
```

## Training

### Baseline (STAR)

```bash
python src/train.py data=fd001
python src/train.py data=fd002 trainer.epochs=80
```

### FiLM (student-teacher)

```bash
# Default hyperparameters per dataset (from get_fd_config)
python src/train_film.py data=fd001 fd_num=1
python src/train_film.py data=fd002 fd_num=2
python src/train_film.py data=fd003 fd_num=3
python src/train_film.py data=fd004 fd_num=4
```

## All parameters (Hydra CLI overrides)

Every parameter from `get_fd_config()` can be set via CLI before training.
Format: `group.param=value`

### Full example — FD002 with all parameters

```bash
python src/train_film.py \
    data=fd002 \
    fd_num=2 \
    seed=44 \
    max_rul=125 \
    model.d_model=256 \
    model.nhead=4 \
    model.num_scales=4 \
    model.ffn_dim=1024 \
    model.patch_size=4 \
    model.dropout=0.14 \
    model.target_noise_std=0.03 \
    model.future_len=40 \
    model.encoder_layers_per_scale=3 \
    model.decoder_layers_per_scale=2 \
    model.pos_learnable=true \
    trainer.epochs=40 \
    trainer.batch_size=64 \
    trainer.lr=0.0002 \
    trainer.weight_decay=0.0005 \
    trainer.max_lambda=200 \
    trainer.gradient_accumulation_steps=2 \
    trainer.device=cuda
```

### Parameter reference

| CLI parameter | Config file | FD001 | FD002 | FD003 | FD004 |
|---------------|-------------|-------|-------|-------|-------|
| `data.window_size` | `configs/data/fd00X.yaml` | 32 | 64 | 48 | 64 |
| `data.n_clusters` | `configs/data/fd00X.yaml` | 1 | 6 | 1 | 6 |
| `data.num_faults` | `configs/data/fd00X.yaml` | 1 | 1 | 2 | 2 |
| `model.d_model` | `configs/model/star_film.yaml` | 128 | 256 | 128 | 256 |
| `model.nhead` | `configs/model/star_film.yaml` | 1 | 4 | 1 | 4 |
| `model.num_scales` | `configs/model/star_film.yaml` | 3 | 4 | 3 | 4 |
| `model.ffn_dim` | `configs/model/star_film.yaml` | 512 | 1024 | 512 | 512 |
| `model.dropout` | `configs/model/star_film.yaml` | 0.05 | 0.14 | 0.08 | 0.16 |
| `model.target_noise_std` | `configs/model/star_film.yaml` | 0.01 | 0.03 | 0.02 | 0.03 |
| `model.future_len` | `configs/model/star_film.yaml` | 40 | 40 | 40 | 40 |
| `model.patch_size` | `configs/model/star_film.yaml` | 4 | 4 | 4 | 4 |
| `model.pos_learnable` | `configs/model/star_film.yaml` | true | true | true | true |
| `model.encoder_layers_per_scale` | `configs/model/star_film.yaml` | 3 | 3 | 3 | 3 |
| `model.decoder_layers_per_scale` | `configs/model/star_film.yaml` | 2 | 2 | 2 | 2 |
| `trainer.epochs` | `configs/trainer/film.yaml` | 40 | 40 | 40 | 40 |
| `trainer.batch_size` | `configs/trainer/film.yaml` | 32 | 64 | 32 | 64 |
| `trainer.lr` | `configs/trainer/film.yaml` | 0.00025 | 0.0002 | 0.0002 | 0.0002 |
| `trainer.weight_decay` | `configs/trainer/film.yaml` | 1e-5 | 5e-4 | 1e-5 | 1e-4 |
| `trainer.max_lambda` | `configs/trainer/film.yaml` | 300 | 200 | 150 | 300 |
| `trainer.gradient_accumulation_steps` | `configs/trainer/film.yaml` | 2 | 2 | 2 | 2 |
| `seed` | `configs/config_film.yaml` | 44 | 44 | 44 | 44 |
| `max_rul` | `configs/config_film.yaml` | 125 | 125 | 125 | 125 |

### Parameters hardcoded in `src/train_film.py`

These require editing the source file:

| Parameter | Default | Location |
|-----------|---------|----------|
| `optim_betas` | `(0.9, 0.999)` | Adam optimizer creation |
| `optim_eps` | `1e-8` | Adam optimizer creation |
| `lambda_warmup` (freeze_fraction) | `0.75` | `get_lambda_schedule()` |
| `min_lambda` | `{1:150, 2:100, 3:75, 4:100}` | `MIN_LAMBDA` dict |

## Inference

### Baseline
```bash
python src/inference.py data=fd001
# Enter path to .pth file when prompted
```

### FiLM
```bash
python src/inference_film.py data=fd001
# Enter path to student .pth checkpoint when prompted
```

Checkpoints are saved to `checkpoints/`.

## Outputs

- Hydra logs: `artifacts/outputs/<timestamp>/`
- Baseline model: `best_model.pth`
- FiLM models: `checkpoints/best_student_FD00X.pth`, `checkpoints/last_student_FD00X.pth`

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE

from src.utils.seed import set_seed
from src.data.cmapss_loader import load_cmapss
from src.data.windowing_film import (
    apply_regime_specific_normalization_v2,
    create_train_windows_film,
    create_test_windows_film,
    CMapssFilmDataset,
)
from src.utils.metrics import rmse, nasa_score
from src.models.star_film import STARModelFull
from torch.utils.data import DataLoader


def get_lambda_schedule(epoch, epochs, max_lambda, min_lambda, peak_fraction=0.5, freeze_fraction=0.75):
    progress = epoch / epochs
    if progress <= peak_fraction:
        return 1.0 + (max_lambda - 1.0) * (progress / peak_fraction)
    elif progress <= freeze_fraction:
        return max_lambda
    else:
        decay_prog = (progress - freeze_fraction) / (1.0 - freeze_fraction)
        return max_lambda - (max_lambda - min_lambda) * decay_prog


MIN_LAMBDA = {1: 150.0, 2: 100.0, 3: 75.0, 4: 100.0}


def train_one_epoch(student, teacher, loader, optimizer, device, scaler, criterion, cfg, epoch):
    student.train()
    fd = cfg.fd_num
    warmup_epochs = max(1, int(cfg.trainer.epochs * 0.75))

    if epoch > warmup_epochs:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
    else:
        teacher.train()
        for p in teacher.parameters():
            p.requires_grad = True

    cur_lambda = get_lambda_schedule(epoch, cfg.trainer.epochs, cfg.trainer.max_lambda, MIN_LAMBDA.get(fd, 100.0))
    grad_acc = cfg.trainer.get('gradient_accumulation_steps', 2)
    optimizer.zero_grad()

    total_loss = total_ls = total_lt = total_llat = 0.0
    for i, (xb, xf_b, yb_all) in enumerate(tqdm(loader, desc="train", leave=False)):
        xb, xf_b = xb.to(device), xf_b.to(device)
        y_meta = yb_all.to(device)
        target_rul = yb_all[:, 0].to(device)

        with torch.amp.autocast('cuda'):
            s_preds, s_enc = student(xb, return_latents=True)
            t_preds, t_enc = teacher(xb, x_future=xf_b, y_meta=y_meta, return_latents=True)

            loss_s = criterion(s_preds, target_rul)
            loss_t = criterion(t_preds, target_rul) if epoch <= warmup_epochs else torch.tensor(0.0, device=device)
            loss_lat = sum(F.smooth_l1_loss(se, te.detach()) for se, te in zip(s_enc, t_enc)) / len(s_enc)
            loss = (loss_s + loss_t + cur_lambda * loss_lat) / grad_acc

        scaler.scale(loss).backward()
        if (i + 1) % grad_acc == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            if epoch <= warmup_epochs:
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_acc
        total_ls += loss_s.item()
        total_lt += loss_t.item() if epoch <= warmup_epochs else 0.0
        total_llat += loss_lat.item()

    n = len(loader)
    return total_loss / n, total_ls / n, total_lt / n, total_llat / n, cur_lambda


def evaluate(student, loader, device, max_rul=125):
    student.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, xf_b, yb in loader:
            preds = student(xb.to(device))
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = preds.clamp(0.0, float(max_rul))
            ys.append(yb[:, 0].numpy())
            ps.append(preds.cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    return rmse(y_true, y_pred), nasa_score(y_true, y_pred), y_true, y_pred


def plot_scatter(y_true, y_pred, epoch, fd):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, c='purple', edgecolors='none', s=20)
    ax.plot([0, 125], [0, 125], 'k--', linewidth=2, label='Ideal (y=x)')
    ax.set_xlabel('Actual RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title(f'FD00{fd} - Scatter (Epoch {epoch})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_engine_trajectory(student, df_test, engine_rul, engine_idx, window, max_rul, device, fd):
    student.eval()
    sub = df_test[df_test['id'] == engine_idx].sort_values('cycle')
    T = len(sub)
    sensors = sub[[c for c in sub.columns if c.startswith('s')]].values

    X_list = []
    for end in range(1, T + 1):
        if end >= window:
            x = sensors[end - window:end, :]
        else:
            pad = np.repeat(sensors[0:1, :], window - end, axis=0)
            x = np.vstack([pad, sensors[:end, :]])
        X_list.append(x)

    X_tensor = torch.tensor(np.stack(X_list), dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = student(X_tensor)
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = preds.clamp(0.0, float(max_rul)).cpu().numpy().flatten()

    final_rul = engine_rul[engine_idx]
    y_true_time = np.minimum(np.array([final_rul + (T - t) for t in range(1, T + 1)]), max_rul)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_true_time, label='Actual RUL', color='blue', linestyle='--', linewidth=2)
    ax.plot(preds, label=f'Predicted RUL', color='red', linewidth=2)
    ax.set_title(f'FD00{fd} - Engine {engine_idx} Degradation', fontsize=12)
    ax.set_xlabel('Cycles')
    ax.set_ylabel('RUL')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_tsne(student, teacher, loader, device, epoch, fd):
    student.eval()
    teacher.eval()
    xb, xf_b, yb_meta = next(iter(loader))
    xb, xf_b, yb_meta = xb.to(device), xf_b.to(device), yb_meta.to(device)

    with torch.no_grad():
        _, s_lat = student(xb, return_latents=True)
        _, t_lat = teacher(xb, x_future=xf_b, y_meta=yb_meta, return_latents=True)
        s_vec = s_lat[-1].mean(dim=(1, 2)).cpu().numpy()
        t_vec = t_lat[-1].mean(dim=(1, 2)).cpu().numpy()

    latents = np.vstack([s_vec, t_vec])
    perplexity = min(30, len(latents) // 3)
    reduced = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000).fit_transform(latents)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(reduced[:len(s_vec), 0], reduced[:len(s_vec), 1],
               label='Student', alpha=0.6, c='blue', s=30)
    ax.scatter(reduced[len(s_vec):, 0], reduced[len(s_vec):, 1],
               label='Teacher', alpha=0.6, c='red', s=30, marker='x')
    ax.set_title(f'FD00{fd} - Latent Space (t-SNE) Epoch {epoch}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@hydra.main(config_path="../configs", config_name="config_film", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.trainer.device)
    fd_num = cfg.fd_num
    torch.backends.cudnn.benchmark = True

    print(f"=== TRAIN {cfg.data.dataset_name} ===")

    df_train = load_cmapss(cfg.data.train_path)
    df_test = load_cmapss(cfg.data.test_path)
    n_clusters = cfg.data.n_clusters
    df_train, df_test = apply_regime_specific_normalization_v2(df_train, df_test, n_clusters=n_clusters)

    future_len = cfg.model.future_len
    max_rul = cfg.max_rul

    X_tr, X_fut_tr, y_tr_meta = create_train_windows_film(
        df_train, cfg.data.window_size, future_len=future_len, max_rul=max_rul,
        stride=cfg.data.stride, fd_num=fd_num
    )
    print(f"Train windows: {X_tr.shape} | Future: {X_fut_tr.shape} | Meta: {y_tr_meta.shape}")

    df_rul = pd.read_csv(cfg.data.rul_path, sep=r'\s+', header=None)
    X_test, y_test = create_test_windows_film(df_test, df_rul, cfg.data.window_size, max_rul)
    X_fut_test = np.zeros((X_test.shape[0], future_len, X_tr.shape[-1]), dtype=np.float32)

    train_ds = CMapssFilmDataset(X_tr, X_fut_tr, y_tr_meta)
    test_ds = CMapssFilmDataset(X_test, X_fut_test, np.stack([y_test, np.zeros_like(y_test), np.zeros_like(y_test)], axis=1))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg.trainer.batch_size, shuffle=True, pin_memory=pin, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.trainer.batch_size, shuffle=False, pin_memory=pin, num_workers=0)

    n_sensors = X_tr.shape[-1]
    model_params = {
        "window": cfg.data.window_size, "n_sensors": n_sensors,
        "d_model": cfg.model.d_model, "nhead": cfg.model.nhead,
        "num_scales": cfg.model.num_scales, "ffn_dim": cfg.model.ffn_dim,
        "patch_size": cfg.model.patch_size, "dropout": cfg.model.dropout,
        "encoder_layers_per_scale": cfg.model.encoder_layers_per_scale,
        "decoder_layers_per_scale": cfg.model.decoder_layers_per_scale,
        "pos_learnable": cfg.model.pos_learnable,
        "target_noise_std": cfg.model.target_noise_std,
        "num_regimes": n_clusters, "num_faults": cfg.data.num_faults,
        "future_len": future_len,
    }

    student = STARModelFull(**model_params, is_teacher=False).to(device)
    teacher = STARModelFull(**model_params, is_teacher=True).to(device)

    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(teacher.parameters()),
        lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.trainer.epochs, eta_min=1e-5)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_rmse = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    # TensorBoard + CSV
    tb_writer = SummaryWriter(log_dir=f"runs/fd00{fd_num}")
    csv_path = f"metrics_fd00{fd_num}.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train_Loss', 'Loss_Student', 'Loss_Teacher', 'Loss_Latent', 'Lambda', 'LR', 'Test_RMSE', 'Test_Score'])

    # Selected engines for trajectory visualization
    engine_ids_test = [int(e) for e in sorted(df_test['id'].unique())]
    engine_rul_test = dict(zip(engine_ids_test, df_rul.values.flatten()))
    selected_engines = {
        "low": min(engine_rul_test, key=engine_rul_test.get),
        "high": max(engine_rul_test, key=engine_rul_test.get),
        "random": random.choice(engine_ids_test),
    }
    print(f"Trajectory engines: {selected_engines}")

    for epoch in range(1, cfg.trainer.epochs + 1):
        t_loss, l_s, l_t, l_lat, lam = train_one_epoch(
            student, teacher, train_loader, optimizer, device, scaler, criterion, cfg, epoch
        )
        cur_rmse, cur_score, y_true, y_pred = evaluate(student, test_loader, device, max_rul)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch}/{cfg.trainer.epochs} | LR={lr:.6f}")
        print(f"  Loss={t_loss:.2f} | L_s={l_s:.2f} | L_t={l_t:.2f} | L_lat={l_lat:.4f} | lambda={lam:.1f}")
        print(f"  RMSE={cur_rmse:.4f} | Score={cur_score:.4f}")

        # TensorBoard scalars
        tb_writer.add_scalar('Loss/Total', t_loss, epoch)
        tb_writer.add_scalar('Loss/Student_MSE', l_s, epoch)
        tb_writer.add_scalar('Loss/Teacher_MSE', l_t, epoch)
        tb_writer.add_scalar('Loss/Latent_SmoothL1', l_lat, epoch)
        tb_writer.add_scalar('Metrics/Test_RMSE', cur_rmse, epoch)
        tb_writer.add_scalar('Metrics/Test_NASA_Score', cur_score, epoch)
        tb_writer.add_scalar('Hyperparams/Lambda', lam, epoch)
        tb_writer.add_scalar('Hyperparams/LR', lr, epoch)

        # TensorBoard histogram — student projectors
        for name, param in student.named_parameters():
            if 'projectors' in name and param.requires_grad:
                tb_writer.add_histogram(f'Weights/Student_Projector_{name}', param.clone().cpu().data.numpy(), epoch)

        # TensorBoard scatter plot
        fig_scatter = plot_scatter(y_true, y_pred, epoch, fd_num)
        tb_writer.add_figure('Analysis/Test_Scatter', fig_scatter, epoch)
        plt.close(fig_scatter)

        # TensorBoard engine trajectories
        for tag, eid in selected_engines.items():
            y_true_traj, y_pred_traj = plot_engine_trajectory(
                student, df_test, engine_rul_test, eid, cfg.data.window_size, max_rul, device, fd_num
            )
            tb_writer.add_figure(f'Engine/{tag}_engine_{eid}', y_true_traj, epoch)
            plt.close(y_true_traj)

        # TensorBoard t-SNE (every 5 epochs + last)
        if epoch % 5 == 0 or epoch == cfg.trainer.epochs:
            fig_tsne = plot_tsne(student, teacher, train_loader, device, epoch, fd_num)
            tb_writer.add_figure('Latent/tSNE', fig_tsne, epoch)
            plt.close(fig_tsne)

        tb_writer.flush()

        # CSV
        csv_writer.writerow([epoch, t_loss, l_s, l_t, l_lat, lam, lr, cur_rmse, cur_score])
        csv_file.flush()

        # Save best model
        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            path = f"checkpoints/best_student_{cfg.data.dataset_name}.pth"
            torch.save({"model": student.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True), "rmse": best_rmse}, path)
            print(f"  -> Saved best model (RMSE={best_rmse:.4f})")

    # Save last model
    path = f"checkpoints/last_student_{cfg.data.dataset_name}.pth"
    torch.save({"model": student.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, path)

    tb_writer.close()
    csv_file.close()
    print(f"\n=== Done. Best RMSE: {best_rmse:.4f} ===")
    print(f"TensorBoard: tensorboard --logdir=runs/fd00{fd_num}")
    print(f"CSV metrics: {csv_path}")


if __name__ == "__main__":
    main()

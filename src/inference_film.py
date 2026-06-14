import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
import numpy as np

from src.data.cmapss_loader import load_cmapss
from src.data.windowing_film import (
    apply_regime_specific_normalization_v2,
    create_test_windows_film,
    CMapssFilmDataset,
)
from src.utils.metrics import rmse, nasa_score
from src.models.star_film import STARModelFull
from torch.utils.data import DataLoader


@hydra.main(config_path="../configs", config_name="config_film", version_base="1.3")
def main(cfg: DictConfig):
    device = torch.device(cfg.trainer.device)

    df_train = load_cmapss(cfg.data.train_path)
    df_test = load_cmapss(cfg.data.test_path)
    n_clusters = cfg.data.n_clusters
    df_train, df_test = apply_regime_specific_normalization_v2(df_train, df_test, n_clusters=n_clusters)

    df_rul = pd.read_csv(cfg.data.rul_path, sep=r'\s+', header=None)
    X_test, y_test = create_test_windows_film(df_test, df_rul, cfg.data.window_size, cfg.max_rul)
    future_len = cfg.model.future_len
    X_fut_test = np.zeros((X_test.shape[0], future_len, X_test.shape[-1]), dtype=np.float32)

    test_ds = CMapssFilmDataset(X_test, X_fut_test, np.stack([y_test, np.zeros_like(y_test), np.zeros_like(y_test)], axis=1))
    test_loader = DataLoader(test_ds, batch_size=cfg.trainer.batch_size, shuffle=False)

    n_sensors = X_test.shape[-1]
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

    model_path = input("Enter path to student .pth checkpoint: ")
    checkpoint = torch.load(model_path, map_location=device)
    student.load_state_dict(checkpoint["model"])
    student.eval()

    ys, ps = [], []
    with torch.no_grad():
        for xb, xf_b, yb in test_loader:
            preds = student(xb.to(device))
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = preds.clamp(0.0, float(cfg.max_rul))
            ys.append(yb[:, 0].numpy())
            ps.append(preds.cpu().numpy())

    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    print(f"\n=== INFERENCE RESULTS ({cfg.data.dataset_name}) ===")
    print(f"Test RMSE: {rmse(y_true, y_pred):.4f} | NASA Score: {nasa_score(y_true, y_pred):.4f}")


if __name__ == "__main__":
    main()

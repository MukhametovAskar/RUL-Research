import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import pandas as pd
import os

from src.utils.seed import set_seed
from src.data.cmapss_loader import load_cmapss, apply_regime_specific_normalization
from src.data.windowing import create_train_windows, create_test_windows, CMapssWindowDataset
from src.engine.trainer import train_one_epoch
from src.engine.evaluator import evaluate_and_plot
from torch.utils.data import DataLoader

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.trainer.device)
    
    # Data Setup
    df_train = load_cmapss(cfg.data.train_path)
    df_test = load_cmapss(cfg.data.test_path)
    df_train, df_test = apply_regime_specific_normalization(df_train, df_test, n_clusters=cfg.data.n_clusters)
    
    X_tr, y_tr = create_train_windows(df_train, cfg.data.window_size, cfg.max_rul, stride=cfg.data.stride)
    df_rul = pd.read_csv(cfg.data.rul_path, sep=r'\s+', header=None)
    X_test, y_test = create_test_windows(df_test, df_rul, cfg.data.window_size, cfg.max_rul)
    
    train_loader = DataLoader(CMapssWindowDataset(X_tr, y_tr), batch_size=cfg.trainer.batch_size, shuffle=True)
    test_loader = DataLoader(CMapssWindowDataset(X_test, y_test), batch_size=cfg.trainer.batch_size, shuffle=False)

    # Model & Optimization
    model = instantiate(cfg.model).to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    criterion = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type != 'cpu'))

    best_rmse = float('inf')
    
    for epoch in range(1, cfg.trainer.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion)
        test_rmse, test_score = evaluate_and_plot(model, test_loader, device, cfg.max_rul, epoch)
        
        print(f"Epoch {epoch} | Loss: {train_loss:.4f} | RMSE: {test_rmse:.4f} | Score: {test_score:.4f}")
        
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(model.state_dict(), "best_model.pth")
            print(f"--> Saved new best model with RMSE {best_rmse:.4f}")

if __name__ == "__main__":
    main()

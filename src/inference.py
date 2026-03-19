import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import pandas as pd

from src.data.cmapss_loader import load_cmapss, apply_regime_specific_normalization
from src.data.windowing import create_test_windows, CMapssWindowDataset
from src.engine.evaluator import evaluate_and_plot
from torch.utils.data import DataLoader

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    device = torch.device(cfg.trainer.device)
    
    df_train = load_cmapss(cfg.data.train_path) 
    df_test = load_cmapss(cfg.data.test_path)
    _, df_test = apply_regime_specific_normalization(df_train, df_test, n_clusters=cfg.data.n_clusters)
    
    df_rul = pd.read_csv(cfg.data.rul_path, sep=r'\s+', header=None)
    X_test, y_test = create_test_windows(df_test, df_rul, cfg.data.window_size, cfg.max_rul)
    test_loader = DataLoader(CMapssWindowDataset(X_test, y_test), batch_size=cfg.trainer.batch_size, shuffle=False)

    model = instantiate(cfg.model).to(device)
    model_path = input("Enter absolute path to the .pth model file: ")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    rmse, score = evaluate_and_plot(model, test_loader, device, cfg.max_rul, epoch="inference_test")
    print(f"\n=== INFERENCE RESULTS ===")
    print(f"Test RMSE: {rmse:.4f} | Test Score: {score:.4f}")

if __name__ == "__main__":
    main()

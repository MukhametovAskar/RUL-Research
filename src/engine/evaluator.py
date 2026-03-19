import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.metrics import rmse, nasa_score

def evaluate_and_plot(model, loader, device, max_rul=125, epoch=None):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).clamp(0.0, float(max_rul))
            ys.append(yb.cpu().numpy())
            ps.append(preds.cpu().numpy())
            
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    
    if epoch is not None:
        plt.figure(figsize=(12, 6))
        idx = np.argsort(y_true)[::-1]
        plt.plot(y_true[idx], label="True RUL", color='blue')
        plt.plot(y_pred[idx], label="Predicted RUL", color='red', alpha=0.6)
        plt.legend()
        plt.title(f"RUL Predictions - Epoch {epoch}")
        plt.xlabel("Samples (sorted by True RUL)")
        plt.ylabel("RUL")
        plt.savefig(f"predictions_epoch_{epoch}.png") 
        plt.close()

    return rmse(y_true, y_pred), nasa_score(y_true, y_pred)

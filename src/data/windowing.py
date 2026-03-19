import numpy as np
import torch
from torch.utils.data import Dataset

def create_train_windows(df, window, max_rul=125, stride=1):
    X_windows, y_windows = [], []
    for eid in df['id'].unique():
        sub = df[df['id']==eid].sort_values('cycle')
        T = len(sub)
        rul_all = np.minimum(np.array([(T-1)-i for i in range(T)]), max_rul)
        sensors = sub[[c for c in sub.columns if c.startswith('s')]].values
        for end in range(window, T+1, stride):
            X_windows.append(sensors[end-window:end, :])
            y_windows.append(rul_all[end-1])
    return np.stack(X_windows), np.array(y_windows, dtype=np.float32)

def create_test_windows(df, df_rul, window, max_rul):
    r_test = df_rul.values.flatten()
    X_test_list, y_test_list = [], []
    for i, eid in enumerate(df['id'].unique()):
        sub = df[df['id']==eid].sort_values('cycle')
        T = len(sub)
        sensors = sub[[c for c in sub.columns if c.startswith('s')]].values

        if T >= window: x = sensors[-window:, :]
        else:
            pad = np.repeat(sensors[0:1,:], window-T, axis=0)
            x = np.vstack([pad, sensors])
        X_test_list.append(x)
        y_test_list.append(min(r_test[i], max_rul))
    return np.stack(X_test_list), np.array(y_test_list, dtype=np.float32)

class CMapssWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X.astype(np.float32), y.astype(np.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

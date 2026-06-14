import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans


def apply_regime_specific_normalization_v2(df_train, df_test, n_clusters=6):
    op_cols = ["op1", "op2", "op3"]
    sensor_cols = [c for c in df_train.columns if c.startswith('s')]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df_train[op_cols])

    df_train['regime_idx'] = kmeans.predict(df_train[op_cols])
    df_test['regime_idx'] = kmeans.predict(df_test[op_cols])

    df_train = df_train.copy()
    df_test = df_test.copy()

    for cluster_id in range(n_clusters):
        tr_mask = df_train['regime_idx'] == cluster_id
        te_mask = df_test['regime_idx'] == cluster_id
        if tr_mask.sum() == 0:
            continue

        mins = df_train.loc[tr_mask, sensor_cols].min()
        maxs = df_train.loc[tr_mask, sensor_cols].max()
        diffs = maxs - mins
        diffs[diffs == 0] = 1.0

        df_train.loc[tr_mask, sensor_cols] = (df_train.loc[tr_mask, sensor_cols] - mins) / diffs
        if te_mask.sum() > 0:
            df_test.loc[te_mask, sensor_cols] = (df_test.loc[te_mask, sensor_cols] - mins) / diffs

    return df_train, df_test


def create_train_windows_film(df, window, future_len=20, max_rul=125, stride=1, fd_num=1):
    X_windows, X_future, y_meta = [], [], []
    sensor_cols = [c for c in df.columns if c.startswith('s')]

    fault_mapping = {}
    if fd_num >= 3:
        df_temp = df.copy()
        df_temp[sensor_cols] = df.groupby('regime_idx')[sensor_cols].transform(lambda x: x - x.mean())
        last_states = df_temp.groupby('id').tail(20).groupby('id')[sensor_cols].mean()
        km_fault = KMeans(n_clusters=2, random_state=42, n_init=10).fit(last_states)
        fault_mapping = dict(zip(last_states.index, km_fault.labels_))
        del df_temp

    for eid in df['id'].unique():
        sub = df[df['id'] == eid].sort_values('cycle')
        fm = fault_mapping.get(eid, 0) if fd_num >= 3 else 0
        T = len(sub)
        rul_all = np.minimum(np.array([(T - 1) - i for i in range(T)]), max_rul)
        sensors = sub[sensor_cols].values
        regimes = sub['regime_idx'].values

        for end in range(window, T + 1, stride):
            X_windows.append(sensors[end - window:end, :])

            f_start = end
            f_end = end + future_len
            if f_end <= T:
                fut = sensors[f_start:f_end, :]
            else:
                pad_len = f_end - T
                last_val = sensors[-1:, :]
                fut = np.vstack([sensors[f_start:T, :], np.repeat(last_val, pad_len, axis=0)])

            X_future.append(fut)
            y_meta.append([rul_all[end - 1], regimes[end - 1], fm])

    return np.stack(X_windows), np.stack(X_future), np.array(y_meta, dtype=np.float32)


def create_test_windows_film(df, df_rul, window, max_rul):
    r_test = df_rul.values.flatten()
    X_test_list, y_test_list = [], []
    for i, eid in enumerate(df['id'].unique()):
        sub = df[df['id'] == eid].sort_values('cycle')
        T = len(sub)
        sensors = sub[[c for c in sub.columns if c.startswith('s')]].values
        if T >= window:
            x = sensors[-window:, :]
        else:
            pad = np.repeat(sensors[0:1, :], window - T, axis=0)
            x = np.vstack([pad, sensors])
        X_test_list.append(x)
        y_test_list.append(min(r_test[i], max_rul))
    return np.stack(X_test_list), np.array(y_test_list, dtype=np.float32)


class CMapssFilmDataset(Dataset):
    def __init__(self, X, X_fut, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.X_fut = torch.tensor(X_fut, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.X_fut[idx], self.y[idx]

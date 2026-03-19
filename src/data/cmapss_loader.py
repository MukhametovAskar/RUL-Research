import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def load_cmapss(path):
    df = pd.read_csv(path, sep=r'\s+', header=None)
    cols = ["id", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
    df.columns = cols
    selected_sensors = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
    keep_cols = ["id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in selected_sensors]
    
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    df[sensor_cols] = df[sensor_cols].astype(np.float32)
    return df[keep_cols]

def apply_regime_specific_normalization(df_train, df_test, n_clusters=6):
    op_cols = ["op1", "op2", "op3"]
    sensor_cols = [c for c in df_train.columns if c.startswith('s')]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df_train[op_cols])

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train['cluster'] = kmeans.predict(df_train[op_cols])
    df_test['cluster'] = kmeans.predict(df_test[op_cols])

    for cluster_id in range(n_clusters):
        tr_mask = df_train['cluster'] == cluster_id
        te_mask = df_test['cluster'] == cluster_id
        if tr_mask.sum() == 0: continue

        mins = df_train.loc[tr_mask, sensor_cols].min()
        maxs = df_train.loc[tr_mask, sensor_cols].max()
        diffs = maxs - mins
        diffs[diffs == 0] = 1.0

        df_train.loc[tr_mask, sensor_cols] = (df_train.loc[tr_mask, sensor_cols] - mins) / diffs
        if te_mask.sum() > 0:
            df_test.loc[te_mask, sensor_cols] = (df_test.loc[te_mask, sensor_cols] - mins) / diffs

    df_train.drop(columns=['cluster'], inplace=True)
    df_test.drop(columns=['cluster'], inplace=True)
    return df_train, df_test

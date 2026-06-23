import torch
import numpy as np
import torch.nn as nn
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def safe_load_npy(path):
    if os.path.exists(path):
        return np.load(path)
    raise FileNotFoundError(f"文件不存在: {path}")


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = 4
        self.feature = args.feature

        feature_dir = f"../{args.feature}"
        locations = safe_load_npy(os.path.join(feature_dir, "locations.npy"))
        locations_e = safe_load_npy(os.path.join(feature_dir, "locations_e.npy"))

        cpan_flat = np.radians(locations_e.reshape(-1, 2))
        station_coord = np.radians(locations.reshape(-1, 2))

        nbrs = NearestNeighbors(n_neighbors=4, algorithm="ball_tree", metric="haversine").fit(cpan_flat)
        _, indices = nbrs.kneighbors(station_coord)

        self.register_buffer("neighbor_indices", torch.from_numpy(indices).long())

        # 加载权重
        w_path = os.path.join(feature_dir, "pure_kriging_weights_norm.csv")
        weights_df = pd.read_csv(w_path, index_col="station_id")
        self.register_buffer("k_weights", torch.FloatTensor(weights_df.values))

    def forward(self, obs_his, era_his, cobs, cera, mask, datee):
        B, N, T, C = obs_his.shape
        idx = self.neighbor_indices
        w = self.k_weights.view(1, N, 4, 1, 1)
        neighbors = era_his[:, idx, :, :]
        krig = torch.sum(neighbors * w, dim=2)
        out = torch.where(mask.bool(), obs_his, krig)
        return out, out
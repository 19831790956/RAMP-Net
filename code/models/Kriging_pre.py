import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors


# --------------------------
# 工具函数
# --------------------------
def safe_load_npy(file_path):
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"文件不存在：{file_path}")


def haversine_dist(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a))


def spherical_variogram(h, nugget, sill, range_a):
    h = np.asanyarray(h)
    res = np.zeros_like(h)
    mask = (h > 0) & (h <= range_a)
    res[mask] = nugget + (sill - nugget) * (1.5 * h[mask] / range_a - 0.5 * (h[mask] / range_a) ** 3)
    res[h > range_a] = sill
    return res


# --------------------------
# ✅ 全局变异函数：只使用网格（科学！）
# --------------------------
def compute_global_variogram_from_grid(locs_grid, grid_data):
    print("✅ 计算全局变异函数（仅使用网格）")
    M = locs_grid.shape[0]
    dists = []
    gammas = []

    step = max(1, M // 200)
    for i in range(0, M, step):
        for j in range(i + 1, M, step):
            d = haversine_dist(locs_grid[i, 0], locs_grid[i, 1], locs_grid[j, 0], locs_grid[j, 1])
            diff = grid_data[:, i] - grid_data[:, j]
            gamma = 0.5 * np.nanmean(diff ** 2)
            if np.isnan(gamma):
                continue
            dists.append(d)
            gammas.append(gamma)

    dists = np.array(dists)
    gammas = np.array(gammas)
    global_var = np.nanvar(grid_data)
    p0 = [1e-3, global_var * 0.9, np.median(dists)]

    try:
        popt, _ = curve_fit(spherical_variogram, dists, gammas, p0=p0, maxfev=15000)
        popt = np.clip(popt, [1e-8, 1e-8, 1e-8], [None, None, None])
    except:
        popt = [1e-3, global_var, np.median(dists)]

    print(f"✅ 全局参数：nugget={popt[0]:.4f}, sill={popt[1]:.4f}, range={popt[2]:.4f}")
    return popt


# --------------------------
# ✅ 克里金权重：只靠位置，不用站点Y
# --------------------------
def pure_ordinary_kriging(args):
    feature_dir = f"../../{args.feature}"
    filename_mean_e = f"../scaler/{args.feature}_mean_e.npy"
    filename_std_e = f"../scaler/{args.feature}_std_e.npy"

    # 加载位置
    locs_station = np.radians(safe_load_npy(os.path.join(feature_dir, "locations.npy")))
    locs_grid = np.radians(safe_load_npy(os.path.join(feature_dir, "locations_e.npy")))

    # 加载网格数据（只用网格！）
    X_grid = safe_load_npy(os.path.join(feature_dir, "X_train_all_e.npy"))
    mean_e = np.load(filename_mean_e)
    std_e = np.load(filename_std_e)
    X_grid_norm = (X_grid - mean_e) / std_e
    X_grid_norm = X_grid_norm[:, :, 0]

    # ========================
    # 全局变异函数（网格专属）
    # ========================
    global_popt = compute_global_variogram_from_grid(locs_grid, X_grid_norm)

    # 4近邻
    nbrs = NearestNeighbors(n_neighbors=4, algorithm="ball_tree", metric="haversine").fit(locs_grid)
    _, indices = nbrs.kneighbors(locs_station)

    n_sta = locs_station.shape[0]
    weights = np.zeros((n_sta, 4))

    for i in range(n_sta):
        try:
            idx = indices[i]
            g = locs_grid[idx]
            s = locs_station[i]

            d0 = haversine_dist(s[0], s[1], g[:, 0], g[:, 1])
            gamma0 = spherical_variogram(d0, *global_popt)

            mat = np.zeros((4, 4))
            for a in range(4):
                for b in range(4):
                    d = haversine_dist(g[a, 0], g[a, 1], g[b, 0], g[b, 1])
                    mat[a, b] = spherical_variogram(d, *global_popt)

            A = np.ones((5, 5))
            A[:4, :4] = mat
            A[4, 4] = 0
            B = np.ones(5)
            B[:4] = gamma0

            w = np.linalg.solve(A, B)
            weights[i] = w[:4]
        except:
            pass

    # 保存权重（文件名不变）
    df = pd.DataFrame(weights, columns=[f"weight_{k + 1}" for k in range(4)])
    df.index.name = "station_id"
    out_path = os.path.join(feature_dir, "pure_kriging_weights_norm.csv")
    df.to_csv(out_path)
    print(f"\n✅ 权重已保存：{out_path}")
    return out_path, n_sta, n_sta


# --------------------------
# 运行
# --------------------------
if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.feature = "v10"


    args = Args()
    pure_ordinary_kriging(args)
    print("✅ 全部完成")
import numpy as np
from scipy.stats import qmc
from scipy.spatial import cKDTree

# ----- 初始设置 -----
seed = 42 # 跟主动学习的随机数种子保持一致
seed_test = seed * 2 
seed_pool = seed

n_test = 200
n_pool = int(1e7)

param_bounds = {
    "thk": (0.02, 0.20),
    "E":   (25e9, 40e9),
    "rho": (2200, 2600),
    "LL":  (0.0, 10.0),
    "SDL": (0.0, 5.0),
    "L":   (1.0, 3.0),
    "W":   (0.5, 2.0),
}

param_order = ["thk", "E", "rho", "LL", "SDL", "L", "W"]

bounds_array = np.array([param_bounds[name] for name in param_order], dtype=float)

# ----- 拉丁超立方采样 -----
def lhs_in_bounds(n, bounds, seed):
    sampler = qmc.LatinHypercube(d=bounds.shape[0], seed=seed)
    X01 = sampler.random(n)
    return qmc.scale(X01, bounds[:, 0], bounds[:, 1]), X01  # 返回原始空间和归一化空间的点

# 1) test
X_test, X_test01 = lhs_in_bounds(n_test, bounds_array, seed_test)

# 2) 自动选 delta：test 内部最近邻距离的中位数 * 0.5 
tree_test = cKDTree(X_test01)
d_nn, _ = tree_test.query(X_test01, k=2)      # k=2: self + nearest other
nn_dist = d_nn[:, 1]
delta = 0.5 * np.median(nn_dist)

# 3) pool + 过滤（距离在归一化空间 [0,1]^d 上算 L2）
X_pool, X_pool01 = lhs_in_bounds(n_pool, bounds_array, seed_pool)
dist_to_test, _ = tree_test.query(X_pool01, k=1)
mask = dist_to_test >= delta
X_pool_filtered = X_pool[mask]

print(f"delta = {delta:.4f}  |  pool kept: {X_pool_filtered.shape[0]} / {n_pool}")

# - X_test 用于最终评估（固定不变）
# - X_pool_filtered 作为主动学习的候选池（无放回挑点跑仿真）
# - 主要的目的是保证测试集和候选池(训练集)之间有足够的距离，避免过拟合和评估偏差，同时保持测试集的代表性。
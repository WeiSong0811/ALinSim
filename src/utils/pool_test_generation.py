import numpy as np
from scipy.stats import qmc
from scipy.spatial import cKDTree
import pandas as pd
from itertools import product
from tqdm import tqdm

def generation_pool_pan(seed=42, n_test=100, n_pool=int(1e5)):
    # ----- 初始设置 -----
    #seed = 42 # 跟主动学习的随机数种子保持一致
    seed_test = seed * 2 
    seed_pool = seed

    param_bounds = {
        'PS:PAN ratio':(3/7, 1),
        'Feed rate(mL/h)':(2/3, 1),
        'Distance(cm)':(0, 1),
        'Mass fraction of solute':(0.35, 1),
        'Mass fraction of SiO2 in solute ':(0.5, 1),
        'Applied voltage(kV)':(4/7, 1),
        'Inner diameter(mm)':(0.311, 1),
    }

    param_order = ["PS:PAN ratio", 
                "Feed rate(mL/h)", 
                "Distance(cm)", 
                "Mass fraction of solute", 
                "Mass fraction of SiO2 in solute ", 
                "Applied voltage(kV)", 
                "Inner diameter(mm)"]

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
    print(f"X_test shape: {X_test.shape}, X_pool_filtered shape: {X_pool_filtered.shape}")

    X_test = pd.DataFrame(X_test, columns=param_order)
    X_pool_filtered = pd.DataFrame(X_pool_filtered, columns=param_order)

    return X_test, X_pool_filtered


def generation_pool_trc(
    seed=42,
    n_test=100,
    delta_factor=0.5,
    verbose=True,
):
    """
    TRC parameter generation with paper-consistent validity rules.

    Rules (from paper):
    - Center block (2) must exist
    - Side blocks (1,3) may not exist
    - If x_i == 0, then y_i == 0
    - If block exists, x_i in {1..9}, y_i in {1..3}
    - Center block always exists: x2 in {1..9}, y2 in {1..3}

    Test set:
    - sampled by paper-like integerized LHS, then filtered to valid combinations
    - continues sampling rounds until enough unique valid test points are collected

    Pool:
    - all valid discrete combinations (should be exactly 21168)

    Filtering:
    - remove pool points too close to test points in normalized [0,1]^6 space
    """

    # ---------------------------------------------
    # 1) Parameter definition for "raw" integerized LHS box
    #    (before validity filtering)
    # ---------------------------------------------
    # We allow side x/y to hit 0 so that "block absent" can occur.
    param_order = ["x_0", "x_1", "x_2", "y_0", "y_1", "y_2"]

    # raw LHS integerization box (inclusive integer domains)
    # x1,x3: 0..9 ; x2: 1..9
    # y1,y3: 0..3 ; y2: 1..3
    lows = np.array([0, 1, 0, 0, 1, 0], dtype=int)
    highs = np.array([9, 9, 9, 3, 3, 3], dtype=int)

    highs_plus1 = highs + 1  # for LHS scaling then floor/int

    # ---------------------------------------------
    # 2) Validity check according to paper rules
    # ---------------------------------------------
    def is_valid_trc_row(row):
        x1, x2, x3, y1, y2, y3 = map(int, row)

        # center block must exist
        if x2 <= 0 or y2 <= 0:
            return False

        # side blocks: x==0 <=> y==0 ; x>0 => y>0
        # (paper explicitly says y=0 if x=0; counting formula implies if x>0 then y in 1..3)
        if (x1 == 0 and y1 != 0) or (x1 > 0 and y1 == 0):
            return False
        if (x3 == 0 and y3 != 0) or (x3 > 0 and y3 == 0):
            return False

        # bounds safety
        if not (0 <= x1 <= 9 and 1 <= x2 <= 9 and 0 <= x3 <= 9):
            return False
        if not (0 <= y1 <= 3 and 1 <= y2 <= 3 and 0 <= y3 <= 3):
            return False

        return True

    def filter_valid(X):
        X = np.asarray(X, dtype=int)
        mask = np.array([is_valid_trc_row(row) for row in X], dtype=bool)
        return X[mask]

    # ---------------------------------------------
    # 3) Build ALL valid pool combinations (exact space)
    # ---------------------------------------------
    all_raw = product(
        range(0, 10),   # x1
        range(1, 10),   # x2
        range(0, 10),   # x3
        range(0, 4),    # y1
        range(1, 4),    # y2
        range(0, 4),    # y3
    )
    X_pool_all = np.array(list(all_raw), dtype=int)
    X_pool_all = filter_valid(X_pool_all)
    print(f"Total valid pool combinations (paper-consistent): {X_pool_all.shape[0]} (should be 21168)")

    # ---------------------------------------------
    # 4) Helper: normalize to [0,1]^6 for distance
    # ---------------------------------------------
    # use raw integer box normalization (same coordinate system for test/pool)
    def to_unit_cube(X_int):
        X = np.asarray(X_int, dtype=float)
        span = (highs - lows).astype(float)
        span[span == 0] = 1.0
        return (X - lows) / span

    # ---------------------------------------------
    # 5) Paper-style test sampling:
    #    LHS -> scale -> int -> valid-filter -> dedup until n_test reached
    # ---------------------------------------------
    def lhs_integerized_batch(n, seed_local):
        sampler = qmc.LatinHypercube(d=6, seed=seed_local)
        X01 = sampler.random(n)
        X_scaled = qmc.scale(X01, lows.astype(float), highs_plus1.astype(float))
        X_int = X_scaled.astype(int)  # floor
        X_int = np.clip(X_int, lows, highs)
        return X_int

    seed_test = seed * 2 + 1
    batch_n = max(n_test * 4, 200)   # 单次采样数量，可按需改大

    X_batch = lhs_integerized_batch(batch_n, seed_test)
    X_batch = filter_valid(X_batch)

    # deduplicate while preserving order
    seen = set()
    X_test_list = []
    for row in X_batch:
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key)
            X_test_list.append(row.copy())
            if len(X_test_list) >= n_test:
                break

    if len(X_test_list) < n_test:
        raise RuntimeError(
            f"Single-shot sampling produced only {len(X_test_list)} unique valid test points, "
            f"but n_test={n_test}. Increase batch_n."
        )

    X_test = np.vstack(X_test_list)

    # ---------------------------------------------
    # 6) Compute delta from test NN distances (normalized space)
    # ---------------------------------------------
    X_test01 = to_unit_cube(X_test)
    tree_test = cKDTree(X_test01)
    d_nn, _ = tree_test.query(X_test01, k=2)  # self + nearest other
    nn_dist = d_nn[:, 1]
    delta = float(delta_factor * np.median(nn_dist))

    # ---------------------------------------------
    # 7) Filter pool:
    #    - remove test points themselves
    #    - remove points too close to test
    # ---------------------------------------------
    test_set = set(map(tuple, X_test.tolist()))
    mask_not_test = np.array([tuple(row) not in test_set for row in X_pool_all], dtype=bool)
    X_pool_candidates = X_pool_all[mask_not_test]

    X_pool01 = to_unit_cube(X_pool_candidates)
    dist_to_test, _ = tree_test.query(X_pool01, k=1)
    mask_far = dist_to_test >= delta
    X_pool_filtered = X_pool_candidates[mask_far]

    # ---------------------------------------------
    # 8) Return DataFrames + metadata
    # ---------------------------------------------
    X_test_df = pd.DataFrame(X_test, columns=param_order)
    X_pool_filtered_df = pd.DataFrame(X_pool_filtered, columns=param_order)

    meta = {
        "param_order": param_order,
        "raw_bounds": {
            "x_0": (0, 9), "x_1": (1, 9), "x_2": (0, 9),
            "y_0": (0, 3), "y_1": (1, 3), "y_2": (0, 3),
        },
        "paper_valid_pool_count": int(X_pool_all.shape[0]),  # should be 21168
        "n_test": int(X_test.shape[0]),
        "delta_factor": float(delta_factor),
        "delta": float(delta),
        "test_nn_dist_median": float(np.median(nn_dist)),
        "test_nn_dist_min": float(np.min(nn_dist)),
        "test_nn_dist_max": float(np.max(nn_dist)),
        "n_pool_after_remove_test": int(X_pool_candidates.shape[0]),
        "n_pool_filtered": int(X_pool_filtered.shape[0]),
    }

    if verbose:
        print(f"Valid pool count (paper-consistent): {X_pool_all.shape[0]}, should be 21168")
        print(f"delta = {delta:.4f}  (={delta_factor} * median NN distance in normalized space)")
        print(f"Pool after removing test points: {X_pool_candidates.shape[0]}")
        print(f"Pool kept after distance filtering: {X_pool_filtered.shape[0]} / {X_pool_candidates.shape[0]}")

    return X_test_df, X_pool_filtered_df, meta
# for seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
#    X_test_trc, X_pool_trc, meta_trc = generation_pool_trc(seed=seed, n_test=150, delta_factor=0.5)
#
#    X_test_trc.to_csv(f'../data/trc_test_seed{seed}.csv', index=False)


def generation_pool_fea(
    seed=42,
    n_test=150,
    n_pool=None,
    delta_factor=0.5,
    verbose=True,
):
    """
    FEA parameter generation with paper-consistent validity rules.
    
    parameters:
    seed : int
    n_test : int
    delta_factor : float
        距离过滤的 delta 是 test 内部最近邻距离中位数的多少倍，建议 0.5~1.0

    verbose : bool
        是否打印详细信息
    """

    # ---------------------------------------------
    # 1) Parameter definition for "raw" integerized LHS box
    #    (before validity filtering)
    # ---------------------------------------------
    # We allow side x/y to hit 0 so that "block absent" can occur.
    param_order = ["d", "h/d", "b", "E", "ll", "sdl"]

    seed_test = seed
    seed_pool = seed * 2 + 1
    # ---------------------------------------------
    # 3) Build ALL valid pool combinations (exact space)
    # ---------------------------------------------
    grid = {
        "d":   np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250], dtype=float),
        "h/d": np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.25], dtype=float),
        "b":   np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000], dtype=float),
        "E":   np.array([24000, 26700, 30100, 32800, 34800, 37400, 39600, 42200], dtype=float),
        "ll":  np.array([-1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0], dtype=float),
        "sdl": np.array([0.0, -0.25, -0.5, -1.0, -1.25, -1.5], dtype=float),
    }

    # 为了归一化方便，记录每个维度的 min/max（注意 ll/sdl 是降序列表，也没关系）
    lows = np.array([grid[p].min() for p in param_order], dtype=float)  # [100, 1.0, 1000, 24000, -5.0, -1.5]
    highs = np.array([grid[p].max() for p in param_order], dtype=float) # [250, 2.25, 5000, 42200, -1.5, 0.0]

    n_levels = np.array([len(grid[p]) for p in param_order], dtype=int)

    all_raw = product(*(grid[p] for p in param_order))

    X_pool_all = np.array(list(all_raw), dtype=float)

    total_pool_count = X_pool_all.shape[0]

    if verbose:
        print(f"Total pool combinations (discrete grid): {total_pool_count}")

    def to_unit_cube(X):
        X = np.asarray(X, dtype=float)
        span = highs - lows
        span[span == 0] = 1.0
        return (X - lows) / span

    def farthest_point_subsample(X01, n_keep, seed=42):
        """
        Greedy farthest point sampling on normalized space X01 (N, d).
        Returns selected indices (length n_keep).
        """
        X01 = np.asarray(X01, dtype=float)
        N = X01.shape[0]

        if n_keep >= N:
            return np.arange(N, dtype=int)
        if n_keep <= 0:
            raise ValueError("n_keep must be >= 1")

        rng = np.random.default_rng(seed)

        selected = np.empty(n_keep, dtype=int)

        # 随机起点（可复现）
        first = int(rng.integers(0, N))
        selected[0] = first

        # 每个点到“已选集合”的最小距离平方（初始化为到第一个点）
        diff = X01 - X01[first]
        min_dist2 = np.einsum("ij,ij->i", diff, diff)
        min_dist2[first] = -1.0  # 标记已选

        for i in tqdm(range(1, n_keep), desc="FPS subsampling", leave=False):
            nxt = int(np.argmax(min_dist2))
            selected[i] = nxt

            diff = X01 - X01[nxt]
            dist2 = np.einsum("ij,ij->i", diff, diff)

            min_dist2 = np.minimum(min_dist2, dist2)
            min_dist2[selected[:i+1]] = -1.0  # 已选点不再参与

        return selected

    # ---------------------------------------------
    # 5) Paper-style test sampling:
    #    LHS -> scale -> int -> valid-filter -> dedup until n_test reached
    # ---------------------------------------------
    def lhs_discrete_batch(n, seed_local):
        """
        For each dimension j with m levels:
        - sample u in [0,1)
        - idx = floor(u * m) in {0, ..., m-1}
        - map idx -> discrete value grid[param][idx]
        """
        sampler = qmc.LatinHypercube(d=len(param_order), seed=seed_local)
        U = sampler.random(n)  # shape (n, 6), values in [0,1)

        # index matrix
        idx = np.floor(U * n_levels).astype(int)
        # 数值安全（理论上 U<1，不会越界，但clip更稳）
        idx = np.clip(idx, 0, n_levels - 1)

        # map to actual discrete parameter values
        X = np.empty((n, len(param_order)), dtype=float)
        for j, p in enumerate(param_order):
            X[:, j] = grid[p][idx[:, j]]
        return X

    X_test = lhs_discrete_batch(n_test, seed_test)

    # 可选：仍然做一次去重检查（建议保留）
    if len({tuple(row.tolist()) for row in X_test}) < n_test:
        raise RuntimeError(
            "LHS sampling produced duplicate test points. "
            "For this seed/n_test, please use a different seed or add a small oversampling factor."
        )


    # ---------------------------------------------
    # 6) Compute delta from test NN distances (normalized space)
    # ---------------------------------------------
    X_test01 = to_unit_cube(X_test)
    tree_test = cKDTree(X_test01)
    d_nn, _ = tree_test.query(X_test01, k=2)  # self + nearest other
    nn_dist = d_nn[:, 1]
    delta = float(delta_factor * np.median(nn_dist))

    # ---------------------------------------------
    # 7) Filter pool:
    #    - remove test points themselves
    #    - remove points too close to test
    # ---------------------------------------------
    test_set = set(map(tuple, X_test.tolist()))
    mask_not_test = np.array([tuple(row.tolist()) not in test_set for row in X_pool_all], dtype=bool)
    X_pool_candidates = X_pool_all[mask_not_test]

    X_pool01 = to_unit_cube(X_pool_candidates)
    dist_to_test, _ = tree_test.query(X_pool01, k=1)
    mask_far = dist_to_test >= delta
    X_pool_filtered = X_pool_candidates[mask_far]

    # ---------------------------------------------
    # 7.5) Optional: limit pool size with FPS
    # ---------------------------------------------
    
    pool_subsample_method = "fps"   # 目前用 fps

    n_pool_before_subsample = X_pool_filtered.shape[0]

    if n_pool is not None and n_pool_before_subsample > n_pool:
        if pool_subsample_method == "fps":
            X_pool_filtered01 = to_unit_cube(X_pool_filtered)
            sel_idx = farthest_point_subsample(
                X_pool_filtered01,
                n_keep=n_pool,
                seed=seed_pool,
            )
            X_pool_filtered = X_pool_filtered[sel_idx]
        else:
            raise ValueError(f"Unknown pool_subsample_method: {pool_subsample_method}")
    else:
        # 没有限量或本来就不超，保留原数
        n_pool_before_subsample = X_pool_filtered.shape[0]
        
    # ---------------------------------------------
    # 8) Return DataFrames + metadata
    # ---------------------------------------------
    X_test_df = pd.DataFrame(X_test, columns=param_order)
    X_pool_filtered_df = pd.DataFrame(X_pool_filtered, columns=param_order)

    
    meta = {
        "param_order": param_order,
        "grid_sizes": {p: int(len(grid[p])) for p in param_order},
        "raw_bounds": {p: (float(grid[p].min()), float(grid[p].max())) for p in param_order},
        "pool_total_count": int(X_pool_all.shape[0]),   # expected 36288 for current grid
        "n_test": int(X_test.shape[0]),
        "delta_factor": float(delta_factor),
        "delta": float(delta),
        "test_nn_dist_median": float(np.median(nn_dist)),
        "test_nn_dist_min": float(np.min(nn_dist)),
        "test_nn_dist_max": float(np.max(nn_dist)),
        "n_pool_after_remove_test": int(X_pool_candidates.shape[0]),
        "n_pool_filtered": int(X_pool_filtered.shape[0]),
    }

    if verbose:
        print(f"Pool total count: {X_pool_all.shape[0]}")
        print(f"delta = {delta:.4f} (={delta_factor} * median NN distance in normalized space)")
        print(f"Pool after removing test points: {X_pool_candidates.shape[0]}")
        print(f"Pool kept after distance filtering: {X_pool_filtered.shape[0]} / {X_pool_candidates.shape[0]}")

    return X_test_df, X_pool_filtered_df, meta

# X_test_fea, _, _ = generation_pool_fea(seed=42, n_test=100, delta_factor=0.5, verbose=True)

#for seed in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
#    X_test_fea, X_pool_fea, meta_fea = generation_pool_fea(seed=seed, n_test=150, delta_factor=0.5, verbose=True)
#    X_test_fea.to_csv(f'../data/fea_test_{seed}.csv', index=False)
#    print(f"Seed {seed} finished. Test set size: {X_test_fea.shape[0]}, Pool size: {X_pool_fea.shape[0]}")
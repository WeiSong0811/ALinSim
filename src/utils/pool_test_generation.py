import numpy as np
from scipy.stats import qmc
from scipy.spatial import cKDTree
import pandas as pd

def generation_pool_pan(seed=42, n_test=100, n_pool=int(1e7)):
    # ----- 初始设置 -----
    #seed = 42 # 跟主动学习的随机数种子保持一致
    seed_test = seed * 2 
    seed_pool = seed

    n_test = 100
    n_pool = int(1e7)

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

import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.spatial import cKDTree
from itertools import product


def generation_pool_trc_valid(
    seed=42,
    n_test=100,
    delta_factor=0.5,
    verbose=True,
    max_test_sampling_rounds=50,
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
    param_order = ["x1", "x2", "x3", "y1", "y2", "y3"]

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

    # sanity check: should match paper count 21168
    expected_count = 21168
    if X_pool_all.shape[0] != expected_count:
        raise RuntimeError(
            f"Valid pool count mismatch: got {X_pool_all.shape[0]}, expected {expected_count}."
        )

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

    seed_test_base = seed * 2 + 1

    collected = set()
    X_test_list = []

    rounds = 0
    while len(X_test_list) < n_test and rounds < max_test_sampling_rounds:
        rounds += 1
        need = n_test - len(X_test_list)
        # oversample because validity filtering + dedup will reduce count
        batch_n = max(need * 4, 200)

        X_batch = lhs_integerized_batch(batch_n, seed_test_base + rounds - 1)
        X_batch = filter_valid(X_batch)

        for row in X_batch:
            key = tuple(row.tolist())
            if key not in collected:
                collected.add(key)
                X_test_list.append(row.copy())
                if len(X_test_list) >= n_test:
                    break

    if len(X_test_list) < n_test:
        raise RuntimeError(
            f"Could only collect {len(X_test_list)} unique valid test points "
            f"after {rounds} rounds. Increase max_test_sampling_rounds."
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
            "x1": (0, 9), "x2": (1, 9), "x3": (0, 9),
            "y1": (0, 3), "y2": (1, 3), "y3": (0, 3),
        },
        "paper_valid_pool_count": int(X_pool_all.shape[0]),  # should be 21168
        "expected_paper_count": expected_count,
        "n_test": int(X_test.shape[0]),
        "delta_factor": float(delta_factor),
        "delta": float(delta),
        "test_nn_dist_median": float(np.median(nn_dist)),
        "test_nn_dist_min": float(np.min(nn_dist)),
        "test_nn_dist_max": float(np.max(nn_dist)),
        "n_pool_after_remove_test": int(X_pool_candidates.shape[0]),
        "n_pool_filtered": int(X_pool_filtered.shape[0]),
        "test_sampling_rounds": rounds,
    }

    if verbose:
        print(f"Valid pool count (paper-consistent): {X_pool_all.shape[0]} (expected {expected_count})")
        print(f"Test set size: {X_test.shape[0]}  | sampling rounds: {rounds}")
        print(f"delta = {delta:.4f}  (={delta_factor} * median NN distance in normalized space)")
        print(f"Pool after removing test points: {X_pool_candidates.shape[0]}")
        print(f"Pool kept after distance filtering: {X_pool_filtered.shape[0]} / {X_pool_candidates.shape[0]}")

    return X_test_df, X_pool_filtered_df, meta

X_test_trc, X_pool_trc, meta_trc = generation_pool_trc_valid(seed=42, n_test=100, delta_factor=0.5)

print("shape of test set:", X_test_trc.shape)
print("shape of pool set:", X_pool_trc.shape)

print("\nSample test points:")
print(X_test_trc.head())

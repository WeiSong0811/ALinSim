import argparse
import csv
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from datetime import datetime

import numpy as np
import pandas as pd

from .src.helpers import define_input

def ensure_compat_exe_dir(sim_trc_dir: str) -> None:
    exe_dir = os.path.join(sim_trc_dir, "exeDir")
    feap_exe = os.path.join(exe_dir, "Feap86.exe")
    if not os.path.isfile(feap_exe):
        raise FileNotFoundError(f"Feap86.exe not found: {feap_exe}")

    # run_sim_trc.py uses a hard-coded relative path:
    # "..\\src\\models\\sim_trc\\exeDir" from its CWD.
    compat_exe_dir = os.path.normpath(
        os.path.join(sim_trc_dir, "..", "src", "models", "sim_trc", "exeDir")
    )
    compat_feap_exe = os.path.join(compat_exe_dir, "Feap86.exe")
    compat_icant = os.path.join(compat_exe_dir, "iCantilever.txt")

    if not os.path.isfile(compat_feap_exe):
        os.makedirs(compat_exe_dir, exist_ok=True)
        shutil.copyfile(feap_exe, compat_feap_exe)

    icant_src = os.path.join(exe_dir, "iCantilever.txt")
    if os.path.isfile(icant_src) and not os.path.isfile(compat_icant):
        shutil.copyfile(icant_src, compat_icant)


def run_one_sim(i, row, run_root, sim_trc_dir, run_sim_py, space):
    sim_dir = os.path.join(run_root, f"sim_{i}")
    os.makedirs(sim_dir, exist_ok=True)

    x_input = define_input(row, space)
    np.save(os.path.join(sim_dir, "search_space.npy"), x_input)

    result = subprocess.run(
        [sys.executable, run_sim_py, f"-f{os.path.abspath(sim_dir)}"],
        cwd=sim_trc_dir,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    row_list = row.tolist()
    errors = []
    if result.returncode != 0:
        errors.append(
            {
                "index": i,
                "row": row_list,
                "returncode": result.returncode,
                "stderr": (result.stderr or "").strip(),
            }
        )

    force_path = os.path.join(sim_dir, "PCantilevera.sum")
    if os.path.isfile(force_path):
        force = pd.read_csv(force_path, names=["time", "f1", "f2", "f3"], sep=r"\s+")
        force["f_res"] = (force[["f1", "f2", "f3"]] ** 2).sum(axis=1) ** 0.5
        y_i = float(force["f_res"].max()) if len(force) == 101 else 0.0
    else:
        errors.append(
            {
                "index": i,
                "row": row_list,
                "returncode": result.returncode,
                "stderr": "Missing PCantilevera.sum",
            }
        )
        y_i = 0.0

    return i, row_list, y_i, errors

def predict_with_retry(
    X_candidates,
    idx_candidates,
    target_valid=5,
    init_parallel=5,
    seed=42,
):
    """
    X_candidates: shape (N, 6), 建议 N=15
    idx_candidates: 长度N，对应每个样本的原始idx
    返回:
      valid_y: 长度=target_valid(若候选不足可能小于)
      consumed_idx: 实际调用过tool的idx（成功+失败）
      valid_idx: 有效y对应的idx
    """
    X_arr = np.asarray(X_candidates)
    if X_arr.ndim != 2 or X_arr.shape[1] != 6:
        raise ValueError(f"X_candidates must be (N, 6), got {X_arr.shape}.")
    if len(idx_candidates) != len(X_arr):
        raise ValueError("idx_candidates length must match X_candidates rows.")

    space = [
        {"name": "x_0", "domain": (1, 9), "type": "integer"},
        {"name": "x_1", "domain": (1, 9), "type": "integer"},
        {"name": "x_2", "domain": (1, 9), "type": "integer"},
        {"name": "y_0", "domain": (0, 3), "type": "integer"},
        {"name": "y_1", "domain": (1, 3), "type": "integer"},
        {"name": "y_2", "domain": (0, 3), "type": "integer"},
    ]
    use_case = "SIM-TRC"
    sim_trc_dir = os.path.join("./run_trc/src")
    run_sim_py = "run_sim_trc.py"
    ensure_compat_exe_dir(sim_trc_dir)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join("workdir", "sim_trc_runs", f"{use_case}_{seed}_{run_tag}")
    os.makedirs(run_root, exist_ok=True)

    valid_y = []
    valid_idx = []
    consumed_idx = []

    next_pos = 0
    sim_counter = 0
    future_to_meta = {}  # future -> (cand_pos, original_idx)

    def submit_one(executor, cand_pos):
        nonlocal sim_counter
        fut = executor.submit(
            run_one_sim,
            sim_counter,                 # 用独立sim id创建sim目录
            X_arr[cand_pos],
            run_root,
            sim_trc_dir,
            run_sim_py,
            space,
        )
        future_to_meta[fut] = (cand_pos, idx_candidates[cand_pos])
        sim_counter += 1

    max_workers = max(1, min(init_parallel, os.cpu_count() or 1, len(X_arr)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 初始先跑满5个（或候选不足5个）
        for _ in range(min(init_parallel, len(X_arr))):
            submit_one(executor, next_pos)
            next_pos += 1

        # 任意一个返回就判断一次；失败立刻补1个
        while future_to_meta and len(valid_y) < target_valid:
            done, _ = wait(list(future_to_meta.keys()), return_when=FIRST_COMPLETED)

            for fut in done:
                _, idx = future_to_meta.pop(fut)
                _, _, y_i, _ = fut.result()
                consumed_idx.append(idx)

                if y_i != 0.0:
                    valid_y.append(float(y_i))
                    valid_idx.append(idx)
                    if len(valid_y) >= target_valid:
                        break
                else:
                    if next_pos < len(X_arr):
                        submit_one(executor, next_pos)
                        next_pos += 1

    return valid_y, consumed_idx, valid_idx

def predict(X, seed=42):
    X_arr = np.asarray(X)
    # Support both a single sample [x0..x5] and batch samples [[...], [...]]
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
        single_input = True
    elif X_arr.ndim == 2:
        single_input = False
    else:
        raise ValueError(f"X must be 1D or 2D, got shape {X_arr.shape}.")

    if X_arr.shape[1] != 6:
        raise ValueError(f"Each sample must have 6 values, got shape {X_arr.shape}.")

    X_df = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(6)])
    space = [
        {"name": "x_0", "domain": (1, 9), "type": "integer"},
        {"name": "x_1", "domain": (1, 9), "type": "integer"},
        {"name": "x_2", "domain": (1, 9), "type": "integer"},
        {"name": "y_0", "domain": (0, 3), "type": "integer"},
        {"name": "y_1", "domain": (1, 3), "type": "integer"},
        {"name": "y_2", "domain": (0, 3), "type": "integer"},
    ]
    use_case = "SIM-TRC"
    sim_trc_dir = os.path.join("./run_trc/src")
    run_sim_py = "run_sim_trc.py"
    ensure_compat_exe_dir(sim_trc_dir)
    seed = seed
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join("workdir", "sim_trc_runs", f"{use_case}_{seed}_{run_tag}")
    os.makedirs(run_root, exist_ok=True)

    n_samples = len(X_df)
    y_values = [0.0] * n_samples

    if n_samples == 1:
        _, _, y, _, = run_one_sim(
            0,
            X_df.iloc[0].values,
            run_root,
            sim_trc_dir,
            run_sim_py,
            space,
        )
        y_values[0] = y
    else:
        max_workers = min(n_samples, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    run_one_sim,
                    i,
                    row.values,
                    run_root,
                    sim_trc_dir,
                    run_sim_py,
                    space,
                )
                for i, (_, row) in enumerate(X_df.iterrows())
            ]
            for future in as_completed(futures):
                i, _, y, _ = future.result()
                y_values[i] = y

    if single_input:
        return y_values[0]
    return y_values

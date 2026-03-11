import argparse
import csv
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

from src.helpers import define_input


def parse_args():
    parser = argparse.ArgumentParser(description="Run SIM-TRC simulations in parallel.")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of parallel workers, e.g. 10 or 20.")
    parser.add_argument("--seed", type=int, default=40, help="Seed to run.")
    return parser.parse_args()


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


def run_one_sim(i, row, seed, idx_width, per_row_dir, header, run_root, sim_trc_dir, run_sim_py, space):
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

    single_out_name = f"{seed}_{i:0{idx_width}d}.csv"
    single_out_path = os.path.join(per_row_dir, single_out_name)
    with open(single_out_path, "w", newline="") as f_single:
        writer = csv.writer(f_single)
        writer.writerow(header)
        writer.writerow(row_list + [y_i])

    return i, row_list, y_i, errors

def main():
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    use_case = "SIM-TRC"
    out_dir = "predicted_datasets1"
    per_row_dir = os.path.join(out_dir, "sim_trc_rows")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(per_row_dir, exist_ok=True)

    space = [
        {"name": "x_0", "domain": (1, 9), "type": "integer"},
        {"name": "x_1", "domain": (1, 9), "type": "integer"},
        {"name": "x_2", "domain": (1, 9), "type": "integer"},
        {"name": "y_0", "domain": (0, 3), "type": "integer"},
        {"name": "y_1", "domain": (1, 3), "type": "integer"},
        {"name": "y_2", "domain": (0, 3), "type": "integer"},
    ]

    sim_trc_dir = os.path.join("src")
    run_sim_py = "run_sim_trc.py"
    ensure_compat_exe_dir(sim_trc_dir)
    
    seed = args.seed  # For now, we run one seed at a time. Can be extended to loop over seeds.

    x_path = f"./data/trc_test_seed{seed}.csv"
    df = pd.read_csv(x_path, header=0).apply(pd.to_numeric, errors="raise")
    x = df.values
    n = x.shape[0]

    print(f"[SEED {seed}] Loaded X: {x.shape}")
    print(f"[SEED {seed}] Start {n} runs with max_workers={args.max_workers}")

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join("workdir", "sim_trc_runs", f"{use_case}_{seed}_{run_tag}")
    os.makedirs(run_root, exist_ok=True)

    out_path = os.path.join(out_dir, f"sim_trc_{seed}.csv")
    results = [None] * n
    error_rows = []
    header = list(df.columns) + ["f_res"]
    idx_width = max(3, len(str(max(0, n - 1))))

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {
            executor.submit(
                run_one_sim,
                i,
                row,
                seed,
                idx_width,
                per_row_dir,
                header,
                run_root,
                sim_trc_dir,
                run_sim_py,
                space,
            ): i
            for i, row in enumerate(x)
        }

        done_count = 0
        for future in as_completed(future_to_idx):
            idx, row_list, y_i, errors = future.result()
            results[idx] = row_list + [y_i]
            for err in errors:
                err["seed"] = seed
                error_rows.append(err)

            done_count += 1
            if done_count % 10 == 0 or done_count == n:
                print(f"[SEED {seed}] done {done_count}/{n}")

    with open(out_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        writer.writerows(results)

    print(f"[OK] Saved: {out_path}  shape=({n}, {df.shape[1] + 1})")
    print(f"[OK] Per-row CSV dir: {per_row_dir}")

    if error_rows:
        err_path = os.path.join(out_dir, f"sim_trc_errors.csv")
        pd.DataFrame(error_rows).to_csv(err_path, index=False)
        print(f"[WARN] Saved errors: {err_path}  count={len(error_rows)}")


if __name__ == "__main__":
    main()

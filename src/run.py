import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

# Simple parameter grid and named presets (dict-based lookup)
PARAM_GRID = {
    "random_state": [40, 41, 42, 43, 44],
    "dataset": ["FEA"],
    "strategy": ["RandomSearch", "GSBAG", 'QueryByCommittee', 'QDD', 'GSi', 'GSx', 'GSy', 'GaussianProcessBased',],
    "initial_method": ["random",'greedy_search', 'kmeans', 'ncc'],
    "n_pro_query": [10],
    "threshold": [0.85],
}
'''
--strategy:
    RandomSearch
    GSBAG
    QueryByCommittee
    TreeBasedRegressor_Diversity
    TreeBasedRegressor_Representativity
    TreeBasedRegressor_Diversity_self
    TreeBasedRegressor_Representativity_self
    GaussianProcessBased
    QDD
    GSi
    GSx
    GSy
    LearningLoss
    EGAL
    mcdropout
    Basic_RD_ALR
    RD_GS_ALR
    RD_QBC_ALR
    RD_EMCM_ALR
    BMDAL

--initial-method:
    random
    greedy_search
    kmeans
    ncc
'''




# Optional presets to run a curated set of experiments
PRESETS = {
    "quick": [
        {"random_state": 42, "dataset": "uci-concrete", "strategy": "RandomSearch", "initial_method": "random", "n_pro_query": 10, "threshold": 0.85},
        {"random_state": 42, "dataset": "uci-concrete", "strategy": "GSBAG", "initial_method": "random", "n_pro_query": 10, "threshold": 0.85},
    ],
}


def build_grid_runs(grid):
    keys = list(grid.keys())
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def build_preset_runs(name):
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    for cfg in PRESETS[name]:
        yield cfg


def run_one(cfg, dry_run=False):
    src_dir = Path(__file__).resolve().parent
    main_py = src_dir / "main.py"
    cmd = [
        sys.executable,
        str(main_py),
        "--random-state", str(cfg["random_state"]),
        "--dataset", cfg["dataset"],
        "--strategy", cfg["strategy"],
        "--initial-method", cfg["initial_method"],
        "--n-pro-query", str(cfg["n_pro_query"]),
        "--threshold", str(cfg["threshold"]),
    ]
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return 0
    print("RUN:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=src_dir)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run multiple AL experiments via grid or preset enumeration.")
    parser.add_argument("--mode", choices=["grid", "preset"], default="grid")
    parser.add_argument("--preset", default="quick", help="Preset name when mode=preset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mode == "grid":
        runs = list(build_grid_runs(PARAM_GRID))
    else:
        runs = list(build_preset_runs(args.preset))

    for i, cfg in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {cfg}")
        code = run_one(cfg, dry_run=args.dry_run)
        if code != 0:
            print(f"Run failed with exit code {code}. Stopping.")
            sys.exit(code)


if __name__ == "__main__":
    main()

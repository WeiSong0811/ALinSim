#!/usr/bin/env python3
"""
Batch runner for active learning strategies on CSV datasets.

Scans CSV files under a data directory, runs one or more strategies, and saves
results into output/<strategy_name>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
STRATEGIES_DIR = SRC_DIR / "strategies"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import active_learning, suggest_next_batch  # noqa: E402

# Edit this block instead of passing CLI arguments.
EMBEDDED_CONFIG: Dict[str, Any] = {
    # "csv" or "search-space"
    "mode": "search-space",
    "data_dir": "data",
    # Use target_columns for multi-target; keep target_column for single-target compatibility.
    "target_columns": ["wca", "q", "sigma"],
    "feature_columns": None,  # e.g. "x1,x2,x3" for csv mode
    "methods": "all",  # or "RandomSearch,GSBAG"
    "output_dir": "output",

    # search-space mode
    "search_space_json": "data/search_space.json",
    "labeled_csv": "data/seed_10_samples.csv",
    "search_rounds": 1,

    # shared run settings
    "random_state": 42,
    "initial_method": "random",  # random, greedy_search, kmeans, ncc
    "n_initial": 500,
    "n_pro_query": 10,
    "test_size": 0.2,
    "threshold": 0.85,
}


def ensure_namespace_package(name: str, path: Path) -> None:
    """Create a lightweight namespace package in sys.modules if needed."""
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg


def load_strategy_class(module_file: str, class_name: str):
    """
    Load a strategy class from src/strategies without importing strategies/__init__.py.
    """
    ensure_namespace_package("strategies", STRATEGIES_DIR)
    module_stem = Path(module_file).stem
    module_name = f"strategies.{module_stem}"
    module_path = STRATEGIES_DIR / module_file

    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return getattr(module, class_name)


def create_kernel():
    return (C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) +
            WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)))


def get_strategy_specs() -> List[Dict]:
    return [
        {"name": "RandomSearch", "module": "randomsearch.py", "class": "RandomSearch",
         "factory": lambda rs, k: {"random_state": rs}},
        {"name": "GSBAG", "module": "GSBAG.py", "class": "GSBAG",
         "factory": lambda rs, k: {"random_state": rs, "kernel": k, "n_restarts_optimizer": 10}},
        {"name": "QueryByCommittee", "module": "qbc.py", "class": "QueryByCommittee",
         "factory": lambda rs, k: {"random_state": rs, "num_learner": 100}},
        {"name": "TreeBasedRegressor_Diversity", "module": "treebased_diversity.py",
         "class": "TreeBasedRegressor_Diversity", "factory": lambda rs, k: {"random_state": rs, "min_samples_leaf": 5}},
        {"name": "TreeBasedRegressor_Representativity", "module": "treebased_representativity.py",
         "class": "TreeBasedRegressor_Representativity",
         "factory": lambda rs, k: {"random_state": rs, "min_samples_leaf": 5}},
        {"name": "TreeBasedRegressor_Diversity_self", "module": "treebased_diversity_self.py",
         "class": "TreeBasedRegressor_Diversity_self", "factory": lambda rs, k: {"random_state": rs, "min_samples_leaf": 5}},
        {"name": "TreeBasedRegressor_Representativity_self", "module": "treebased_representativity_self.py",
         "class": "TreeBasedRegressor_Representativity_self",
         "factory": lambda rs, k: {"random_state": rs, "min_samples_leaf": 5}},
        {"name": "GaussianProcessBased", "module": "gaussianprocess.py", "class": "GaussianProcessBased",
         "factory": lambda rs, k: {"random_state": rs, "kernel": k, "n_restarts_optimizer": 10}},
        {"name": "QDD", "module": "QDD.py", "class": "QDD",
         "factory": lambda rs, k: {"random_state": rs}},
        {"name": "GSi", "module": "gsi.py", "class": "GSi",
         "factory": lambda rs, k: {"random_state": rs}},
        {"name": "GSx", "module": "gsx.py", "class": "GSx",
         "factory": lambda rs, k: {"random_state": rs}},
        {"name": "GSy", "module": "gsy.py", "class": "GSy",
         "factory": lambda rs, k: {"random_state": rs}},
        {"name": "LearningLoss", "module": "LL4AL.py", "class": "LearningLoss",
         "factory": lambda rs, k: {
             "BATCH": 16, "LR": 0.01, "MARGIN": 1, "WEIGHT": 0.0001,
             "EPOCH": 200, "EPOCHL": 75, "WDECAY": 5e-4, "random_state": rs
         }},
        {"name": "EGAL", "module": "egal.py", "class": "EGAL",
         "factory": lambda rs, k: {"b_factor": 0.25, "random_state": rs}},
        {"name": "mcdropout", "module": "mcd.py", "class": "mcdropout",
         "factory": lambda rs, k: {"random_state": rs, "learning_rate": 0.01, "num_epochs": 200, "batch_size": 16}},
        {"name": "Basic_RD_ALR", "module": "RDALR.py", "class": "Basic_RD_ALR",
         "factory": lambda rs, k: {"random_state": rs}},
        {"name": "RD_GS_ALR", "module": "RDGS.py", "class": "RD_GS_ALR",
         "factory": lambda rs, k: {"random_state": rs}},
        {"name": "RD_QBC_ALR", "module": "RDQBC.py", "class": "RD_QBC_ALR",
         "factory": lambda rs, k: {"random_state": rs, "num_learner": 100}},
        {"name": "RD_EMCM_ALR", "module": "RDEMCM.py", "class": "RD_EMCM_ALR",
         "factory": lambda rs, k: {"random_state": rs}},
        {"name": "BMDAL", "module": "bmdal.py", "class": "BMDAL",
         "factory": lambda rs, k: {"random_state": rs, "selection_method": "lcmd"}},
    ]


def discover_csv_files(data_dir: Path) -> List[Path]:
    return sorted(p for p in data_dir.rglob("*.csv") if p.is_file())


def parse_feature_columns(text: str | None) -> List[str] | None:
    if not text:
        return None
    cols = [c.strip() for c in text.split(",") if c.strip()]
    return cols or None


def normalize_target_columns(cfg: argparse.Namespace) -> List[str]:
    """
    Resolve target columns from embedded config.
    Supports:
    - target_columns: list[str]
    - target_column: str (legacy single-target)
    """
    if hasattr(cfg, "target_columns") and cfg.target_columns is not None:
        if not isinstance(cfg.target_columns, list) or not cfg.target_columns:
            raise ValueError("'target_columns' must be a non-empty list of column names.")
        cols = [str(c).strip() for c in cfg.target_columns if str(c).strip()]
        if not cols:
            raise ValueError("'target_columns' contains no valid column names.")
        return cols

    if hasattr(cfg, "target_column") and cfg.target_column is not None:
        if isinstance(cfg.target_column, list):
            cols = [str(c).strip() for c in cfg.target_column if str(c).strip()]
            if not cols:
                raise ValueError("'target_column' list contains no valid column names.")
            return cols
        col = str(cfg.target_column).strip()
        if not col:
            raise ValueError("'target_column' cannot be empty.")
        return [col]

    raise ValueError("Please set 'target_columns' (preferred) or 'target_column' in EMBEDDED_CONFIG.")


def load_search_space(search_space_json: Path) -> Dict[str, Any]:
    with open(search_space_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Search space JSON must be a non-empty object.")

    # Normalize feature names to avoid trailing/leading whitespace mismatches.
    search_space: Dict[str, Any] = {}
    for key, spec in raw.items():
        clean_key = str(key).strip()
        if not clean_key:
            raise ValueError("Search space contains an empty feature name after stripping whitespace.")
        if clean_key in search_space:
            raise ValueError(
                f"Duplicate feature name after stripping whitespace: '{clean_key}'. "
                "Please fix your JSON keys."
            )
        search_space[clean_key] = spec
    return search_space


def sample_dimension(rng: np.random.Generator, spec: Any, n: int) -> np.ndarray:
    """
    Sample one feature from search-space spec.

    Supported spec forms:
    - [v1, v2, ...] (categorical/discrete choices)
    - {"values": [v1, v2, ...]}
    - {"min": a, "max": b, "type": "float"|"int", "step": s(optional)}
    - {"min": a, "max": b, "count": n, "type": "float"|"int"}
    """
    if isinstance(spec, list):
        if not spec:
            raise ValueError("Feature spec list cannot be empty.")
        return rng.choice(np.array(spec, dtype=object), size=n, replace=True)

    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported feature spec: {spec}")

    if "values" in spec:
        values = spec["values"]
        if not isinstance(values, list) or not values:
            raise ValueError("'values' must be a non-empty list.")
        return rng.choice(np.array(values, dtype=object), size=n, replace=True)

    if "min" in spec and "max" in spec:
        low = spec["min"]
        high = spec["max"]
        if low > high:
            raise ValueError(f"Invalid range: min ({low}) > max ({high}).")
        dtype = str(spec.get("type", "float")).lower()
        step = spec.get("step")
        count = spec.get("count")

        if count is not None:
            count = int(count)
            if count <= 0:
                raise ValueError("'count' must be > 0.")
            if dtype == "int":
                values = np.rint(np.linspace(int(low), int(high), num=count)).astype(int)
                values = np.unique(values)
            else:
                values = np.linspace(float(low), float(high), num=count)
            return rng.choice(values, size=n, replace=True)

        if dtype == "int":
            if step is None:
                return rng.integers(int(low), int(high) + 1, size=n)
            values = np.arange(int(low), int(high) + 1, int(step))
            if values.size == 0:
                raise ValueError("No integer values generated for given min/max/step.")
            return rng.choice(values, size=n, replace=True)

        if step is None:
            return rng.uniform(float(low), float(high), size=n)

        values = np.arange(float(low), float(high) + float(step) * 0.5, float(step))
        if values.size == 0:
            raise ValueError("No float values generated for given min/max/step.")
        return rng.choice(values, size=n, replace=True)

    raise ValueError(f"Unsupported feature spec: {spec}")


def expand_dimension_values(spec: Any) -> List[Any]:
    """
    Expand one feature spec into a finite list of values for exhaustive mode.
    """
    if isinstance(spec, list):
        if not spec:
            raise ValueError("Feature spec list cannot be empty.")
        return list(spec)

    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported feature spec: {spec}")

    if "values" in spec:
        values = spec["values"]
        if not isinstance(values, list) or not values:
            raise ValueError("'values' must be a non-empty list.")
        return list(values)

    if "min" in spec and "max" in spec:
        low = spec["min"]
        high = spec["max"]
        if low > high:
            raise ValueError(f"Invalid range: min ({low}) > max ({high}).")
        dtype = str(spec.get("type", "float")).lower()
        count = spec.get("count")
        if count is None:
            raise ValueError(
                "Exhaustive traversal requires finite values. "
                "For range specs, please provide 'count'."
            )
        count = int(count)
        if count <= 0:
            raise ValueError("'count' must be > 0.")

        if dtype == "int":
            values = np.rint(np.linspace(int(low), int(high), num=count)).astype(int)
            values = np.unique(values)
            if values.size == 0:
                raise ValueError("No integer values generated for given min/max/count.")
            return [int(v) for v in values.tolist()]

        values = np.linspace(float(low), float(high), num=count)
        if values.size == 0:
            raise ValueError("No float values generated for given min/max/count.")
        return [float(v) for v in values.tolist()]

    raise ValueError(f"Unsupported feature spec: {spec}")


def generate_pool_from_search_space(
    search_space: Dict[str, Any], pool_size: int, random_state: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed=random_state)
    data = {}
    for feature, spec in search_space.items():
        data[feature] = sample_dimension(rng, spec, pool_size)
    return pd.DataFrame(data)


def generate_exhaustive_pool_from_search_space(search_space: Dict[str, Any]) -> pd.DataFrame:
    """
    Build full Cartesian product pool from search space.
    """
    feature_names = list(search_space.keys())
    value_lists: List[List[Any]] = [expand_dimension_values(search_space[f]) for f in feature_names]

    rows = itertools.product(*value_lists)
    return pd.DataFrame(rows, columns=feature_names)


def prepare_dataset(
    csv_path: Path,
    target_columns: List[str],
    feature_columns: List[str] | None,
    random_state: int,
    test_size: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    target_columns = [str(c).strip() for c in target_columns]
    if feature_columns is not None:
        feature_columns = [str(c).strip() for c in feature_columns]

    missing_targets = [c for c in target_columns if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found: {missing_targets}")

    used_features = feature_columns or [c for c in df.columns if c not in target_columns]
    missing = [c for c in used_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    clean_df = df.dropna(subset=used_features + target_columns).copy()
    if clean_df.empty:
        raise ValueError("No rows remain after dropping NaN values.")

    X = clean_df[used_features]
    y = clean_df[target_columns]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=used_features, index=X.index)

    X_t, X_val, y_t, y_val = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    X_t = X_t.reset_index(drop=True)
    y_t = y_t.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    return X_t, X_val, y_t, y_val, used_features


def prepare_seed_labeled(
    labeled_csv: Path,
    target_columns: List[str],
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(labeled_csv)
    df.columns = [str(c).strip() for c in df.columns]
    target_columns = [str(c).strip() for c in target_columns]
    feature_columns = [str(c).strip() for c in feature_columns]

    required_cols = feature_columns + target_columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Labeled CSV missing columns: {missing}")
    df = df.dropna(subset=required_cols).copy()
    if df.empty:
        raise ValueError("No valid labeled rows remain after dropping NaN.")
    X = df[feature_columns].reset_index(drop=True)
    y = df[target_columns].reset_index(drop=True)
    return X, y


def safe_name(path: Path) -> str:
    return path.with_suffix("").as_posix().replace("/", "__")


def run_one(
    estimator,
    X_t: pd.DataFrame,
    y_t: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    n_initial: int,
    n_pro_query: int,
    n_queries: int,
    threshold: float,
    initial_method: str,
    random_state: int,
):
    return active_learning(
        [estimator], X_t, y_t, X_val, y_val,
        n_initial, n_pro_query, n_queries, threshold,
        initial_method=initial_method,
        test_methods=["normal"],
        random_state=random_state,
        record_metrics=True,
    )


def run_search_space_mode(
    args: argparse.Namespace,
    target_columns: List[str],
    strategy_specs: List[Dict[str, Any]],
    kernel,
    output_dir: Path,
    run_log: Dict[str, Any],
) -> None:
    search_space_json = (ROOT_DIR / args.search_space_json).resolve()
    labeled_csv = (ROOT_DIR / args.labeled_csv).resolve()
    search_space = load_search_space(search_space_json)
    feature_columns = list(search_space.keys())

    X_seed, y_seed = prepare_seed_labeled(
        labeled_csv=labeled_csv,
        target_columns=target_columns,
        feature_columns=feature_columns,
    )
    if len(X_seed) < args.n_initial:
        raise ValueError(
            f"Seed labeled rows ({len(X_seed)}) < n_initial ({args.n_initial}). "
            "Please provide at least n_initial labeled samples."
        )

    # Search-space mode always traverses full Cartesian combinations.
    pool_raw = generate_exhaustive_pool_from_search_space(search_space=search_space)

    # Remove exact duplicates from seed to avoid immediate re-query.
    seed_tuples = set(map(tuple, X_seed[feature_columns].to_numpy().tolist()))
    pool_raw = pool_raw[
        ~pool_raw[feature_columns].apply(lambda r: tuple(r.to_numpy()), axis=1).isin(seed_tuples)
    ].reset_index(drop=True)

    for spec in strategy_specs:
        method_name = spec["name"]
        method_dir = output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        try:
            strategy_cls = load_strategy_class(spec["module"], spec["class"])
            estimator = strategy_cls(**spec["factory"](args.random_state, kernel))
        except Exception as e:
            run_log["failed"].append({
                "method": method_name,
                "dataset": "search_space",
                "reason": f"strategy_load_or_init_failed: {e}",
            })
            continue

        X_labeled_raw = X_seed.copy()
        y_labeled = y_seed.copy()
        X_pool_raw = pool_raw.copy()

        rounds: List[Dict[str, Any]] = []
        if not X_pool_raw.empty:
            scaler = StandardScaler()
            scaler.fit(pd.concat([X_labeled_raw, X_pool_raw], axis=0))
            X_labeled_scaled = pd.DataFrame(
                scaler.transform(X_labeled_raw), columns=feature_columns, index=X_labeled_raw.index
            )
            X_pool_scaled = pd.DataFrame(
                scaler.transform(X_pool_raw), columns=feature_columns, index=X_pool_raw.index
            )

            selected_idx = suggest_next_batch(
                estimator=estimator,
                X_labeled=X_labeled_scaled,
                y_labeled=y_labeled,
                X_unlabeled=X_pool_scaled,
                n_pro_query=args.n_pro_query,
                random_state=args.random_state,
            )

            if selected_idx:
                selected_raw = X_pool_raw.loc[selected_idx].copy()
                rounds.append({
                    "round": 1,
                    "selected_indices": [int(i) for i in selected_idx],
                    "selected_candidates": selected_raw.to_dict(orient="records"),
                })
                X_pool_raw = X_pool_raw.drop(index=selected_idx).reset_index(drop=True)

        out_file = method_dir / "search_space_results.json"
        result = {
            "mode": "search_space",
            "method": method_name,
            "search_space_json": str(search_space_json),
            "labeled_seed_csv": str(labeled_csv),
            "target_columns": target_columns,
            "feature_columns": feature_columns,
            "total_candidates": int(len(pool_raw)),
            "search_rounds": 1,
            "n_pro_query": args.n_pro_query,
            "random_state": args.random_state,
            "rounds": rounds,
            "final_labeled_size": int(len(X_labeled_raw)),
            "remaining_pool_size": int(len(X_pool_raw)),
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        run_log["success"].append({
            "method": method_name,
            "dataset": "search_space",
            "output_file": str(out_file),
        })


def main() -> None:
    args = argparse.Namespace(**EMBEDDED_CONFIG)
    target_columns = normalize_target_columns(args)

    data_dir = (ROOT_DIR / args.data_dir).resolve()
    output_dir = (ROOT_DIR / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_methods = None
    if args.methods.strip().lower() != "all":
        requested_methods = {m.strip() for m in args.methods.split(",") if m.strip()}

    feature_columns = parse_feature_columns(args.feature_columns)
    kernel = create_kernel()
    strategy_specs = get_strategy_specs()

    if requested_methods is not None:
        strategy_specs = [s for s in strategy_specs if s["name"] in requested_methods]
        if not strategy_specs:
            raise ValueError("No valid methods selected. Check --methods names.")

    run_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "target_columns": target_columns,
        "success": [],
        "failed": [],
        "skipped": [],
    }

    if args.mode == "search-space":
        if not args.search_space_json:
            raise ValueError("--search-space-json is required in search-space mode.")
        if not args.labeled_csv:
            raise ValueError("--labeled-csv is required in search-space mode.")
        run_search_space_mode(
            args=args,
            target_columns=target_columns,
            strategy_specs=strategy_specs,
            kernel=kernel,
            output_dir=output_dir,
            run_log=run_log,
        )
        log_file = output_dir / "run_summary.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(run_log, f, indent=2, ensure_ascii=False)
        print(f"Done. Summary written to: {log_file}")
        print(f"Success: {len(run_log['success'])}, Failed: {len(run_log['failed'])}, Skipped: {len(run_log['skipped'])}")
        return

    csv_files = discover_csv_files(data_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    for spec in strategy_specs:
        method_name = spec["name"]
        method_dir = output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        try:
            strategy_cls = load_strategy_class(spec["module"], spec["class"])
            estimator = strategy_cls(**spec["factory"](args.random_state, kernel))
        except Exception as e:
            run_log["failed"].append({
                "method": method_name,
                "dataset": None,
                "reason": f"strategy_load_or_init_failed: {e}",
            })
            continue

        for csv_file in csv_files:
            rel = csv_file.relative_to(data_dir)
            dataset_key = safe_name(rel)
            out_file = method_dir / f"{dataset_key}.json"

            try:
                X_t, X_val, y_t, y_val, used_features = prepare_dataset(
                    csv_file=csv_file,
                    target_columns=target_columns,
                    feature_columns=feature_columns,
                    random_state=args.random_state,
                    test_size=args.test_size,
                )

                total_data_volume = X_t.shape[0]
                if total_data_volume <= args.n_initial:
                    run_log["skipped"].append({
                        "method": method_name,
                        "dataset": str(rel),
                        "reason": f"not_enough_train_samples: {total_data_volume} <= n_initial({args.n_initial})",
                    })
                    continue

                n_queries = (total_data_volume - args.n_initial) // args.n_pro_query
                if n_queries <= 0:
                    run_log["skipped"].append({
                        "method": method_name,
                        "dataset": str(rel),
                        "reason": "n_queries_computed_as_0",
                    })
                    continue

                query_idx_all, query_time_all, metrics_all = run_one(
                    estimator=estimator,
                    X_t=X_t,
                    y_t=y_t,
                    X_val=X_val,
                    y_val=y_val,
                    n_initial=args.n_initial,
                    n_pro_query=args.n_pro_query,
                    n_queries=n_queries,
                    threshold=args.threshold,
                    initial_method=args.initial_method,
                    random_state=args.random_state,
                )

                result = {
                    "method": method_name,
                    "dataset_file": str(rel),
                    "target_columns": target_columns,
                    "feature_columns": used_features,
                    "random_state": args.random_state,
                    "initial_method": args.initial_method,
                    "n_initial": args.n_initial,
                    "n_pro_query": args.n_pro_query,
                    "n_queries": n_queries,
                    "threshold": args.threshold,
                    "query_idx_all": query_idx_all,
                    "query_time_all": query_time_all,
                    "metrics_all": metrics_all,
                }

                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                run_log["success"].append({
                    "method": method_name,
                    "dataset": str(rel),
                    "output_file": str(out_file),
                })
            except Exception as e:
                run_log["failed"].append({
                    "method": method_name,
                    "dataset": str(rel),
                    "reason": str(e),
                })

    log_file = output_dir / "run_summary.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, ensure_ascii=False)

    print(f"Done. Summary written to: {log_file}")
    print(f"Success: {len(run_log['success'])}, Failed: {len(run_log['failed'])}, Skipped: {len(run_log['skipped'])}")


if __name__ == "__main__":
    main()

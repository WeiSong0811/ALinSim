#!/usr/bin/env python3
"""
Active Learning Benchmark for Materials Science

This script runs active learning benchmarks on various materials datasets
using different query strategies for regression tasks.

Author: [Jinghou Bi]
Email: [jinghou.bi@tu-dresden.de]
License: [MIT License]

Usage example:
    python run_benchmark.py \
        --random-state 42 \
        --dataset uci-concrete \
        --strategy RD_EMCM_ALR \
        --initial-method random \
        --n-pro-query 10 \
        --threshold 0.85
"""

import argparse
import json
import logging
import os
import random
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import data_process_meta, active_learning, suggest_next_batch
from strategies import (
    TreeBasedRegressor_Diversity, TreeBasedRegressor_Representativity, GaussianProcessBased,
    QueryByCommittee, Basic_RD_ALR, GSBAG, QDD, RandomSearch, GSi, LearningLoss, BMDAL,
    RD_GS_ALR, RD_QBC_ALR, RD_EMCM_ALR, EGAL, GSx, GSy, mcdropout,
    TreeBasedRegressor_Diversity_self, TreeBasedRegressor_Representativity_self
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('active_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")


class ActiveLearningBenchmark:
    """Main class for running active learning benchmarks."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the benchmark with configuration.

        Args:
            config: Dictionary containing benchmark configuration
        """
        self.config = config
        self.random_state = config['random_state']
        self.initial_method = config['initial_method']
        self.strategy_name = config['strategy']        # now string
        self.dataset_name = config['dataset']          # now string
        self.n_pro_query = config['n_pro_query']

        # Set up paths
        self.result_path = self._setup_result_paths()

        # Initialize kernel and other parameters
        self.kernel = self._create_kernel()
        self.threshold = config.get('threshold', 0.85)

        # Set random seed for reproducibility
        self._set_seed(self.random_state)

        logger.info(f"Initialized benchmark with config: {config}")

    def _setup_result_paths(self) -> str:
        """Create and return result directory paths."""
        result_path = os.path.join(
            "..", "result",
            str(self.n_pro_query),
            str(self.random_state),
            self.initial_method
        )

        time_record_path = os.path.join(result_path, 'time_record')

        for path in [result_path, time_record_path]:
            os.makedirs(path, exist_ok=True)

        logger.info(f"Result path: {result_path}")
        logger.info(f"Time record path: {time_record_path}")

        return result_path

    def _create_kernel(self) -> object:
        """Create and return the Gaussian process kernel."""
        return (C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) +
                WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)))

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set random seed to {seed}")

    def _create_strategy_registry(self) -> Dict[str, object]:
        """
        Build a registry mapping strategy names (strings) to initialized estimators.
        Matching will be case-insensitive when selecting.
        """
        rs = self.random_state
        k = self.kernel

        registry = {
            # keep the keys equal to class/alias names you actually use
            'RandomSearch': RandomSearch(random_state=rs),
            'GSBAG': GSBAG(random_state=rs, kernel=k, n_restarts_optimizer=10),
            'QueryByCommittee': QueryByCommittee(random_state=rs, num_learner=100),
            'TreeBasedRegressor_Diversity': TreeBasedRegressor_Diversity(random_state=rs, min_samples_leaf=5),
            'TreeBasedRegressor_Representativity': TreeBasedRegressor_Representativity(random_state=rs, min_samples_leaf=5),
            'TreeBasedRegressor_Diversity_self': TreeBasedRegressor_Diversity_self(random_state=rs, min_samples_leaf=5),
            'TreeBasedRegressor_Representativity_self': TreeBasedRegressor_Representativity_self(random_state=rs, min_samples_leaf=5),
            'GaussianProcessBased': GaussianProcessBased(random_state=rs, kernel=k, n_restarts_optimizer=10),
            'QDD': QDD(random_state=rs),
            'GSi': GSi(random_state=rs),
            'GSx': GSx(random_state=rs),
            'GSy': GSy(random_state=rs),
            'LearningLoss': LearningLoss(
                BATCH=16, LR=0.01, MARGIN=1, WEIGHT=0.0001,
                EPOCH=200, EPOCHL=75, WDECAY=5e-4,
                random_state=rs
            ),
            'EGAL': EGAL(b_factor=0.25, random_state=rs),
            'mcdropout': mcdropout(random_state=rs, learning_rate=0.01, num_epochs=200, batch_size=16),
            'Basic_RD_ALR': Basic_RD_ALR(random_state=rs),
            'RD_GS_ALR': RD_GS_ALR(random_state=rs),
            'RD_QBC_ALR': RD_QBC_ALR(random_state=rs, num_learner=100),
            'RD_EMCM_ALR': RD_EMCM_ALR(random_state=rs),
            'BMDAL': BMDAL(random_state=rs, selection_method='lcmd'),
        }
        return registry

    def load_datasets(self) -> Dict[str, Tuple]:
        """Load and return datasets."""
        try:
            datasets = data_process_meta('../dataset/meta.csv', random_state=self.random_state)
            logger.info(f"Loaded {len(datasets)} datasets")
            return datasets
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise

    @staticmethod
    def _normalize_key(s: str) -> str:
        return s.strip().lower().replace(' ', '').replace('-', '_')

    def _resolve_dataset(self, datasets: Dict[str, Tuple]) -> Tuple[str, Tuple]:
        """Resolve dataset by (case-insensitive) name; raise with helpful message if not found."""
        target = self._normalize_key(self.dataset_name)
        name_map = {self._normalize_key(k): k for k in datasets.keys()}
        if target not in name_map:
            available = ', '.join(sorted(datasets.keys()))
            raise ValueError(
                f"Dataset '{self.dataset_name}' not found. Available datasets: {available}"
            )
        resolved_name = name_map[target]
        return resolved_name, datasets[resolved_name]

    def _resolve_strategy(self, registry: Dict[str, object]) -> object:
        """Resolve strategy by (case-insensitive) name; raise with helpful message if not found."""
        target = self._normalize_key(self.strategy_name)
        name_map = {self._normalize_key(k): k for k in registry.keys()}
        if target not in name_map:
            available = ', '.join(sorted(registry.keys()))
            raise ValueError(
                f"Strategy '{self.strategy_name}' not found. Available strategies: {available}"
            )
        return registry[name_map[target]]

    def run_benchmark(self) -> None:
        """Run the active learning benchmark."""
        try:
            # Load datasets and strategy registry
            datasets = self.load_datasets()
            strategy_registry = self._create_strategy_registry()

            # Resolve dataset and strategy by name
            dataset_name, dataset_value = self._resolve_dataset(datasets)
            selected_estimator = self._resolve_strategy(strategy_registry)

            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"Using strategy: {selected_estimator.__class__.__name__}")

            # Prepare data
            X_t, X_val, y_t, y_val = dataset_value

            # Check for duplicates
            duplicate_samples = X_t.duplicated()
            if duplicate_samples.any():
                logger.warning(f"Found {duplicate_samples.sum()} duplicate samples")

            # Calculate experiment parameters
            total_data_volume = X_t.shape[0]
            n_initial = 10
            n_queries = max(1, (total_data_volume - n_initial - 1) // self.n_pro_query)

            logger.info(f"Total data volume: {total_data_volume}")
            logger.info(f"Initial samples: {n_initial}")
            logger.info(f"Queries per iteration: {self.n_pro_query}")
            logger.info(f"Number of query iterations: {n_queries}")

            # Run active learning
            test_methods = ["normal"]  # Can be made configurable

            query_idx_all, query_time_all, metrics_all = active_learning(
                [selected_estimator], X_t, y_t, X_val, y_val,
                n_initial, self.n_pro_query, n_queries, self.threshold,
                initial_method=self.initial_method,
                test_methods=test_methods,
                random_state=self.random_state,
                record_metrics=True
            )

            # Save results
            self._save_results(
                query_idx_all, query_time_all, metrics_all,
                dataset_name, selected_estimator
            )

            logger.info("Benchmark completed successfully")

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise

    def _save_results(self, query_idx_all: Dict, query_time_all: Dict, metrics_all: Dict,
                      dataset_name: str, estimator: object) -> None:
        """Save benchmark results to JSON files."""
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Prepare result data
        result = query_idx_all.copy()
        result.update({
            'dataset_name': dataset_name,
            'initial_method': self.initial_method,
            'random_state': self.random_state,
            'strategy_name': estimator.__class__.__name__,
            'timestamp': current_time,
            'metrics': metrics_all
        })

        result_time_record = query_time_all.copy()
        result_time_record.update({
            'dataset_name': dataset_name,
            'initial_method': self.initial_method,
            'random_state': self.random_state,
            'strategy_name': estimator.__class__.__name__,
            'timestamp': current_time
        })

        # Save main results
        result_filename = os.path.join(
            self.result_path,
            f"{estimator.__class__.__name__}_{dataset_name}_{current_time}.json"
        )

        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {result_filename}")

        # Save time records
        time_record_path = os.path.join(self.result_path, 'time_record')
        time_filename = os.path.join(
            time_record_path,
            f"{estimator.__class__.__name__}_{dataset_name}_{current_time}.json"
        )

        with open(time_filename, 'w', encoding='utf-8') as f:
            json.dump(result_time_record, f, indent=2, ensure_ascii=False)

        logger.info(f"Time records saved to {time_filename}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run active learning benchmark on materials datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mode", type=str, default="benchmark",
        choices=["benchmark", "suggest"],
        help="benchmark: run full offline AL with known labels; suggest: query next batch from partial labels only"
    )

    parser.add_argument(
        "--random-state", type=int, required=True,
        help="Random state for reproducibility"
    )

    parser.add_argument(
        "--initial-method", type=str, default='random',
        choices=['random', 'greedy_search', 'kmeans', 'ncc'],
        help="Initial sampling method for active learning"
    )

    # NEW: names instead of indices
    parser.add_argument(
        "--strategy", type=str, required=True,
        help="Name of the active learning strategy (case-insensitive, e.g., 'RD_EMCM_ALR')"
    )

    parser.add_argument(
        "--dataset", type=str,
        help="Name of the dataset (case-insensitive, e.g., 'uci-concrete')"
    )

    parser.add_argument(
        "--n-pro-query", type=int, default=10,
        help="Number of samples to query per iteration"
    )

    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Performance threshold for early stopping"
    )

    parser.add_argument(
        "--config-file", type=str,
        help="Path to JSON configuration file (keys: dataset, strategy, random_state, initial_method, n_pro_query, threshold)"
    )

    parser.add_argument(
        "--labeled-csv", type=str,
        help="Path to labeled CSV (required in suggest mode)"
    )

    parser.add_argument(
        "--unlabeled-csv", type=str,
        help="Path to unlabeled/pool CSV with features only (required in suggest mode)"
    )

    parser.add_argument(
        "--target-column", type=str,
        help="Target column name in labeled CSV (required in suggest mode)"
    )

    parser.add_argument(
        "--feature-columns", type=str,
        help="Comma-separated feature names; default is all columns except target-column"
    )

    parser.add_argument(
        "--output-dir", type=str, default=os.path.join("..", "result", "suggest"),
        help="Output directory for suggest mode"
    )

    return parser.parse_args()


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        raise


def run_partial_label_suggestion(args: argparse.Namespace) -> None:
    """
    Suggest the next batch to label using only:
    - a small labeled set (features + target)
    - a large unlabeled pool (features only)
    """
    required = {
        "labeled_csv": args.labeled_csv,
        "unlabeled_csv": args.unlabeled_csv,
        "target_column": args.target_column,
        "strategy": args.strategy,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing required arguments for suggest mode: {', '.join(missing)}")

    labeled_df = pd.read_csv(args.labeled_csv)
    unlabeled_df = pd.read_csv(args.unlabeled_csv)

    if args.target_column not in labeled_df.columns:
        raise ValueError(f"Target column '{args.target_column}' not found in labeled CSV.")

    if args.feature_columns:
        feature_columns = [c.strip() for c in args.feature_columns.split(",") if c.strip()]
    else:
        feature_columns = [c for c in labeled_df.columns if c != args.target_column]

    missing_in_labeled = [c for c in feature_columns if c not in labeled_df.columns]
    missing_in_unlabeled = [c for c in feature_columns if c not in unlabeled_df.columns]
    if missing_in_labeled:
        raise ValueError(f"Feature columns missing in labeled CSV: {missing_in_labeled}")
    if missing_in_unlabeled:
        raise ValueError(f"Feature columns missing in unlabeled CSV: {missing_in_unlabeled}")

    labeled_clean = labeled_df.dropna(subset=feature_columns + [args.target_column]).copy()
    unlabeled_clean = unlabeled_df.dropna(subset=feature_columns).copy()
    if labeled_clean.empty:
        raise ValueError("No valid labeled rows remain after dropping NA values.")
    if unlabeled_clean.empty:
        raise ValueError("No valid unlabeled rows remain after dropping NA values.")

    X_labeled_raw = labeled_clean[feature_columns]
    y_labeled = labeled_clean[[args.target_column]]
    X_unlabeled_raw = unlabeled_clean[feature_columns]

    # Keep scaling consistent with existing project behavior.
    scaler = StandardScaler()
    scaler.fit(pd.concat([X_labeled_raw, X_unlabeled_raw], axis=0))
    X_labeled = pd.DataFrame(
        scaler.transform(X_labeled_raw), columns=feature_columns, index=labeled_clean.index
    )
    X_unlabeled = pd.DataFrame(
        scaler.transform(X_unlabeled_raw), columns=feature_columns, index=unlabeled_clean.index
    )

    config = {
        "random_state": args.random_state,
        "initial_method": args.initial_method,
        "strategy": args.strategy,
        "dataset": args.dataset or "__partial__",
        "n_pro_query": args.n_pro_query,
        "threshold": args.threshold,
    }
    benchmark = ActiveLearningBenchmark(config)
    strategy_registry = benchmark._create_strategy_registry()
    selected_estimator = benchmark._resolve_strategy(strategy_registry)

    query_idx = suggest_next_batch(
        estimator=selected_estimator,
        X_labeled=X_labeled,
        y_labeled=y_labeled,
        X_unlabeled=X_unlabeled,
        n_pro_query=args.n_pro_query,
        random_state=args.random_state,
    )

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.output_dir, exist_ok=True)

    selected_rows = unlabeled_clean.loc[query_idx].copy()
    selected_rows_path = os.path.join(
        args.output_dir,
        f"to_label_{selected_estimator.__class__.__name__}_{current_time}.csv"
    )
    selected_rows.to_csv(selected_rows_path, index=True)

    summary = {
        "mode": "suggest",
        "strategy_name": selected_estimator.__class__.__name__,
        "random_state": args.random_state,
        "n_pro_query": args.n_pro_query,
        "target_column": args.target_column,
        "feature_columns": feature_columns,
        "selected_indices": [int(i) for i in query_idx],
        "selected_rows_csv": selected_rows_path,
        "timestamp": current_time,
    }

    summary_path = os.path.join(
        args.output_dir,
        f"suggest_summary_{selected_estimator.__class__.__name__}_{current_time}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Suggested {len(query_idx)} samples to label.")
    logger.info(f"Selected rows saved to {selected_rows_path}")
    logger.info(f"Summary saved to {summary_path}")


def main():
    """Main entry point."""
    args = parse_arguments()

    if args.mode == "suggest":
        run_partial_label_suggestion(args)
        return

    # Load configuration
    if args.config_file:
        cfg = load_config_file(args.config_file)
        # minimal validation / defaults
        config = {
            'random_state': cfg['random_state'],
            'initial_method': cfg.get('initial_method', 'random'),
            'strategy': cfg['strategy'],
            'dataset': cfg['dataset'],
            'n_pro_query': cfg.get('n_pro_query', 10),
            'threshold': cfg.get('threshold', 0.85),
        }
    else:
        if not args.dataset:
            raise ValueError("--dataset is required in benchmark mode.")
        config = {
            'random_state': args.random_state,
            'initial_method': args.initial_method,
            'strategy': args.strategy,
            'dataset': args.dataset,
            'n_pro_query': args.n_pro_query,
            'threshold': args.threshold
        }

    # Run benchmark
    benchmark = ActiveLearningBenchmark(config)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate metrics across seeds and plot averaged curves."
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default="../result_review/2",
        help="Root directory containing per-seed folders (e.g., ../result_review/2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../result_review/2",
        help="Output directory for aggregated JSON and plots.",
    )
    parser.add_argument(
        "--filename-keyword",
        type=str,
        default="_metrics_",
        help="Keyword used to identify metrics files.",
    )
    return parser.parse_args()


def load_latest_metrics_per_seed(result_root: Path, filename_keyword: str):
    per_seed_latest = {}
    for path in result_root.rglob("*.json"):
        if filename_keyword not in path.name:
            continue
        seed_dir = path.parent.name
        if not seed_dir.isdigit():
            continue
        seed = int(seed_dir)
        current = per_seed_latest.get(seed)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            per_seed_latest[seed] = path

    if not per_seed_latest:
        raise FileNotFoundError(
            f"No metrics JSON found in {result_root} with keyword '{filename_keyword}'."
        )

    seed_metrics = []
    strategy_name = None

    for seed in sorted(per_seed_latest):
        file_path = per_seed_latest[seed]
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"Invalid metrics format in {file_path}")

        local_strategy_name = next(iter(payload.keys()))
        local_metrics = payload[local_strategy_name]
        if strategy_name is None:
            strategy_name = local_strategy_name
        elif local_strategy_name != strategy_name:
            raise ValueError(
                f"Strategy mismatch: got '{local_strategy_name}', expected '{strategy_name}'."
            )

        metric_rows = []
        for i, row in enumerate(local_metrics):
            metric_rows.append(
                {
                    "seed": seed,
                    "iteration": i,
                    "r2": float(row["r2"]),
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                }
            )

        seed_metrics.append(
            {
                "seed": seed,
                "file": str(file_path.resolve()),
                "metrics": metric_rows,
            }
        )

    return strategy_name, seed_metrics


def compute_average_metrics(seed_metrics):
    lengths = [len(item["metrics"]) for item in seed_metrics]
    n_iter = min(lengths)

    avg_metrics = []
    for i in range(n_iter):
        r2_values = np.array([item["metrics"][i]["r2"] for item in seed_metrics], dtype=float)
        mae_values = np.array([item["metrics"][i]["mae"] for item in seed_metrics], dtype=float)
        rmse_values = np.array([item["metrics"][i]["rmse"] for item in seed_metrics], dtype=float)

        avg_metrics.append(
            {
                "iteration": i,
                "r2_mean": float(r2_values.mean()),
                "r2_std": float(r2_values.std(ddof=0)),
                "mae_mean": float(mae_values.mean()),
                "mae_std": float(mae_values.std(ddof=0)),
                "rmse_mean": float(rmse_values.mean()),
                "rmse_std": float(rmse_values.std(ddof=0)),
            }
        )

    return avg_metrics, lengths


def plot_average_metrics(avg_metrics, strategy_name: str, output_path: Path):
    iters = [m["iteration"] for m in avg_metrics]

    r2_mean = np.array([m["r2_mean"] for m in avg_metrics], dtype=float)
    r2_std = np.array([m["r2_std"] for m in avg_metrics], dtype=float)
    mae_mean = np.array([m["mae_mean"] for m in avg_metrics], dtype=float)
    mae_std = np.array([m["mae_std"] for m in avg_metrics], dtype=float)
    rmse_mean = np.array([m["rmse_mean"] for m in avg_metrics], dtype=float)
    rmse_std = np.array([m["rmse_std"] for m in avg_metrics], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(iters, r2_mean, color="tab:blue", marker="o", label="mean")
    axes[0].fill_between(iters, r2_mean - r2_std, r2_mean + r2_std, color="tab:blue", alpha=0.2, label="std")
    axes[0].set_title(f"{strategy_name} | R2 vs Iteration")
    axes[0].set_ylabel("R2")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(iters, mae_mean, color="tab:orange", marker="o", label="mean")
    axes[1].fill_between(iters, mae_mean - mae_std, mae_mean + mae_std, color="tab:orange", alpha=0.2, label="std")
    axes[1].set_title(f"{strategy_name} | MAE vs Iteration")
    axes[1].set_ylabel("MAE")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(iters, rmse_mean, color="tab:green", marker="o", label="mean")
    axes[2].fill_between(
        iters, rmse_mean - rmse_std, rmse_mean + rmse_std, color="tab:green", alpha=0.2, label="std"
    )
    axes[2].set_title(f"{strategy_name} | RMSE vs Iteration")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("RMSE")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    result_root = Path(args.result_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_name, seed_metrics = load_latest_metrics_per_seed(result_root, args.filename_keyword)
    avg_metrics, lengths = compute_average_metrics(seed_metrics)

    summary = {
        "strategy_name": strategy_name,
        "result_root": str(result_root),
        "n_seeds": len(seed_metrics),
        "seed_metrics": seed_metrics,
        "iteration_lengths_by_seed": {str(item["seed"]): len(item["metrics"]) for item in seed_metrics},
        "note": (
            "Average is computed up to min iteration length across seeds."
            if len(set(lengths)) > 1
            else "All seeds have equal iteration length."
        ),
        "average_metrics": avg_metrics,
    }

    summary_json_path = output_dir / f"{strategy_name}_metrics_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_path = output_dir / f"{strategy_name}_metrics_average.png"
    plot_average_metrics(avg_metrics, strategy_name, plot_path)

    print(f"Loaded seeds: {[item['seed'] for item in seed_metrics]}")
    print(f"Summary JSON saved to: {summary_json_path}")
    print(f"Average plot saved to: {plot_path}")


if __name__ == "__main__":
    main()

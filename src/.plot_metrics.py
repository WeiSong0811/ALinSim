import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEED_FILE_REGEX = re.compile(r"GP_AL_results_seed_(\d+)\.json$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge GP_AL per-seed metrics, compute seed-average curves, and plot trends."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=".",
        help="Directory containing GP_AL_results_seed_*.json (default: current dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save merged summary JSON and plots (default: current dir).",
    )
    return parser.parse_args()


def load_seed_results(input_dir: Path):
    files = sorted(input_dir.glob("GP_AL_results_seed_*.json"))
    if not files:
        raise FileNotFoundError(f"No GP_AL_results_seed_*.json found in {input_dir}")

    per_seed = []
    for file_path in files:
        match = SEED_FILE_REGEX.search(file_path.name)
        if not match:
            continue
        seed = int(match.group(1))

        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        for key in ("mae", "rmse", "r2"):
            if key not in payload:
                raise ValueError(f"Missing key '{key}' in {file_path}")

        lengths = [len(payload["mae"]), len(payload["rmse"]), len(payload["r2"])]
        n_iter = min(lengths)

        metrics = []
        for i in range(n_iter):
            metrics.append(
                {
                    "seed": seed,
                    "iteration": i,
                    "mae": float(payload["mae"][i]),
                    "rmse": float(payload["rmse"][i]),
                    "r2": float(payload["r2"][i]),
                }
            )

        per_seed.append(
            {
                "seed": seed,
                "file": str(file_path.resolve()),
                "metrics": metrics,
            }
        )

    per_seed.sort(key=lambda x: x["seed"])
    return per_seed


def compute_average(per_seed):
    lengths = [len(item["metrics"]) for item in per_seed]
    n_iter = min(lengths)

    avg = []
    for i in range(n_iter):
        mae_values = np.array([item["metrics"][i]["mae"] for item in per_seed], dtype=float)
        rmse_values = np.array([item["metrics"][i]["rmse"] for item in per_seed], dtype=float)
        r2_values = np.array([item["metrics"][i]["r2"] for item in per_seed], dtype=float)

        avg.append(
            {
                "iteration": i,
                "mae_mean": float(mae_values.mean()),
                "mae_std": float(mae_values.std(ddof=0)),
                "rmse_mean": float(rmse_values.mean()),
                "rmse_std": float(rmse_values.std(ddof=0)),
                "r2_mean": float(r2_values.mean()),
                "r2_std": float(r2_values.std(ddof=0)),
            }
        )

    return avg, lengths


def plot_average(avg_metrics, output_path: Path):
    iters = np.array([row["iteration"] for row in avg_metrics], dtype=int)

    mae_mean = np.array([row["mae_mean"] for row in avg_metrics], dtype=float)
    mae_std = np.array([row["mae_std"] for row in avg_metrics], dtype=float)
    rmse_mean = np.array([row["rmse_mean"] for row in avg_metrics], dtype=float)
    rmse_std = np.array([row["rmse_std"] for row in avg_metrics], dtype=float)
    r2_mean = np.array([row["r2_mean"] for row in avg_metrics], dtype=float)
    r2_std = np.array([row["r2_std"] for row in avg_metrics], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(iters, r2_mean, color="tab:blue", marker="o", label="mean")
    axes[0].fill_between(iters, r2_mean - r2_std, r2_mean + r2_std, color="tab:blue", alpha=0.2, label="std")
    axes[0].set_title("R2 vs Iteration (Seed Average)")
    axes[0].set_ylabel("R2")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(iters, mae_mean, color="tab:orange", marker="o", label="mean")
    axes[1].fill_between(iters, mae_mean - mae_std, mae_mean + mae_std, color="tab:orange", alpha=0.2, label="std")
    axes[1].set_title("MAE vs Iteration (Seed Average)")
    axes[1].set_ylabel("MAE")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(iters, rmse_mean, color="tab:green", marker="o", label="mean")
    axes[2].fill_between(
        iters, rmse_mean - rmse_std, rmse_mean + rmse_std, color="tab:green", alpha=0.2, label="std"
    )
    axes[2].set_title("RMSE vs Iteration (Seed Average)")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("RMSE")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed = load_seed_results(input_dir)
    avg_metrics, lengths = compute_average(per_seed)

    summary = {
        "input_dir": str(input_dir),
        "n_seeds": len(per_seed),
        "seeds": [item["seed"] for item in per_seed],
        "per_seed_metrics": per_seed,
        "iteration_lengths_by_seed": {str(item["seed"]): len(item["metrics"]) for item in per_seed},
        "note": (
            "Average computed up to min iteration length across seeds."
            if len(set(lengths)) > 1
            else "All seeds have same iteration length."
        ),
        "average_metrics": avg_metrics,
    }

    summary_path = output_dir / "GP_AL_metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_path = output_dir / "GP_AL_metrics_average.png"
    plot_average(avg_metrics, plot_path)

    print(f"Loaded seeds: {summary['seeds']}")
    print(f"Summary JSON saved to: {summary_path}")
    print(f"Average plot saved to: {plot_path}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ["r2", "mae", "rmse"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare GP and AutoML metrics by averaging across seeds and plotting trends."
    )
    parser.add_argument(
        "--gp-json",
        type=str,
        default="GP_AL_metrics_summary.json",
        help="Path to GP summary JSON.",
    )
    parser.add_argument(
        "--automl-json",
        type=str,
        default="AutoML.json",
        help="Path to AutoML summary JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory to save output figure and merged summary.",
    )
    return parser.parse_args()


def _extract_seed_metrics(payload: dict):
    if "per_seed_metrics" in payload:
        return payload["per_seed_metrics"]
    if "seed_metrics" in payload:
        return payload["seed_metrics"]
    raise ValueError("JSON does not contain 'per_seed_metrics' or 'seed_metrics'.")


def average_over_seeds(payload: dict):
    seed_metrics = _extract_seed_metrics(payload)
    if not seed_metrics:
        raise ValueError("No seed metrics found.")

    lengths = [len(item["metrics"]) for item in seed_metrics]
    n_iter = min(lengths)

    avg = []
    for i in range(n_iter):
        row = {"iteration": i}
        for m in METRICS:
            vals = np.array([item["metrics"][i][m] for item in seed_metrics], dtype=float)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=0))
        avg.append(row)

    return {
        "seeds": [item.get("seed") for item in seed_metrics],
        "n_seeds": len(seed_metrics),
        "n_iterations_used": n_iter,
        "iteration_lengths": lengths,
        "average_metrics": avg,
    }


def to_arrays(avg_result: dict, metric: str):
    arr = avg_result["average_metrics"]
    x = np.array([r["iteration"] for r in arr], dtype=int)
    y = np.array([r[f"{metric}_mean"] for r in arr], dtype=float)
    s = np.array([r[f"{metric}_std"] for r in arr], dtype=float)
    return x, y, s


def plot_compare(gp_avg: dict, automl_avg: dict, out_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    style = {
        "GP": ("tab:blue", gp_avg),
        "AutoML": ("tab:orange", automl_avg),
    }

    for ax, metric in zip(axes, METRICS):
        for name, (color, avg_data) in style.items():
            x, y, s = to_arrays(avg_data, metric)
            ax.plot(x, y, color=color, marker="o", label=f"{name} mean")
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.18, label=f"{name} std")

        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs Iteration")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Iteration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    gp_path = Path(args.gp_json).resolve()
    automl_path = Path(args.automl_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with gp_path.open("r", encoding="utf-8") as f:
        gp_payload = json.load(f)
    with automl_path.open("r", encoding="utf-8") as f:
        automl_payload = json.load(f)

    gp_avg = average_over_seeds(gp_payload)
    automl_avg = average_over_seeds(automl_payload)

    merged = {
        "gp": gp_avg,
        "automl": automl_avg,
    }
    merged_json = out_dir / "GP_vs_AutoML_avg_metrics.json"
    with merged_json.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    fig_path = out_dir / "GP_vs_AutoML_metrics.png"
    plot_compare(gp_avg, automl_avg, fig_path)

    print(f"Saved merged summary: {merged_json}")
    print(f"Saved comparison plot: {fig_path}")


if __name__ == "__main__":
    main()

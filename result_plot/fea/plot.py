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
        default="./GP_metrics_summary.json",
        help="Path to GP summary JSON.",
    )
    parser.add_argument(
        "--automl-json",
        type=str,
        default="./AL_metrics_summary.json",
        help="Path to AutoML summary JSON.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="./",
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


def _looks_like_idx_summary(payload: dict) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    for value in payload.values():
        if not isinstance(value, dict):
            return False
        if not any(f"{m}_mean" in value for m in METRICS):
            return False
    return True


def _from_metric_summary(payload: dict) -> dict:
    n_iter = min(len(payload.get(f"{m}_mean", [])) for m in METRICS)
    avg = []
    for i in range(n_iter):
        row = {"iteration": i}
        for m in METRICS:
            mean_list = payload.get(f"{m}_mean", [])
            std_list = payload.get(f"{m}_std", [0.0] * len(mean_list))
            row[f"{m}_mean"] = float(mean_list[i])
            row[f"{m}_std"] = float(std_list[i]) if i < len(std_list) else 0.0
        avg.append(row)
    return {
        "n_iterations_used": n_iter,
        "average_metrics": avg,
    }


def _from_avg_metrics(payload: dict) -> dict:
    if "average_metrics" not in payload:
        raise ValueError("JSON does not contain 'average_metrics'.")
    return {
        "n_iterations_used": len(payload["average_metrics"]),
        "average_metrics": payload["average_metrics"],
    }


def _average_over_idx(payload: dict) -> dict:
    by_idx = [_from_metric_summary(item) for item in payload.values()]
    n_iter = min(item["n_iterations_used"] for item in by_idx)

    avg = []
    for i in range(n_iter):
        row = {"iteration": i}
        for m in METRICS:
            means = np.array([item["average_metrics"][i][f"{m}_mean"] for item in by_idx], dtype=float)
            stds = np.array([item["average_metrics"][i][f"{m}_std"] for item in by_idx], dtype=float)
            row[f"{m}_mean"] = float(means.mean())
            row[f"{m}_std"] = float(np.sqrt(np.mean(stds**2)))
        avg.append(row)

    return {
        "n_iterations_used": n_iter,
        "n_idx": len(by_idx),
        "average_metrics": avg,
    }


def _looks_like_metric_summary(payload: dict) -> bool:
    return all(isinstance(payload.get(f"{m}_mean"), list) for m in METRICS)


def load_results(payload: dict) -> dict:
    if _looks_like_idx_summary(payload):
        return _average_over_idx(payload)
    if _looks_like_metric_summary(payload):
        return _from_metric_summary(payload)
    if "average_metrics" in payload:
        return _from_avg_metrics(payload)
    return average_over_seeds(payload)


def to_arrays(avg_result: dict, metric: str):
    arr = avg_result["average_metrics"]
    x = np.array([r["iteration"] for r in arr], dtype=int)
    y = np.array([r[f"{metric}_mean"] for r in arr], dtype=float)
    s = np.array([r[f"{metric}_std"] for r in arr], dtype=float)
    return x, y, s


def plot_metric(metric: str, series_map: dict, out_path: Path, use_log_scale: bool):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    colors = {
        "GP": "tab:blue",
        "AutoML": "tab:orange",
        "QBC": "tab:green",
    }

    for name, idx_data in series_map.items():
        x, y, s = to_arrays(idx_data, metric)
        color = colors.get(name, None)
        ax.plot(x, y, color=color, marker="o", label=f"{name} mean")
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.2, label=f"{name} std")

    if use_log_scale:
        if metric == "r2":
            ax.set_yscale("symlog", linthresh=1e-2)
        else:
            ax.set_yscale("log")

    scale_tag = "LOG" if use_log_scale else "LINEAR"
    ax.set_title(f"{metric.upper()} vs Iteration ({scale_tag})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric.upper())
    ax.grid(alpha=0.3)
    ax.legend()
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

    gp_result = load_results(gp_payload)
    automl_result = load_results(automl_payload)

    merged = {
        "gp": gp_result,
        "automl": automl_result,
    }
    merged_json = out_dir / "GP_vs_AutoML_vs_QBC.json"
    with merged_json.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    saved = []
    for metric in METRICS:
        series_map = {
            "GP": gp_result,
            "AutoML": automl_result,
        }
        for use_log_scale in [False, True]:
            suffix = "linear" if not use_log_scale else "log"
            fig_path = out_dir / f"GP_vs_AutoML_vs_{metric}_{suffix}.png"
            plot_metric(metric, series_map, fig_path, use_log_scale=use_log_scale)
            saved.append(fig_path)

    print(f"Saved merged summary: {merged_json}")
    for p in saved:
        print(f"Saved comparison plot: {p}")


if __name__ == "__main__":
    main()

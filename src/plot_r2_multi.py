import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    if not metrics:
        return None, None, None, None
    return data, metrics, data.get("dataset_name"), data.get("initial_method")


def iter_result_files(base_dir: Path):
    for path in base_dir.rglob("*.json"):
        if "time_record" in path.parts:
            continue
        yield path


def main():
    parser = argparse.ArgumentParser(
        description="Plot R2 curves for all results grouped by random_state."
    )
    parser.add_argument("--result-root", default="result", help="Root result directory.")
    parser.add_argument("--n-pro-query", required=True, help="n_pro_query folder name, e.g. 10 or 40")
    parser.add_argument("--dataset", help="Filter by dataset_name (optional)")
    parser.add_argument("--initial-method", help="Filter by initial_method (optional)")
    parser.add_argument("--out-dir", default="r2_plots", help="Output directory for plots.")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    base_dir = result_root / str(args.n_pro_query)
    if not base_dir.exists():
        raise SystemExit(f"Path not found: {base_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group curves by random_state (first folder under n_pro_query)
    random_state_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not random_state_dirs:
        raise SystemExit(f"No random_state dirs under {base_dir}")

    for rs_dir in random_state_dirs:
        curves = []
        for json_path in iter_result_files(rs_dir):
            data, metrics, dataset_name, initial_method = load_metrics(json_path)
            if metrics is None:
                continue
            if args.dataset and dataset_name != args.dataset:
                continue
            if args.initial_method and initial_method != args.initial_method:
                continue

            strategy_name = data.get("strategy_name", json_path.stem.split("_")[0])

            for key, series in metrics.items():
                r2_vals = [float(item["r2"]) for item in series]
                label = f"{strategy_name}"
                if key != strategy_name:
                    label = f"{strategy_name}:{key}"
                if dataset_name:
                    label += f" | {dataset_name}"
                if initial_method:
                    label += f" | {initial_method}"
                curves.append((label, r2_vals))

        if not curves:
            continue

        plt.figure(figsize=(9, 5))
        max_r2 = None
        max_label = None
        for label, r2_vals in curves:
            rounds = list(range(len(r2_vals)))
            plt.plot(rounds, r2_vals, linewidth=1.5, label=label)
            for idx, val in enumerate(r2_vals):
                if max_r2 is None or val > max_r2:
                    max_r2 = val
                    max_label = label

        plt.title(f"R2 over rounds (random_state={rs_dir.name})")
        plt.xlabel("Round (0 = after init)")
        plt.ylabel("R2")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        if max_r2 is not None:
            text = f"max R2={max_r2:.4f} | {max_label}"
            plt.figtext(0.5, 0.01, text, ha="center", fontsize=9)
        plt.tight_layout()

        out_path = out_dir / f"r2_random_state_{rs_dir.name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

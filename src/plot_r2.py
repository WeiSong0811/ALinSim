import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    if not metrics:
        raise ValueError("No 'metrics' found in JSON.")
    return metrics


def pick_strategy(metrics: dict, strategy: str | None):
    keys = list(metrics.keys())
    if strategy is None:
        if len(keys) == 1:
            return keys[0], metrics[keys[0]]
        raise ValueError(f"Multiple strategies found: {keys}. Use --strategy.")
    if strategy not in metrics:
        raise ValueError(f"Strategy '{strategy}' not in metrics. Available: {keys}")
    return strategy, metrics[strategy]


def extract_r2(series):
    # series: list[dict] with keys r2/mae/rmse per round
    return [float(item["r2"]) for item in series]


def main():
    parser = argparse.ArgumentParser(description="Plot R2 over AL rounds from result JSON.")
    parser.add_argument("json_path", help="Path to result JSON containing metrics")
    parser.add_argument("--strategy", help="Strategy key inside metrics (if multiple)")
    parser.add_argument("--out", help="Output image path (png). If omitted, show window.")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    metrics = load_metrics(json_path)
    strat, series = pick_strategy(metrics, args.strategy)
    r2_vals = extract_r2(series)

    rounds = list(range(len(r2_vals)))
    plt.figure(figsize=(8, 4.5))
    plt.plot(rounds, r2_vals, marker="o", linewidth=1.5)
    plt.title(f"R2 over rounds ({strat})")
    plt.xlabel("Round (0 = after init)")
    plt.ylabel("R2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

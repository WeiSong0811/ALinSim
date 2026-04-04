"""
Plot averaged MAE/RMSE/R2 curves for AutoML, GP and QBC experiment results.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_DIRS = {
    "pan_2": Path("results/result_single_pan"),
    "pan_5": Path("results/result_pan_5"),
    "fea_2": Path("results/result_single_fea_2"),
    "fea_5": Path("results/result_single_fea_5"),
    "fea_10": Path("results/result_single_fea_10"),
    "trc_5": Path("results/result_single_trc"),
}

MODEL_SAMPLE_SETTINGS = {
    "pan_2": {"n_initial": 5, "query_size": 2},
    "pan_5": {"n_initial": 5, "query_size": 5},
    "fea_2": {"n_initial": 5, "query_size": 2},
    "fea_5": {"n_initial": 5, "query_size": 5},
    "fea_10": {"n_initial": 5, "query_size": 10},
    "trc_5": {"n_initial": 5, "query_size": 5},
}

MODEL_SEEDS = {
    "pan_2": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    "pan_5": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49], 
    "fea_2": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49], 
    "fea_5": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    "fea_10": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    "trc_5": [40, 41, 42, 43, 44, 46, 47, 49, 50, 51],
}

MODEL_AUTOML_CSV = {
    "trc_5": Path("results/result_single_trc/result_trc_auto/AutoML_AL_results.csv"),
}

MODEL_AUTOML_ROOT_PATTERNS = {
    "pan_5": {
        "prefix": "AM_AL_results_seed_",
        "suffix_template": "_{group}.json",
        "file_pattern_template": "AM_AL_results_seed_{seed}_{group}.json",
    },
}

METRICS = ("mae", "rmse", "r2")
METHOD_STYLES = {
    "AutoML": {"color": "tab:orange"},
    "GP": {"color": "tab:blue"},
    "QBC": {"color": "tab:green"},
}
METRIC_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R2",
}
ALL_MODELS = tuple(MODEL_DIRS.keys())


def load_json_data(data):
    if all(metric in data for metric in METRICS) and all(
        isinstance(data[metric], list) for metric in METRICS
    ):
        return (
            [np.nan if value is None else float(value) for value in data["mae"]],
            [np.nan if value is None else float(value) for value in data["rmse"]],
            [np.nan if value is None else float(value) for value in data["r2"]],
        )

    if all(metric in data for metric in METRICS):
        return (
            [np.nan if data["mae"] is None else float(data["mae"])],
            [np.nan if data["rmse"] is None else float(data["rmse"])],
            [np.nan if data["r2"] is None else float(data["r2"])],
        )

    method_key = next(iter(data.keys()))
    steps = data[method_key]
    return (
        [np.nan if step.get("mae") is None else float(step.get("mae")) for step in steps],
        [np.nan if step.get("rmse") is None else float(step.get("rmse")) for step in steps],
        [np.nan if step.get("r2") is None else float(step.get("r2")) for step in steps],
    )


def _numeric_dirs(base_dir):
    return sorted(
        [path for path in base_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def _find_single_file(search_dir, pattern):
    files = sorted(search_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matched '{pattern}' in {search_dir}")
    return files[0]


def _extract_seed(file_path, prefix, suffix):
    pattern = re.compile(rf"^{re.escape(prefix)}(?P<seed>\d+){re.escape(suffix)}$")
    match = pattern.match(file_path.name)
    return match.group("seed") if match else None


def _available_root_seeds(base_dir, prefix, suffix):
    seeds = set()
    for path in base_dir.iterdir():
        if not path.is_file():
            continue
        seed = _extract_seed(path, prefix, suffix)
        if seed is not None:
            seeds.add(seed)
    return seeds


def _load_metrics_file(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        return load_json_data(json.load(file))


def _load_automl_csv_by_seed(csv_path):
    metrics_by_seed = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            seed = str(int(row["seed"]))
            run_id = int(row["run_id"])
            metrics_by_seed.setdefault(seed, []).append(
                (
                    run_id,
                    np.nan if row["mae"] == "" else float(row["mae"]),
                    np.nan if row["rmse"] == "" else float(row["rmse"]),
                    np.nan if row["r2"] == "" else float(row["r2"]),
                )
            )

    result = {}
    for seed, rows in metrics_by_seed.items():
        rows.sort(key=lambda item: item[0])
        result[seed] = (
            [item[1] for item in rows],
            [item[2] for item in rows],
            [item[3] for item in rows],
        )

    return result


def _load_automl_root_json_by_seed(base_dir, allowed_seeds, file_pattern_template):
    metrics_by_seed = {}
    for seed in allowed_seeds:
        file_path = _find_single_file(base_dir, file_pattern_template.format(seed=seed))
        metrics_by_seed[str(seed)] = _load_metrics_file(file_path)
    return metrics_by_seed


def _truncate_to_common_length(*arrays):
    min_length = min(len(array) for array in arrays)
    return [np.asarray(array[:min_length], dtype=float) for array in arrays]


def _safe_nan_stats(matrix):
    valid_counts = np.sum(~np.isnan(matrix), axis=0)

    sum_values = np.nansum(matrix, axis=0)
    mean = np.divide(
        sum_values,
        valid_counts,
        out=np.full(matrix.shape[1], np.nan, dtype=float),
        where=valid_counts > 0,
    )

    centered = matrix - mean
    centered[np.isnan(matrix)] = np.nan
    sq_sum = np.nansum(centered ** 2, axis=0)
    var = np.divide(
        sq_sum,
        valid_counts,
        out=np.full(matrix.shape[1], np.nan, dtype=float),
        where=valid_counts > 0,
    )
    std = np.sqrt(var)

    return mean.tolist(), std.tolist()


def _summarize_runs(mae_all, rmse_all, r2_all):
    mae_all = _truncate_to_common_length(*mae_all)
    rmse_all = _truncate_to_common_length(*rmse_all)
    r2_all = _truncate_to_common_length(*r2_all)

    mae_all = np.array(mae_all, dtype=float)
    rmse_all = np.array(rmse_all, dtype=float)
    r2_all = np.array(r2_all, dtype=float)
    mae_mean, mae_std = _safe_nan_stats(mae_all)
    rmse_mean, rmse_std = _safe_nan_stats(rmse_all)
    r2_mean, r2_std = _safe_nan_stats(r2_all)

    return {
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "n_runs": int(len(mae_all)),
    }


def _read_one_group(
    base_dir,
    automl_dir,
    allowed_seeds,
    gp_pattern,
    qbc_pattern,
    gp_seed_prefix,
    gp_seed_suffix,
    qbc_seed_prefix,
    qbc_seed_suffix,
    automl_by_seed=None,
):
    auto_mae_all, auto_rmse_all, auto_r2_all = [], [], []
    gp_mae_all, gp_rmse_all, gp_r2_all = [], [], []
    qbc_mae_all, qbc_rmse_all, qbc_r2_all = [], [], []

    automl_seed_dirs = {seed_dir.name: seed_dir for seed_dir in _numeric_dirs(automl_dir)}
    automl_seeds = set(automl_seed_dirs) if automl_by_seed is None else set(automl_by_seed.keys())
    gp_seeds = _available_root_seeds(base_dir, gp_seed_prefix, gp_seed_suffix)
    qbc_seeds = _available_root_seeds(base_dir, qbc_seed_prefix, qbc_seed_suffix)
    allowed_seed_set = {str(seed) for seed in allowed_seeds}
    common_seeds = sorted(
        (automl_seeds & gp_seeds & qbc_seeds & allowed_seed_set),
        key=int,
    )

    if not common_seeds:
        raise FileNotFoundError(
            f"No common seeds found across AutoML/GP/QBC in {automl_dir} and {base_dir}."
        )

    for seed in common_seeds:
        if automl_by_seed is None:
            seed_dir = automl_seed_dirs[seed]
            auto_file = _find_single_file(
                seed_dir, "TreeBasedRegressor_Representativity_self_metrics_*.json"
            )
            auto_mae_seq, auto_rmse_seq, auto_r2_seq = _load_metrics_file(auto_file)
        else:
            auto_mae_seq, auto_rmse_seq, auto_r2_seq = automl_by_seed[seed]
        auto_mae_all.append(auto_mae_seq)
        auto_rmse_all.append(auto_rmse_seq)
        auto_r2_all.append(auto_r2_seq)

        gp_file = _find_single_file(base_dir, gp_pattern.format(seed=seed))
        gp_mae_seq, gp_rmse_seq, gp_r2_seq = _load_metrics_file(gp_file)
        gp_mae_all.append(gp_mae_seq)
        gp_rmse_all.append(gp_rmse_seq)
        gp_r2_all.append(gp_r2_seq)

        qbc_file = _find_single_file(base_dir, qbc_pattern.format(seed=seed))
        qbc_mae_seq, qbc_rmse_seq, qbc_r2_seq = _load_metrics_file(qbc_file)
        qbc_mae_all.append(qbc_mae_seq)
        qbc_rmse_all.append(qbc_rmse_seq)
        qbc_r2_all.append(qbc_r2_seq)

    return (
        _summarize_runs(auto_mae_all, auto_rmse_all, auto_r2_all),
        _summarize_runs(gp_mae_all, gp_rmse_all, gp_r2_all),
        _summarize_runs(qbc_mae_all, qbc_rmse_all, qbc_r2_all),
    )


def read_data(model_name):
    base_dir = MODEL_DIRS.get(model_name)
    allowed_seeds = MODEL_SEEDS.get(model_name)
    automl_csv_path = MODEL_AUTOML_CSV.get(model_name)
    automl_root_pattern = MODEL_AUTOML_ROOT_PATTERNS.get(model_name)
    if base_dir is None:
        raise ValueError(f"Unknown model name: {model_name}")
    if allowed_seeds is None:
        raise ValueError(f"Missing seed configuration for model: {model_name}")
    if not base_dir.exists():
        raise FileNotFoundError(f"Result directory does not exist: {base_dir}")

    idx_dirs = _numeric_dirs(base_dir)
    if not idx_dirs:
        raise FileNotFoundError(f"No numeric directories found in {base_dir}")

    automl_by_seed = None
    if automl_csv_path is not None:
        if not automl_csv_path.exists():
            raise FileNotFoundError(f"AutoML CSV does not exist: {automl_csv_path}")
        automl_by_seed = _load_automl_csv_by_seed(automl_csv_path)

    has_nested_seed_dirs = any(_numeric_dirs(idx_dir) for idx_dir in idx_dirs)

    if has_nested_seed_dirs:
        result = {}
        for idx_dir in idx_dirs:
            group_automl_by_seed = automl_by_seed
            if automl_root_pattern is not None:
                group_automl_by_seed = _load_automl_root_json_by_seed(
                    base_dir=base_dir,
                    allowed_seeds=allowed_seeds,
                    file_pattern_template=automl_root_pattern["file_pattern_template"].format(
                        seed="{seed}",
                        group=idx_dir.name,
                    ),
                )
            result[idx_dir.name] = _read_one_group(
                base_dir=base_dir,
                automl_dir=idx_dir,
                allowed_seeds=allowed_seeds,
                gp_pattern=f"GP_AL_results_seed_{{seed}}_{idx_dir.name}.json",
                qbc_pattern=f"QBC_AL_PAN_results_seed_{{seed}}_{idx_dir.name}.json",
                gp_seed_prefix="GP_AL_results_seed_",
                gp_seed_suffix=f"_{idx_dir.name}.json",
                qbc_seed_prefix="QBC_AL_PAN_results_seed_",
                qbc_seed_suffix=f"_{idx_dir.name}.json",
                automl_by_seed=group_automl_by_seed,
            )
        return result

    return _read_one_group(
        base_dir=base_dir,
        automl_dir=base_dir,
        allowed_seeds=allowed_seeds,
        gp_pattern="GP_AL_results_seed_{seed}*.json",
        qbc_pattern="QBC_AL_PAN_results_seed_{seed}*.json",
        gp_seed_prefix="GP_AL_results_seed_",
        gp_seed_suffix=".json",
        qbc_seed_prefix="QBC_AL_PAN_results_seed_",
        qbc_seed_suffix=".json",
        automl_by_seed=automl_by_seed,
    )


def build_sample_counts(model_name, n_points):
    settings = MODEL_SAMPLE_SETTINGS.get(model_name)
    if settings is None:
        raise ValueError(f"Unknown model name for sample counts: {model_name}")

    n_initial = settings["n_initial"]
    query_size = settings["query_size"]
    return np.array([n_initial + query_size * idx for idx in range(n_points)], dtype=int)


def _normalize_processed_data(processed_data):
    if isinstance(processed_data, tuple) and len(processed_data) == 3:
        return {"default": processed_data}

    if isinstance(processed_data, dict):
        normalized = {}
        for key, value in processed_data.items():
            if not isinstance(value, tuple) or len(value) != 3:
                raise ValueError(
                    "Each dict value in processed_data must be a "
                    "(avg_auto, avg_gp, avg_qbc) tuple."
                )
            normalized[str(key)] = value
        return normalized

    raise ValueError(
        "processed_data must be either a 3-item tuple or a dict of 3-item tuples."
    )


def _plot_one_metric(
    ax,
    x,
    metric,
    avg_auto,
    avg_gp,
    avg_qbc,
    x_label,
    show_std=True,
    use_log_scale=False,
):
    series = {
        "AutoML": avg_auto,
        "GP": avg_gp,
        "QBC": avg_qbc,
    }

    for method_name, stats in series.items():
        y = np.asarray(stats[f"{metric}_mean"], dtype=float)
        std = np.asarray(stats[f"{metric}_std"], dtype=float)
        length = min(len(x), len(y), len(std))
        if length == 0:
            continue

        x_plot = x[:length]
        y_plot = y[:length]
        std_plot = std[:length]
        color = METHOD_STYLES[method_name]["color"]

        ax.plot(x_plot, y_plot, marker="o", linewidth=2, color=color, label=method_name)
        if show_std:
            lower = y_plot - std_plot
            upper = y_plot + std_plot
            if use_log_scale and np.all(np.isfinite(y_plot)) and np.all(y_plot > 0):
                lower = np.clip(lower, 1e-12, None)
                upper = np.clip(upper, 1e-12, None)
            ax.fill_between(x_plot, lower, upper, color=color, alpha=0.18)

    ax.set_xlabel(x_label)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(METRIC_LABELS[metric])
    ax.grid(alpha=0.3, linestyle="--")


def _metric_y_scale(metric, avg_auto, avg_gp, avg_qbc):
    candidates = []
    for stats in (avg_auto, avg_gp, avg_qbc):
        values = np.asarray(stats[f"{metric}_mean"], dtype=float)
        candidates.append(values[np.isfinite(values)])

    merged = np.concatenate([vals for vals in candidates if vals.size > 0]) if candidates else np.array([])
    if merged.size == 0:
        return "log"
    if np.all(merged > 0):
        return "log"
    return "symlog"


def _apply_y_axis_scale(ax, metric, avg_auto, avg_gp, avg_qbc, use_log_scale):
    if not use_log_scale:
        return

    scale = _metric_y_scale(metric, avg_auto, avg_gp, avg_qbc)
    if scale == "log":
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=1.0)


def _build_metric_save_path(save_path, metric, group_name, multi_group, with_std, use_log_scale):
    save_target = Path(save_path)
    suffix = save_target.suffix or ".png"
    stem_parts = [save_target.stem]
    if multi_group:
        stem_parts.append(group_name)
    stem_parts.append(metric)
    stem_parts.append("with_std" if with_std else "no_std")
    directory = save_target.parent / "log" if use_log_scale else save_target.parent
    file_path = directory / ("_".join(stem_parts) + suffix)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def visualize_data(
    processed_data,
    save_path=None,
    figsize=(8, 5),
    x_values=None,
    x_label="Number of Samples",
    show_std=True,
    use_log_scale=False,
):
    normalized = _normalize_processed_data(processed_data)
    figures = {}
    multi_group = len(normalized) > 1

    for group_name, (avg_auto, avg_gp, avg_qbc) in normalized.items():
        lengths = [
            len(avg_auto["mae_mean"]),
            len(avg_gp["mae_mean"]),
            len(avg_qbc["mae_mean"]),
        ]
        n_points = min(lengths)
        if n_points == 0:
            raise ValueError(f"No plottable data found for group '{group_name}'.")

        if x_values is None:
            x = np.arange(n_points)
            current_x_label = "Iteration"
        else:
            x = np.asarray(x_values[:n_points], dtype=float)
            current_x_label = x_label

        group_figures = {}
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=figsize)
            _plot_one_metric(
                ax,
                x,
                metric,
                avg_auto,
                avg_gp,
                avg_qbc,
                current_x_label,
                show_std=show_std,
                use_log_scale=use_log_scale,
            )
            _apply_y_axis_scale(ax, metric, avg_auto, avg_gp, avg_qbc, use_log_scale)

            n_runs = avg_auto.get("n_runs", "unknown")
            title = f"{METRIC_LABELS[metric]} Comparison of AutoML, GP and QBC (n={n_runs})"
            if group_name != "default":
                title = f"{title} | Group {group_name}"
            if not show_std:
                title = f"{title} | Mean Only"
            if use_log_scale:
                title = f"{title} | Log Y"
            fig.suptitle(title, fontsize=14, y=0.97)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.91),
                    ncol=3,
                    frameon=False,
                )

            fig.tight_layout(rect=(0, 0, 1, 0.82))

            if save_path is not None:
                file_path = _build_metric_save_path(
                    save_path=save_path,
                    metric=metric,
                    group_name=group_name,
                    multi_group=multi_group,
                    with_std=show_std,
                    use_log_scale=use_log_scale,
                )
                fig.savefig(file_path, dpi=300, bbox_inches="tight")

            plt.close(fig)
            group_figures[metric] = fig

        figures[group_name] = group_figures

    return figures if len(figures) > 1 else next(iter(figures.values()))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read experiment results and plot AutoML/GP/QBC comparisons."
    )
    parser.add_argument(
        "--model",
        choices=ALL_MODELS + ("all",),
        default="pan_2",
        help="Model/result group to plot, or 'all' to run every model.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional output path. Multi-group results append the group name automatically.",
    )
    parser.add_argument(
        "--hide",
        action="store_true",
        help="Save figures without displaying the matplotlib window.",
    )
    return parser.parse_args()


def _infer_n_points(processed_data):
    if isinstance(processed_data, dict):
        first_group = next(iter(processed_data.values()))
        return min(
            len(first_group[0]["mae_mean"]),
            len(first_group[1]["mae_mean"]),
            len(first_group[2]["mae_mean"]),
        )

    return min(
        len(processed_data[0]["mae_mean"]),
        len(processed_data[1]["mae_mean"]),
        len(processed_data[2]["mae_mean"]),
    )

def _run_one_model(model_name, save_path=None):
    processed_data = read_data(model_name)
    resolved_save_path = save_path or f"result_plot/{model_name}.png"
    n_points = _infer_n_points(processed_data)
    sample_counts = build_sample_counts(model_name, n_points)
    visualize_data(
        processed_data,
        save_path=resolved_save_path,
        x_values=sample_counts,
        show_std=True,
        use_log_scale=False,
    )
    visualize_data(
        processed_data,
        save_path=resolved_save_path,
        x_values=sample_counts,
        show_std=False,
        use_log_scale=False,
    )
    visualize_data(
        processed_data,
        save_path=resolved_save_path,
        x_values=sample_counts,
        show_std=True,
        use_log_scale=True,
    )
    visualize_data(
        processed_data,
        save_path=resolved_save_path,
        x_values=sample_counts,
        show_std=False,
        use_log_scale=True,
    )

def main():
    args = parse_args()
    if args.model == "all":
        output_dir = Path(args.save_path) if args.save_path is not None else Path("result_plot")
        for model_name in ALL_MODELS:
            model_save_path = output_dir / f"{model_name}.png"
            _run_one_model(model_name, save_path=model_save_path)
        return

    save_path = args.save_path or f"result_plot/{args.model}.png"
    _run_one_model(args.model, save_path=save_path)

if __name__ == "__main__":
    main()

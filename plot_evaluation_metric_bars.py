#!/usr/bin/env python3
"""Plot best-validation server and layer-averaged client metrics as bars."""

import argparse
import csv
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib


# Keep file generation headless, but select a GUI backend for --show-only.
SHOW_ONLY = "--show-only" in sys.argv
if not SHOW_ONLY:
    matplotlib.use("Agg")
elif "agg" in str(matplotlib.get_backend()).lower():
    gui_backends = ["MacOSX", "TkAgg"] if platform.system() == "Darwin" else ["TkAgg"]
    for backend in gui_backends:
        try:
            matplotlib.use(backend, force=True)
            break
        except (ImportError, ModuleNotFoundError):
            continue

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font',family='Times New Roman')
matplotlib.rcParams['figure.figsize'] = [6.5, 4] # for square canvas
font_format = {'size':24,'weight':1.5}
line_style = ['--','--','-.',':','--','--','-', '-.']
marker = ["8",">","s","p","P","*","h","H"]
color = ['#8dd3c7','#fdb462','#bebada','#80b1d3','#fb8072','#ffffb3','#b3de69','#fccde5']
HATCHES = ["///", "ooo", "xxx", "..."]
matplotlib.rcParams["hatch.linewidth"] = 0.8



METHOD_ORDER = ["plain_heterofl", "inclusive_fl", "fedprox", "pwrh"]
EXCLUDED_METHODS = {"hetero_fedavg"}
METHOD_LABELS = {
    "plain_heterofl": "FedAvg",
    "inclusive_fl": "InclusiveFL",
    "fedprox": "FedProx",
    "pwrh": "PWRH",
}
METRIC_LABELS = {
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": r"$R^2$",
    "nrmse": "NRMSE",
}
MINIMIZE_METRICS = {"mse", "rmse", "mae", "nrmse"}
CLIENT_FILE_RE = re.compile(
    r"^(?P<method>.+)_client_(?P<cid>.+)_L(?P<layer>[0-9]+)_metrics[.]csv$"
)
SERVER_SUFFIX = "_server_metrics.csv"


class TwoDecimalScalarFormatter(ScalarFormatter):
    """Keep exactly two decimal places, including scientific notation."""

    def _set_format(self) -> None:
        self.format = "%1.2f"
        if self._useMathText:
            self.format = r"$\mathdefault{%1.2f}$"


def normalize_metric(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "")
    aliases = {"r^2": "r2", "r²": "r2"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in METRIC_LABELS:
        raise argparse.ArgumentTypeError(
            "metric must be one of: mse, rmse, mae, r2 (or r^2), nrmse"
        )
    return normalized


def ordered_methods(methods: Iterable[str]) -> List[str]:
    available = set(methods) - EXCLUDED_METHODS
    ordered = [method for method in METHOD_ORDER if method in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def read_validation_values(path: Path, metric: str) -> List[Tuple[int, float]]:
    values: List[Tuple[int, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"round", "split", metric}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("{} is missing columns: {}".format(path, sorted(missing)))
        for row in reader:
            if row["split"].strip().lower() != "val" or not row[metric]:
                continue
            value = float(row[metric])
            if np.isfinite(value):
                values.append((int(row["round"]), value))
    return values


def select_best(values: Sequence[Tuple[int, float]], metric: str) -> Tuple[int, float]:
    if not values:
        raise ValueError("Cannot select a best value from an empty validation series.")
    key = (lambda item: item[1])
    if metric in MINIMIZE_METRICS:
        return min(values, key=key)
    return max(values, key=key)


def load_server_best(
    metrics_dir: Path, metric: str
) -> Dict[str, Tuple[int, float]]:
    results: Dict[str, Tuple[int, float]] = {}
    for path in sorted(metrics_dir.glob("*" + SERVER_SUFFIX)):
        method = path.name[: -len(SERVER_SUFFIX)]
        values = read_validation_values(path, metric)
        if values:
            results[method] = select_best(values, metric)
    if not results:
        raise FileNotFoundError(
            "No server validation values for metric '{}' found in {}".format(metric, metrics_dir)
        )
    return results


def load_client_layer_best(
    metrics_dir: Path, metric: str
) -> Dict[int, Dict[str, List[Tuple[str, int, float]]]]:
    # layer -> method -> [(client id, best round, best value)]
    results: DefaultDict[int, DefaultDict[str, List[Tuple[str, int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in sorted(metrics_dir.glob("*_client_*_metrics.csv")):
        match = CLIENT_FILE_RE.match(path.name)
        if match is None:
            continue
        values = read_validation_values(path, metric)
        if not values:
            continue
        best_round, best_value = select_best(values, metric)
        results[int(match.group("layer"))][match.group("method")].append(
            (match.group("cid"), best_round, best_value)
        )
    if not results:
        raise FileNotFoundError(
            "No client validation values for metric '{}' found in {}".format(metric, metrics_dir)
        )
    return {layer: dict(methods) for layer, methods in results.items()}


def read_round_training_times(path: Path) -> List[float]:
    times: List[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "round_train_time_seconds"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("{} is missing columns: {}".format(path, sorted(missing)))
        for row in reader:
            raw_time = row["round_train_time_seconds"]
            if row["split"].strip().lower() != "train" or not raw_time:
                continue
            training_time = float(raw_time)
            if np.isfinite(training_time):
                times.append(training_time)
    return times


def load_client_training_efficiency(
    metrics_dir: Path,
) -> Dict[int, Dict[str, float]]:
    client_times: DefaultDict[int, DefaultDict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in sorted(metrics_dir.glob("*_client_*_metrics.csv")):
        match = CLIENT_FILE_RE.match(path.name)
        if match is None:
            continue
        client_times[int(match.group("layer"))][match.group("method")].extend(
            read_round_training_times(path)
        )

    client_results = {
        layer: {
            method: float(np.mean(times))
            for method, times in methods.items()
            if times
        }
        for layer, methods in client_times.items()
    }
    if not client_results:
        raise FileNotFoundError(
            "No round_train_time_seconds values found in {}".format(metrics_dir)
        )
    return client_results


def format_bar_value(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 0.01 or abs(value) >= 1000:
        return "{:.2e}".format(value)
    return "{:.3f}".format(value)


def apply_bar_style(ax: plt.Axes, metric: str) -> None:
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=14)
    formatter = TwoDecimalScalarFormatter(useMathText=True)
    if metric != "r2":
        formatter.set_powerlimits((-2, 2))
    else:
        formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    # Anchor the multiplier by its right edge so it remains left of the axis.
    # offset_text = ax.yaxis.get_offset_text()
    # offset_text.set_size(13)
    # offset_text.set_horizontalalignment("right")
    # offset_text.set_x(0.0)


def finish_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
    show_only: bool,
) -> List[Path]:
    outputs: List[Path] = []
    if show_only:
        # Display figures sequentially: closing this window creates the next.
        plt.show()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            path = output_dir / "{}.{}".format(stem, extension)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            outputs.append(path)
    plt.close(fig)
    return outputs


def plot_method_bars(
    values_by_method: Dict[str, float],
    metric: str,
    output_dir: Path,
    stem: str,
    dpi: int,
    show_only: bool,
) -> List[Path]:
    methods = ordered_methods(values_by_method)
    values = [values_by_method[method] for method in methods]
    x_positions = np.arange(len(methods))

    fig, ax = plt.subplots()
    bars = ax.bar(
        x_positions,
        values,
        width=0.68,
        color=[color[index % len(color)] for index in range(len(methods))],
        edgecolor="black",
        linewidth=0.8,
    )
    for method_index, bar in enumerate(bars):
        bar.set_hatch(HATCHES[method_index % len(HATCHES)])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [METHOD_LABELS.get(method, method) for method in methods],
        rotation=0,
        ha="center",
    )
    ax.set_xlabel("Heterogeneous FL Methods", fontsize=font_format["size"])
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=font_format["size"])
    apply_bar_style(ax, metric)

    minimum = min(values)
    maximum = max(values)
    span = max(maximum - minimum, max(abs(maximum), abs(minimum)) * 0.1, 1e-12)
    if all(value >= 0 for value in values):
        ax.set_ylim(0, maximum * 1.08 if maximum > 0 else 1.0)
    else:
        margin = 0.15 * span
        ax.set_ylim(min(0.0, minimum - margin), max(0.0, maximum + margin))
    fig.subplots_adjust(left=0.127, right=0.98, bottom=0.157, top=0.933)
    return finish_figure(fig, output_dir, stem, dpi, show_only)


def plot_grouped_bars(
    category_values: Sequence[Tuple[str, Dict[str, float]]],
    metric: str,
    output_dir: Path,
    stem: str,
    dpi: int,
    show_only: bool,
    y_label: Optional[str] = None,
) -> List[Path]:
    methods = ordered_methods(
        method
        for _, values_by_method in category_values
        for method in values_by_method
    )
    if not methods:
        raise ValueError("No methods are available for the grouped bar plot.")

    category_labels = [label for label, _ in category_values]
    x_positions = np.arange(len(category_values))
    bar_width = 0.8 / len(methods)

    fig, ax = plt.subplots()
    plotted_values: List[float] = []
    for method_index, method in enumerate(methods):
        values = [values_by_method.get(method, np.nan) for _, values_by_method in category_values]
        plotted_values.extend(value for value in values if np.isfinite(value))
        offset = (method_index - (len(methods) - 1) / 2.0) * bar_width
        ax.bar(
            x_positions + offset,
            values,
            width=bar_width,
            color=color[method_index % len(color)],
            edgecolor="black",
            linewidth=0.8,
            hatch=HATCHES[method_index % len(HATCHES)],
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(category_labels, rotation=0, ha="center")
    ax.set_xlabel("Evaluation Group", fontsize=font_format["size"])
    ax.set_ylabel(y_label or METRIC_LABELS[metric], fontsize=font_format["size"])
    apply_bar_style(ax, metric)
    ax.legend(
        fontsize=13,
        ncol=len(methods),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        columnspacing=1.0,
        handletextpad=0.4,
        fancybox=True,
    )

    minimum = min(plotted_values)
    maximum = max(plotted_values)
    span = max(maximum - minimum, max(abs(maximum), abs(minimum)) * 0.1, 1e-12)
    if all(value >= 0 for value in plotted_values):
        ax.set_ylim(0, maximum * 1.08 if maximum > 0 else 1.0)
    else:
        margin = 0.15 * span
        ax.set_ylim(min(0.0, minimum - margin), max(0.0, maximum + margin))
    fig.subplots_adjust(left=0.145, right=0.98, bottom=0.158, top=0.87)
    return finish_figure(fig, output_dir, stem, dpi, show_only)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot each method's best validation metric for the server and for clients "
            "averaged within each layer class."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("evaluation_results/20260713_214205"),
        help="Evaluation run directory containing metrics/.",
    )
    parser.add_argument(
        "--metric",
        type=normalize_metric,
        default="rmse",
        metavar="METRIC",
        help="One of: mse, rmse, mae, r2 (or r^2), nrmse. Default: rmse.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to <run-dir>/plots.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--show-only",
        action="store_true",
        help="Display figures sequentially without saving PNG or PDF files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_only and "agg" in str(matplotlib.get_backend()).lower():
        raise RuntimeError(
            "--show-only requires a GUI Matplotlib backend, but the active backend is {}. "
            "Use MPLBACKEND=MacOSX on macOS or MPLBACKEND=TkAgg.".format(
                matplotlib.get_backend()
            )
        )

    run_dir = args.run_dir.resolve()
    metrics_dir = run_dir / "metrics"
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "plots"
    if not metrics_dir.is_dir():
        raise FileNotFoundError("Metrics directory not found: {}".format(metrics_dir))

    server_best = load_server_best(metrics_dir, args.metric)
    client_best = load_client_layer_best(metrics_dir, args.metric)
    output_paths: List[Path] = []

    print("Server best validation {}:".format(args.metric))
    for method in ordered_methods(server_best):
        best_round, best_value = server_best[method]
        print("  {}: {} (round {})".format(method, best_value, best_round))

    client_averages: Dict[int, Dict[str, float]] = {}
    for layer in sorted(client_best):
        averaged = {
            method: float(np.mean([entry[2] for entry in entries]))
            for method, entries in client_best[layer].items()
        }
        client_averages[layer] = averaged
        print("{}-layer client average of per-client best validation {}:".format(layer, args.metric))
        for method in ordered_methods(averaged):
            print("  {}: {}".format(method, averaged[method]))

    missing_layers = [layer for layer in (1, 2, 3) if layer not in client_averages]
    if missing_layers:
        raise ValueError(
            "Cannot create grouped plot; missing client layer classes: {}".format(
                ", ".join(map(str, missing_layers))
            )
        )

    server_values = {
        method: value for method, (_, value) in server_best.items()
    }
    output_paths.extend(
        plot_method_bars(
            server_values,
            args.metric,
            output_dir,
            "server_best_validation_{}_bar".format(args.metric),
            args.dpi,
            args.show_only,
        )
    )
    for layer in (1, 2, 3):
        output_paths.extend(
            plot_method_bars(
                client_averages[layer],
                args.metric,
                output_dir,
                "client_best_validation_{}_layer_{}_bar".format(args.metric, layer),
                args.dpi,
                args.show_only,
            )
        )

    category_values = [
        ("Server", server_values),
        ("Client-L1", client_averages[1]),
        ("Client-L2", client_averages[2]),
        ("Client-L3", client_averages[3]),
    ]
    output_paths.extend(
        plot_grouped_bars(
            category_values,
            args.metric,
            output_dir,
            "grouped_best_validation_{}_bar".format(args.metric),
            args.dpi,
            args.show_only,
        )
    )

    client_training_time = load_client_training_efficiency(metrics_dir)
    missing_time_layers = [layer for layer in (1, 2, 3) if layer not in client_training_time]
    if missing_time_layers:
        raise ValueError(
            "Cannot create training-efficiency plot; missing client layer classes: {}".format(
                ", ".join(map(str, missing_time_layers))
            )
        )

    print("Average round training time in seconds:")
    for group_label, values_by_method in [
        ("Client-L1", client_training_time[1]),
        ("Client-L2", client_training_time[2]),
        ("Client-L3", client_training_time[3]),
    ]:
        print("  {}:".format(group_label))
        for method in ordered_methods(values_by_method):
            print("    {}: {}".format(method, values_by_method[method]))

    training_efficiency_categories = [
        ("Client-L1", client_training_time[1]),
        ("Client-L2", client_training_time[2]),
        ("Client-L3", client_training_time[3]),
    ]
    output_paths.extend(
        plot_grouped_bars(
            training_efficiency_categories,
            "training_time",
            output_dir,
            "grouped_average_round_train_time_bar",
            args.dpi,
            args.show_only,
            y_label="Training Latency (seconds)",
        )
    )

    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()

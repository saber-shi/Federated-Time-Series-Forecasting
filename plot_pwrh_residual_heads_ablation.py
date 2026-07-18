#!/usr/bin/env python3
"""Plot the PWRH residual-head ablation as grouped validation-metric bars."""

import argparse
import csv
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

import matplotlib


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

METRIC_LABELS = {
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": r"$R^2$",
    "nrmse": "NRMSE",
}
MINIMIZE_METRICS = {"mse", "rmse", "mae", "nrmse"}
CONFIG_DIR_RE = re.compile(r"^heads_(?P<heads>[0-9]+)_scale_(?P<scale>[0-9]+p[0-9]+)$")
CLIENT_FILE_RE = re.compile(
    r"^client_(?P<cid>.+)_L(?P<layer>[0-9]+)_metrics[.]csv$"
)
HATCHES = ["///", "xxx", "ooo", "...", "***"]


class TwoDecimalScalarFormatter(ScalarFormatter):
    """Keep exactly two decimal places, including scientific notation."""

    def _set_format(self) -> None:
        self.format = "%1.2f"
        if self._useMathText:
            self.format = r"$\mathdefault{%1.2f}$"


def normalize_metric(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "")
    normalized = {"r^2": "r2", "r²": "r2"}.get(normalized, normalized)
    if normalized not in METRIC_LABELS:
        raise argparse.ArgumentTypeError(
            "metric must be one of: mse, rmse, mae, r2 (or r^2), nrmse"
        )
    return normalized


def discover_configurations(run_dir: Path) -> List[Tuple[int, str, Path]]:
    configurations: List[Tuple[int, str, Path]] = []
    for path in run_dir.iterdir():
        if not path.is_dir():
            continue
        match = CONFIG_DIR_RE.match(path.name)
        if match is None:
            continue
        metrics_dir = path / "metrics"
        if metrics_dir.is_dir():
            configurations.append(
                (int(match.group("heads")), match.group("scale"), metrics_dir)
            )
    configurations.sort(key=lambda item: item[0])
    if not configurations:
        raise FileNotFoundError(
            "No heads_*_scale_*/metrics directories found in {}".format(run_dir)
        )

    scales = {scale for _, scale, _ in configurations}
    if len(scales) != 1:
        raise ValueError(
            "Expected one fixed head scale for a head-count ablation, found: {}".format(
                ", ".join(sorted(scales))
            )
        )
    return configurations


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
        raise ValueError("Cannot select the best value from an empty validation series.")
    if metric in MINIMIZE_METRICS:
        return min(values, key=lambda item: item[1])
    return max(values, key=lambda item: item[1])


def load_ablation_values(
    configurations: Sequence[Tuple[int, str, Path]], metric: str
) -> Tuple[Dict[int, float], Dict[int, Dict[int, float]]]:
    server_values: Dict[int, float] = {}
    layer_values: DefaultDict[int, Dict[int, float]] = defaultdict(dict)

    for heads, _, metrics_dir in configurations:
        server_path = metrics_dir / "server_metrics.csv"
        if server_path.is_file():
            values = read_validation_values(server_path, metric)
            if values:
                _, server_values[heads] = select_best(values, metric)

        per_layer: DefaultDict[int, List[float]] = defaultdict(list)
        for path in sorted(metrics_dir.glob("client_*_metrics.csv")):
            match = CLIENT_FILE_RE.match(path.name)
            if match is None:
                continue
            values = read_validation_values(path, metric)
            if not values:
                continue
            _, best_value = select_best(values, metric)
            per_layer[int(match.group("layer"))].append(best_value)

        for layer, best_values in per_layer.items():
            layer_values[layer][heads] = float(np.mean(best_values))

    if not server_values:
        raise FileNotFoundError("No server validation values were found for {}.".format(metric))
    return server_values, dict(layer_values)


def apply_axis_style(ax: plt.Axes, metric: str) -> None:
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=14)
    formatter = TwoDecimalScalarFormatter(useMathText=True)
    if metric != "r2":
        formatter.set_powerlimits((-2, 2))
    else:
        formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)


def finish_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
    show_only: bool,
) -> List[Path]:
    outputs: List[Path] = []
    if show_only:
        plt.show()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            path = output_dir / "{}.{}".format(stem, extension)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            outputs.append(path)
    plt.close(fig)
    return outputs


def plot_grouped_bars(
    category_values: Sequence[Tuple[str, Dict[int, float]]],
    head_counts: Sequence[int],
    metric: str,
    output_dir: Path,
    dpi: int,
    show_only: bool,
) -> List[Path]:
    x_positions = np.arange(len(category_values))
    bar_width = 0.8 / len(head_counts)
    plotted_values: List[float] = []

    fig, ax = plt.subplots()
    for head_index, heads in enumerate(head_counts):
        values = [values_by_heads.get(heads, np.nan) for _, values_by_heads in category_values]
        plotted_values.extend(value for value in values if np.isfinite(value))
        offset = (head_index - (len(head_counts) - 1) / 2.0) * bar_width
        ax.bar(
            x_positions + offset,
            values,
            width=bar_width,
            color=color[head_index % len(color)],
            edgecolor="black",
            linewidth=0.8,
            hatch=HATCHES[head_index % len(HATCHES)],
            label=r"$K={}$".format(heads),
        )

    if not plotted_values:
        raise ValueError("No finite values are available for the ablation plot.")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for label, _ in category_values], rotation=0, ha="center")
    ax.set_xlabel("Evaluation Group", fontsize=font_format["size"])
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=font_format["size"])
    apply_axis_style(ax, metric)
    ax.legend(
        fontsize=13,
        ncol=len(head_counts),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.225),
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
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.167, top=0.855)
    return finish_figure(
        fig,
        output_dir,
        "pwrh_residual_heads_ablation_{}".format(metric),
        dpi,
        show_only,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the effect of PWRH residual-head count using each server/client's best "
            "validation metric and layer-averaged client values."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("evaluation_results/pwrh_ablation_20260715_113440"),
        help="Ablation directory containing heads_*_scale_*/metrics subdirectories.",
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
        help="Display the figure without saving PNG or PDF files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_only and "agg" in str(matplotlib.get_backend()).lower():
        raise RuntimeError(
            "--show-only requires a GUI Matplotlib backend, but the active backend is {}.".format(
                matplotlib.get_backend()
            )
        )

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "plots"
    configurations = discover_configurations(run_dir)
    server_values, layer_values = load_ablation_values(configurations, args.metric)

    missing_layers = [layer for layer in (1, 2, 3) if layer not in layer_values]
    if missing_layers:
        raise ValueError(
            "Cannot create grouped plot; missing client layer classes: {}".format(
                ", ".join(map(str, missing_layers))
            )
        )

    category_values = [
        ("Server", server_values),
        ("Client-L1", layer_values[1]),
        ("Client-L2", layer_values[2]),
        ("Client-L3", layer_values[3]),
    ]
    head_counts = [heads for heads, _, _ in configurations]
    output_paths = plot_grouped_bars(
        category_values,
        head_counts,
        args.metric,
        output_dir,
        args.dpi,
        args.show_only,
    )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()

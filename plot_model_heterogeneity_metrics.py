#!/usr/bin/env python3
"""Plot best-validation metrics across the model-heterogeneity beta sweep."""

import argparse
import csv
import platform
import re
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

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
# Match mathtext (the "x10^exponent" offset text) to the surrounding Times
# New Roman text; matplotlib's mathtext otherwise falls back to its own
# default font for anything inside $...$.
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['figure.figsize'] = [6.5, 4] # for square canvas
font_format = {'size':24,'weight':1.5}
line_style = ['--','-.',':','--','--','-', '-.']
marker = ["8",">","p","s","P","*","h","H"]
color = ['#9ec4be','#abd0f1','#c59d94','#e56f5e','#80b1d3','#fb8072','#ffffb3','#b3de69','#fccde5']


matplotlib.rcParams["mathtext.fontset"] = "stix"

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
SERVER_SUFFIX = "_server_metrics.csv"
BETA_DIR_RE = re.compile(r"^beta_(?P<value>[0-9]+(?:_[0-9]+)?)$")


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


def ordered_methods(methods: Iterable[str]) -> List[str]:
    available = set(methods) - EXCLUDED_METHODS
    ordered = [method for method in METHOD_ORDER if method in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def parse_beta_dir(path: Path) -> Tuple[float, str]:
    match = BETA_DIR_RE.match(path.name)
    if match is None:
        raise ValueError("Invalid beta directory name: {}".format(path.name))
    parts = match.group("value").split("_")
    fraction = Fraction(int(parts[0]), int(parts[1])) if len(parts) == 2 else Fraction(int(parts[0]))
    label = str(fraction.numerator)
    if fraction.denominator != 1:
        label = "{}/{}".format(fraction.numerator, fraction.denominator)
    return float(fraction), label


def discover_beta_runs(run_dir: Path) -> List[Tuple[float, str, Path]]:
    runs: List[Tuple[float, str, Path]] = []
    for path in run_dir.iterdir():
        if not path.is_dir() or BETA_DIR_RE.match(path.name) is None:
            continue
        beta, label = parse_beta_dir(path)
        metrics_dir = path / "metrics"
        if metrics_dir.is_dir():
            runs.append((beta, label, metrics_dir))
    runs.sort(key=lambda item: item[0])
    if not runs:
        raise FileNotFoundError("No beta_*/metrics directories found in {}".format(run_dir))
    return runs


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


def load_server_sweep_values(
    beta_runs: Sequence[Tuple[float, str, Path]], metric: str
) -> Dict[str, Dict[float, float]]:
    server_values: DefaultDict[str, Dict[float, float]] = defaultdict(dict)

    for beta, _, metrics_dir in beta_runs:
        for path in sorted(metrics_dir.glob("*" + SERVER_SUFFIX)):
            method = path.name[: -len(SERVER_SUFFIX)]
            validation_values = read_validation_values(path, metric)
            if validation_values:
                _, best_value = select_best(validation_values, metric)
                server_values[method][beta] = best_value
    if not server_values:
        raise FileNotFoundError("No server validation metrics found in the beta sweep.")
    return dict(server_values)


def apply_axis_style(ax: plt.Axes, metric: str) -> None:
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=16)
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


def plot_beta_lines(
    values_by_method: Dict[str, Dict[float, float]],
    beta_runs: Sequence[Tuple[float, str, Path]],
    metric: str,
    output_dir: Path,
    stem: str,
    dpi: int,
    show_only: bool,
) -> List[Path]:
    methods = ordered_methods(values_by_method)
    if not methods:
        return []

    fig, ax = plt.subplots()
    for method_index, method in enumerate(methods):
        beta_values = values_by_method[method]
        x_values = sorted(beta_values)
        ax.plot(
            x_values,
            [beta_values[beta] for beta in x_values],
            label=METHOD_LABELS.get(method, method),
            linewidth=2.0,
            linestyle=line_style[method_index % len(line_style)],
            marker=marker[method_index % len(marker)],
            markersize=7,
            color=color[method_index % len(color)],
        )

    beta_ticks = [beta for beta, _, _ in beta_runs]
    beta_labels = [label for _, label, _ in beta_runs]
    ax.set_xticks(beta_ticks)
    ax.set_xticklabels(beta_labels)
    ax.set_xlim(min(beta_ticks) - 0.03, max(beta_ticks) + 0.03)
    ax.set_xlabel(r"Heterogeneity Level ($\beta$)", fontsize=font_format["size"])
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=font_format["size"])
    apply_axis_style(ax, metric)
    ax.legend(
        fontsize=13,
        ncol=len(methods),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        columnspacing=1.0,
        handletextpad=0.4,
        fancybox=True,
    )
    fig.subplots_adjust(left=0.133, right=0.98, bottom=0.176, top=0.85)
    return finish_figure(fig, output_dir, stem, dpi, show_only)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot best-validation server metrics across the model-heterogeneity beta sweep."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("evaluation_results/model_heterogeneity_20260714_204346"),
        help="Sweep directory containing beta_*/metrics subdirectories.",
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
            "--show-only requires a GUI Matplotlib backend, but the active backend is {}.".format(
                matplotlib.get_backend()
            )
        )

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "plots"
    beta_runs = discover_beta_runs(run_dir)
    server_values = load_server_sweep_values(beta_runs, args.metric)

    output_paths: List[Path] = []
    output_paths.extend(
        plot_beta_lines(
            server_values,
            beta_runs,
            args.metric,
            output_dir,
            "heterogeneity_server_best_validation_{}".format(args.metric),
            args.dpi,
            args.show_only,
        )
    )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot how plain-HeteroFL's client model_rate (submodel width) affects performance."""

import argparse
import csv
import platform
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

from plot_style import color, font_format, line_style, marker


matplotlib.rcParams["mathtext.fontset"] = "stix"

METRIC_LABELS = {
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": r"$R^2$",
    "nrmse": "NRMSE",
}
MINIMIZE_METRICS = {"mse", "rmse", "mae", "nrmse"}
CONFIG_DIR_RE = re.compile(r"^rate_(?P<rate>[0-9]+p[0-9]+)$")
CLIENT_FILE_RE = re.compile(r"^client_(?P<cid>.+)_metrics[.]csv$")


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


def rate_from_slug(slug: str) -> float:
    return float(slug.replace("p", "."))


def discover_configurations(run_dir: Path) -> List[Tuple[float, Path]]:
    configurations: List[Tuple[float, Path]] = []
    for path in run_dir.iterdir():
        if not path.is_dir():
            continue
        match = CONFIG_DIR_RE.match(path.name)
        if match is None:
            continue
        metrics_dir = path / "metrics"
        if metrics_dir.is_dir():
            configurations.append((rate_from_slug(match.group("rate")), metrics_dir))
    configurations.sort(key=lambda item: item[0])
    if not configurations:
        raise FileNotFoundError("No rate_*/metrics directories found in {}".format(run_dir))
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
    configurations: Sequence[Tuple[float, Path]], metric: str
) -> Tuple[Dict[float, float], Dict[float, float], Dict[float, float]]:
    # rate -> server best value; rate -> mean/std of per-client best values.
    server_values: Dict[float, float] = {}
    client_mean: Dict[float, float] = {}
    client_std: Dict[float, float] = {}

    for rate, metrics_dir in configurations:
        server_path = metrics_dir / "server_metrics.csv"
        if server_path.is_file():
            values = read_validation_values(server_path, metric)
            if values:
                _, server_values[rate] = select_best(values, metric)

        per_client: List[float] = []
        for path in sorted(metrics_dir.glob("client_*_metrics.csv")):
            if CLIENT_FILE_RE.match(path.name) is None:
                continue
            values = read_validation_values(path, metric)
            if not values:
                continue
            _, best_value = select_best(values, metric)
            per_client.append(best_value)

        if per_client:
            client_mean[rate] = float(np.mean(per_client))
            client_std[rate] = float(np.std(per_client))

    if not server_values:
        raise FileNotFoundError("No server validation values were found for {}.".format(metric))
    return server_values, client_mean, client_std


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
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_size(13)
    offset_text.set_horizontalalignment("right")
    offset_text.set_x(0.0)


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


def plot_model_rate_effect(
    rates: Sequence[float],
    server_values: Dict[float, float],
    client_mean: Dict[float, float],
    client_std: Dict[float, float],
    metric: str,
    output_dir: Path,
    dpi: int,
    show_only: bool,
) -> List[Path]:
    fig, ax = plt.subplots()

    server_ys = np.array([server_values.get(rate, np.nan) for rate in rates], dtype=float)
    client_ys = np.array([client_mean.get(rate, np.nan) for rate in rates], dtype=float)
    client_err = np.array([client_std.get(rate, 0.0) for rate in rates], dtype=float)

    ax.plot(
        rates,
        server_ys,
        label="Server",
        linewidth=2,
        linestyle=line_style[0],
        color=color[0],
        marker=marker[0],
        markersize=7,
    )
    ax.plot(
        rates,
        client_ys,
        label="Client average",
        linewidth=2,
        linestyle=line_style[1],
        color=color[1],
        marker=marker[1],
        markersize=7,
    )
    ax.fill_between(
        rates,
        client_ys - client_err,
        client_ys + client_err,
        color=color[1],
        alpha=0.15,
        linewidth=0,
    )

    ax.set_xlabel("Model rate (submodel width)", fontsize=font_format["size"])
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=font_format["size"])
    apply_axis_style(ax, metric)
    ax.set_xticks(list(rates))
    ax.set_xticklabels(["{:g}".format(rate) for rate in rates])
    ax.legend(
        fontsize=13,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        fancybox=True,
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.85)
    return finish_figure(
        fig,
        output_dir,
        "plain_heterofl_model_rate_ablation_{}".format(metric),
        dpi,
        show_only,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot how plain-HeteroFL's client model_rate (submodel width) affects "
            "the server's and clients' best validation metric."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Ablation directory containing rate_*/metrics subdirectories "
        "(from run_plain_heterofl_model_rate_ablation.sh).",
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
    server_values, client_mean, client_std = load_ablation_values(configurations, args.metric)
    rates = [rate for rate, _ in configurations]

    print("Best validation {} by model_rate:".format(args.metric))
    for rate in rates:
        server_val = server_values.get(rate)
        client_val = client_mean.get(rate)
        print(
            "  rate={:g}: server={} client_avg={}".format(
                rate,
                "n/a" if server_val is None else "{:.6g}".format(server_val),
                "n/a" if client_val is None else "{:.6g}".format(client_val),
            )
        )

    output_paths = plot_model_rate_effect(
        rates,
        server_values,
        client_mean,
        client_std,
        args.metric,
        output_dir,
        args.dpi,
        args.show_only,
    )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()

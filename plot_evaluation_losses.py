#!/usr/bin/env python3
"""Plot server and layer-averaged client loss from an evaluation run."""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import matplotlib

# Use a non-interactive backend for normal file generation. In --show-only
# mode, preserve the system's GUI backend so plt.show() can open a window.
if "--show-only" not in sys.argv:
    matplotlib.use('Agg')
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
color = ['#4363d8','#8dd3c7','#fdb462','#bebada','#80b1d3','#fb8072','#ffffb3','#b3de69','#fccde5']
color = ['#ffe119', '#8dd3c7', '#4363d8', '#f58231']

METHOD_ORDER = ["inclusive_fl", "plain_heterofl", "fedprox", "pwrh"]
EXCLUDED_METHODS = {"hetero_fedavg"}
METHOD_LABELS = {
    "inclusive_fl": "InclusiveFL",
    "plain_heterofl": "FedAvg",
    "fedprox": "FedProx",
    "pwrh": "PWRH",
}
CLIENT_FILE_RE = re.compile(
    r"^(?P<method>.+)_client_(?P<cid>.+)_L(?P<layer>[0-9]+)_metrics[.]csv$"
)
SERVER_SUFFIX = "_server_metrics.csv"


def read_loss_rows(path: Path) -> Iterable[Tuple[int, str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"round", "split", "loss"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("{} is missing columns: {}".format(path, sorted(missing)))
        for row in reader:
            if not row["round"] or not row["loss"]:
                continue
            yield int(row["round"]), row["split"].strip().lower(), float(row["loss"])


def ordered_methods(methods: Iterable[str]) -> List[str]:
    available = set(methods) - EXCLUDED_METHODS
    ordered = [method for method in METHOD_ORDER if method in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def load_server_losses(metrics_dir: Path) -> Dict[str, Dict[str, Dict[int, float]]]:
    losses: Dict[str, Dict[str, Dict[int, float]]] = {}
    for path in sorted(metrics_dir.glob("*" + SERVER_SUFFIX)):
        method = path.name[: -len(SERVER_SUFFIX)]
        method_data: DefaultDict[str, Dict[int, float]] = defaultdict(dict)
        for round_number, split, loss in read_loss_rows(path):
            method_data[split][round_number] = loss
        losses[method] = dict(method_data)
    if not losses:
        raise FileNotFoundError("No server metric CSV files found in {}".format(metrics_dir))
    return losses


def load_client_losses(
    metrics_dir: Path,
) -> Dict[str, Dict[str, Dict[int, Dict[int, List[float]]]]]:
    # method -> split -> layer -> round -> losses from clients in that layer
    losses: DefaultDict[
        str, DefaultDict[str, DefaultDict[int, DefaultDict[int, List[float]]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    matched_files = 0
    for path in sorted(metrics_dir.glob("*_client_*_metrics.csv")):
        match = CLIENT_FILE_RE.match(path.name)
        if match is None:
            continue
        matched_files += 1
        method = match.group("method")
        layer = int(match.group("layer"))
        for round_number, split, loss in read_loss_rows(path):
            losses[method][split][layer][round_number].append(loss)

    if matched_files == 0:
        raise FileNotFoundError("No client metric CSV files found in {}".format(metrics_dir))
    return losses


def apply_axis_style(ax: plt.Axes, y_label: str) -> None:
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=16)

    # Show the shared order-of-magnitude as a "x10^exponent" marker to the
    # left of the axis, above the y-tick labels, instead of repeating it on
    # every tick.
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(formatter)
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_size(14)
    offset_text.set_horizontalalignment("right")
    offset_text.set_x(0.0)

    ax.set_ylabel(y_label, fontsize=font_format["size"])


def draw_method_line(
    ax: plt.Axes,
    method: str,
    rounds: Sequence[int],
    values: Sequence[float],
    method_index: int,
    label: bool = True,
) -> None:
    mark_every = max(1, len(rounds) // 10)
    ax.plot(
        rounds,
        values,
        label=METHOD_LABELS.get(method, method) if label else None,
        linewidth=2,
        linestyle=line_style[method_index % len(line_style)],
        color=color[method_index % len(color)],
        marker=marker[method_index % len(marker)],
        markersize=6,
        markevery=mark_every,
    )


def finish_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
    show_only: bool,
) -> List[Path]:
    outputs = []
    if show_only:
        # Block until this window is closed, then let the caller create the
        # next figure.
        plt.show()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            path = output_dir / "{}.{}".format(stem, extension)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            outputs.append(path)
    plt.close(fig)
    return outputs


def plot_server_split(
    server_losses: Dict[str, Dict[str, Dict[int, float]]],
    split: str,
    output_dir: Path,
    dpi: int,
    show_only: bool,
) -> List[Path]:
    methods = ordered_methods(server_losses)
    fig, ax = plt.subplots()
    plotted = 0
    for method_index, method in enumerate(methods):
        round_to_loss = server_losses[method].get(split, {})
        if not round_to_loss:
            continue
        rounds = sorted(round_to_loss)
        draw_method_line(
            ax,
            method,
            rounds,
            [round_to_loss[round_number] for round_number in rounds],
            method_index,
        )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return []

    ax.set_xlabel("Communication Round", fontsize=font_format["size"])
    ax.set_xlim(left=1)
    apply_axis_style(ax, "Loss")
    ax.legend(
        fontsize=14,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        handlelength=1.5,     # width of each line sample
        handletextpad=0.4,    # gap between line sample and label
        columnspacing=1.0,    # horizontal gap between legend columns
        labelspacing=0.4,     # vertical gap between rows
        borderpad=0.4,        # padding inside the legend frame
        fancybox=True,
    )
    fig.subplots_adjust(left=0.121, right=0.99, bottom=0.16, top=0.868)
    return finish_figure(fig, output_dir, "server_{}_loss".format(split), dpi, show_only)


def plot_client_split_by_layer(
    client_losses: Dict[str, Dict[str, Dict[int, Dict[int, List[float]]]]],
    split: str,
    output_dir: Path,
    dpi: int,
    show_only: bool,
) -> List[Path]:
    methods = ordered_methods(client_losses)
    layers = sorted(
        {
            layer
            for method in methods
            for layer in client_losses[method].get(split, {})
        }
    )
    if not layers:
        return []

    output_paths: List[Path] = []
    for layer in layers:
        fig, ax = plt.subplots()
        for method_index, method in enumerate(methods):
            round_to_losses = client_losses[method].get(split, {}).get(layer, {})
            if not round_to_losses:
                continue
            rounds = sorted(round_to_losses)
            means = np.asarray(
                [np.mean(round_to_losses[round_number]) for round_number in rounds], dtype=float
            )
            draw_method_line(ax, method, rounds, means, method_index)

        ax.set_xlabel("Communication Round", fontsize=font_format["size"])
        ax.set_xlim(left=1)
        apply_axis_style(ax, "Loss")
        ax.legend(
            fontsize=14,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
            handlelength=1.5,     # width of each line sample
            handletextpad=0.4,    # gap between line sample and label
            columnspacing=1.0,    # horizontal gap between legend columns
            labelspacing=0.4,     # vertical gap between rows
            borderpad=0.4,        # padding inside the legend frame
            fancybox=True,
        )
        fig.subplots_adjust(left=0.115, right=0.99, bottom=0.16, top=0.868)
        output_paths.extend(
            finish_figure(
                fig,
                output_dir,
                "client_{}_loss_layer_{}".format(split, layer),
                dpi,
                show_only,
            )
        )
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot server and layer-averaged client losses from evaluation metric CSVs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("evaluation_results/20260713_214205"),
        help="Evaluation run directory containing metrics/.",
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
        help="Display figures interactively instead of saving PNG or PDF files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metrics_dir = run_dir / "metrics"
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "plots"
    if not metrics_dir.is_dir():
        raise FileNotFoundError("Metrics directory not found: {}".format(metrics_dir))

    server_losses = load_server_losses(metrics_dir)
    client_losses = load_client_losses(metrics_dir)
    output_paths: List[Path] = []
    output_paths.extend(
        plot_server_split(server_losses, "train", output_dir, args.dpi, args.show_only)
    )
    output_paths.extend(
        plot_client_split_by_layer(
            client_losses,
            "train",
            output_dir,
            args.dpi,
            args.show_only,
        )
    )

    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()

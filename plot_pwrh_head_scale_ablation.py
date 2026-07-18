#!/usr/bin/env python3
"""Plot the PWRH residual-head scale ablation as grouped metric bars."""

import argparse
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

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

from plot_pwrh_residual_heads_ablation import (
    CLIENT_FILE_RE,
    HATCHES,
    METRIC_LABELS,
    apply_axis_style,
    color,
    finish_figure,
    font_format,
    normalize_metric,
    read_validation_values,
    select_best,
)


CONFIG_DIR_RE = re.compile(
    r"^heads_(?P<heads>[0-9]+)_scale_(?P<scale>[0-9]+p[0-9]+)$"
)


def scale_token_to_float(token: str) -> float:
    return float(token.replace("p", "."))


def format_scale(scale: float) -> str:
    return "{:g}".format(scale)


def discover_configurations(run_dir: Path) -> List[Tuple[float, int, Path]]:
    configurations: List[Tuple[float, int, Path]] = []
    for path in run_dir.iterdir():
        if not path.is_dir():
            continue
        match = CONFIG_DIR_RE.match(path.name)
        if match is None:
            continue
        metrics_dir = path / "metrics"
        if metrics_dir.is_dir():
            configurations.append(
                (
                    scale_token_to_float(match.group("scale")),
                    int(match.group("heads")),
                    metrics_dir,
                )
            )
    configurations.sort(key=lambda item: item[0])
    if not configurations:
        raise FileNotFoundError(
            "No heads_*_scale_*/metrics directories found in {}".format(run_dir)
        )

    head_counts = {heads for _, heads, _ in configurations}
    if len(head_counts) != 1:
        raise ValueError(
            "Expected one fixed residual-head count for a scale ablation, found: {}".format(
                ", ".join(map(str, sorted(head_counts)))
            )
        )
    return configurations


def load_ablation_values(
    configurations: Sequence[Tuple[float, int, Path]], metric: str
) -> Tuple[Dict[float, float], Dict[int, Dict[float, float]]]:
    server_values: Dict[float, float] = {}
    layer_values: DefaultDict[int, Dict[float, float]] = defaultdict(dict)

    for scale, _, metrics_dir in configurations:
        server_path = metrics_dir / "server_metrics.csv"
        if server_path.is_file():
            values = read_validation_values(server_path, metric)
            if values:
                _, server_values[scale] = select_best(values, metric)

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
            layer_values[layer][scale] = float(np.mean(best_values))

    if not server_values:
        raise FileNotFoundError("No server validation values were found for {}.".format(metric))
    return server_values, dict(layer_values)


def plot_grouped_bars(
    category_values: Sequence[Tuple[str, Dict[float, float]]],
    scales: Sequence[float],
    metric: str,
    output_dir: Path,
    dpi: int,
    show_only: bool,
) -> List[Path]:
    x_positions = np.arange(len(category_values))
    bar_width = 0.8 / len(scales)
    plotted_values: List[float] = []

    fig, ax = plt.subplots()
    for scale_index, scale in enumerate(scales):
        values = [values_by_scale.get(scale, np.nan) for _, values_by_scale in category_values]
        plotted_values.extend(value for value in values if np.isfinite(value))
        offset = (scale_index - (len(scales) - 1) / 2.0) * bar_width
        ax.bar(
            x_positions + offset,
            values,
            width=bar_width,
            color=color[scale_index % len(color)],
            edgecolor="black",
            linewidth=0.8,
            hatch=HATCHES[scale_index % len(HATCHES)],
            label=r"$\lambda={}$".format(format_scale(scale)),
        )

    if not plotted_values:
        raise ValueError("No finite values are available for the scale-ablation plot.")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for label, _ in category_values], rotation=0, ha="center")
    ax.set_xlabel("Evaluation Group", fontsize=font_format["size"])
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=font_format["size"])
    apply_axis_style(ax, metric)
    ax.legend(
        fontsize=13,
        ncol=len(scales),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        columnspacing=0.8,
        handletextpad=0.35,
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
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.17, top=0.87)
    return finish_figure(
        fig,
        output_dir,
        "pwrh_head_scale_ablation_{}".format(metric),
        dpi,
        show_only,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the effect of PWRH residual-head scale using each server/client's best "
            "validation metric and layer-averaged client values."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("evaluation_results/pwrh_ablation_20260715_145914"),
        help="Scale-ablation directory containing heads_*_scale_*/metrics.",
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
    scales = [scale for scale, _, _ in configurations]
    output_paths = plot_grouped_bars(
        category_values,
        scales,
        args.metric,
        output_dir,
        args.dpi,
        args.show_only,
    )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()

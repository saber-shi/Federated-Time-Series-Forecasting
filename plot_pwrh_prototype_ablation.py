#!/usr/bin/env python3
"""Plot the PWRH prototype ablation as grouped validation-metric bars."""

import argparse
import platform
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


MODE_ORDER = ["adaptive", "uniform", "frozen", "random", "hard"]
MODE_LABELS = {
    "adaptive": "Adaptive",
    "uniform": "Uniform",
    "frozen": "Frozen",
    "random": "Random",
    "hard": "Hard",
}


def discover_configurations(run_dir: Path) -> List[Tuple[str, Path]]:
    configurations: List[Tuple[str, Path]] = []
    for mode in MODE_ORDER:
        metrics_dir = run_dir / mode / "metrics"
        if metrics_dir.is_dir():
            configurations.append((mode, metrics_dir))
    if not configurations:
        raise FileNotFoundError(
            "No prototype-mode metrics directories found in {}".format(run_dir)
        )
    missing_modes = [mode for mode in MODE_ORDER if not (run_dir / mode / "metrics").is_dir()]
    if missing_modes:
        raise FileNotFoundError(
            "Missing prototype modes: {}".format(", ".join(missing_modes))
        )
    return configurations


def load_ablation_values(
    configurations: Sequence[Tuple[str, Path]], metric: str
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    server_values: Dict[str, float] = {}
    layer_values: DefaultDict[int, Dict[str, float]] = defaultdict(dict)

    for mode, metrics_dir in configurations:
        server_path = metrics_dir / "server_metrics.csv"
        if server_path.is_file():
            values = read_validation_values(server_path, metric)
            if values:
                _, server_values[mode] = select_best(values, metric)

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
            layer_values[layer][mode] = float(np.mean(best_values))

    if not server_values:
        raise FileNotFoundError("No server validation values were found for {}.".format(metric))
    return server_values, dict(layer_values)


def plot_grouped_bars(
    category_values: Sequence[Tuple[str, Dict[str, float]]],
    modes: Sequence[str],
    metric: str,
    output_dir: Path,
    dpi: int,
    show_only: bool,
) -> List[Path]:
    x_positions = np.arange(len(category_values))
    bar_width = 0.8 / len(modes)
    plotted_values: List[float] = []

    fig, ax = plt.subplots()
    for mode_index, mode in enumerate(modes):
        values = [values_by_mode.get(mode, np.nan) for _, values_by_mode in category_values]
        plotted_values.extend(value for value in values if np.isfinite(value))
        offset = (mode_index - (len(modes) - 1) / 2.0) * bar_width
        ax.bar(
            x_positions + offset,
            values,
            width=bar_width,
            color=color[mode_index % len(color)],
            edgecolor="black",
            linewidth=0.8,
            hatch=HATCHES[mode_index % len(HATCHES)],
            label=MODE_LABELS.get(mode, mode.title()),
        )

    if not plotted_values:
        raise ValueError("No finite values are available for the prototype-ablation plot.")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for label, _ in category_values], rotation=0, ha="center")
    ax.set_xlabel("Evaluation Group", fontsize=font_format["size"])
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=font_format["size"])
    apply_axis_style(ax, metric)
    ax.legend(
        fontsize=12,
        ncol=len(modes),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        columnspacing=0.75,
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
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.171, top=0.86)
    return finish_figure(
        fig,
        output_dir,
        "pwrh_prototype_ablation_{}".format(metric),
        dpi,
        show_only,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the PWRH prototype ablation using each server/client's best validation "
            "metric and layer-averaged client values."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("evaluation_results/pwrh_prototype_ablation_20260715_191555"),
        help="Prototype-ablation directory containing one subdirectory per mode.",
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
    modes = [mode for mode, _ in configurations]
    output_paths = plot_grouped_bars(
        category_values,
        modes,
        args.metric,
        output_dir,
        args.dpi,
        args.show_only,
    )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()

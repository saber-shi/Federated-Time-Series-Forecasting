#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AVAILABLE_METRICS = ["loss", "mse", "rmse", "mae", "r2", "nrmse"]
DEFAULT_METRICS = ["mse", "rmse", "mae", "nrmse"]
SERVER_FILES = {
    "hetero_fedavg": "hetero_fedavg_server_metrics.csv",
    "inclusive_fl": "inclusive_fl_server_metrics.csv",
    "plain": "plain_heterofl_server_metrics.csv",
    "fedprox": "fedprox_server_metrics.csv",
    "pwrh": "pwrh_server_metrics.csv",
}
METHOD_LABELS = {
    "hetero_fedavg": "Hetero FedAvg",
    "inclusive_fl": "InclusiveFL",
    "plain": "Plain HeteroFL",
    "fedprox": "FedProx",
    "pwrh": "PWRH",
}
METHOD_COLORS = {
    "hetero_fedavg": "#7B2CBF",
    "inclusive_fl": "#F2C14E",
    "plain": "#2E86AB",
    "fedprox": "#6A994E",
    "pwrh": "#E07A5F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final validation server metrics in one bar chart for heterogeneous FedAvg, InclusiveFL, HeteroFL, FedProx, and PWRH."
    )
    parser.add_argument("--log_dir", type=str, default="benchmark_logs")
    parser.add_argument("--output_path", type=str, default="benchmark_logs/figures/server_metric_bars.png")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        choices=AVAILABLE_METRICS,
        help="Validation metrics to plot on the x-axis.",
    )
    return parser.parse_args()


def read_final_val_row(path: Path) -> Dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == "val"]
    if not rows:
        raise ValueError(f"No validation rows found in {path}")
    return rows[-1]


def collect_server_results(log_dir: Path, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for method, filename in SERVER_FILES.items():
        path = log_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing server metrics file: {path}")
        final_row = read_final_val_row(path)
        results[method] = {metric: float(final_row[metric]) for metric in metrics}
    return results


def plot_server_metric_bars(results: Dict[str, Dict[str, float]], metrics: List[str], output_path: Path) -> Path:
    x_positions = list(range(len(metrics)))
    methods = list(results.keys())
    width = min(0.28, 0.8 / max(1, len(methods)))

    fig, ax = plt.subplots(figsize=(max(9, len(metrics) * 1.2), 6))
    for idx, method in enumerate(methods):
        offset = (idx - (len(methods) - 1) / 2) * width
        values = [results[method][metric] for metric in metrics]
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )

    ax.set_title("Final Validation Server Metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([metric.upper() for metric in metrics])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    output_path = Path(args.output_path)

    results = collect_server_results(log_dir, args.metrics)
    saved_path = plot_server_metric_bars(results, args.metrics, output_path)
    print(f"Saved server comparison figure: {saved_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ["loss", "mse", "rmse", "mae", "r2", "nrmse"]
METHOD_PATTERNS = {
    "hetero_fedavg": "hetero_fedavg_client_*_metrics.csv",
    "inclusive_fl": "inclusive_fl_client_*_metrics.csv",
    "plain": "plain_heterofl_client_*_metrics.csv",
    "fedprox": "fedprox_client_*_metrics.csv",
    "pwrh": "pwrh_client_*_metrics.csv",
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
        description="Plot final validation client metrics as bar charts for heterogeneous FedAvg, InclusiveFL, HeteroFL, FedProx, and PWRH."
    )
    parser.add_argument("--log_dir", type=str, default="benchmark_logs")
    parser.add_argument("--output_dir", type=str, default="benchmark_logs/figures/client_metric_bars")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=METRICS,
        choices=METRICS,
        help="Validation metrics to plot.",
    )
    return parser.parse_args()


def read_final_val_row(path: Path) -> Dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == "val"]
    if not rows:
        raise ValueError(f"No validation rows found in {path}")
    return rows[-1]


def collect_method_results(log_dir: Path, method: str) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for path in sorted(log_dir.glob(METHOD_PATTERNS[method])):
        final_row = read_final_val_row(path)
        cid = final_row["cid"]
        results[cid] = {metric: float(final_row[metric]) for metric in METRICS}
    return results


def ordered_clients(*method_results: Dict[str, Dict[str, float]]) -> List[str]:
    clients = set()
    for result in method_results:
        clients.update(result.keys())
    return sorted(clients)


def plot_metric(
    metric: str,
    clients: List[str],
    method_results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
) -> Path:
    x_positions = list(range(len(clients)))
    methods = list(method_results.keys())
    width = min(0.28, 0.8 / max(1, len(methods)))

    fig, ax = plt.subplots(figsize=(max(10, len(clients) * 1.2), 6))
    for idx, method in enumerate(methods):
        offset = (idx - (len(methods) - 1) / 2) * width
        values = [method_results[method][cid][metric] for cid in clients]
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )

    ax.set_title(f"Final Validation {metric.upper()} by Client")
    ax.set_xlabel("Client ID")
    ax.set_ylabel(metric.upper())
    ax.set_xticks(x_positions)
    ax.set_xticklabels(clients, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    output_path = output_dir / f"client_bar_{metric}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def validate_results(
    method_results: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    for method, results in method_results.items():
        if not results:
            raise ValueError(f"No {METHOD_LABELS[method]} client metric files were found.")
    reference_method = next(iter(method_results))
    reference_clients = set(method_results[reference_method])
    for method, results in method_results.items():
        missing_in_reference = sorted(set(results) - reference_clients)
        missing_in_method = sorted(reference_clients - set(results))
        if missing_in_reference or missing_in_method:
            raise ValueError(
                "Client mismatch between methods. "
                f"{method} has extra clients: {missing_in_reference}. "
                f"{method} is missing clients: {missing_in_method}."
            )
    return method_results


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_results = {
        method: collect_method_results(log_dir, method)
        for method in METHOD_PATTERNS
    }
    method_results = validate_results(method_results)
    clients = ordered_clients(*method_results.values())

    saved_paths = []
    for metric in args.metrics:
        saved_paths.append(plot_metric(metric, clients, method_results, output_dir))

    print("Saved client comparison figures:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

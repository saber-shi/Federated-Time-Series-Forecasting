#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ["loss", "mse", "rmse", "mae", "r2", "nrmse"]
METHOD_PATTERNS = {
    "plain": "plain_heterofl_client_*_metrics.csv",
    "spa": "spa_hfl_client_*_metrics.csv",
}
METHOD_LABELS = {
    "plain": "Plain HeteroFL",
    "spa": "SPA-HFL",
}
METHOD_COLORS = {
    "plain": "#2E86AB",
    "spa": "#E07A5F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final validation client metrics as bar charts for plain HeteroFL vs SPA-HFL."
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
    plain_results: Dict[str, Dict[str, float]],
    spa_results: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> Path:
    x_positions = list(range(len(clients)))
    width = 0.38

    plain_values = [plain_results[cid][metric] for cid in clients]
    spa_values = [spa_results[cid][metric] for cid in clients]

    fig, ax = plt.subplots(figsize=(max(10, len(clients) * 1.2), 6))
    ax.bar(
        [x - width / 2 for x in x_positions],
        plain_values,
        width=width,
        label=METHOD_LABELS["plain"],
        color=METHOD_COLORS["plain"],
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        spa_values,
        width=width,
        label=METHOD_LABELS["spa"],
        color=METHOD_COLORS["spa"],
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
    plain_results: Dict[str, Dict[str, float]],
    spa_results: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    if not plain_results:
        raise ValueError("No plain client metric files were found.")
    if not spa_results:
        raise ValueError("No SPA client metric files were found.")
    missing_in_plain = sorted(set(spa_results) - set(plain_results))
    missing_in_spa = sorted(set(plain_results) - set(spa_results))
    if missing_in_plain or missing_in_spa:
        raise ValueError(
            "Client mismatch between methods. "
            f"Missing in plain: {missing_in_plain}. Missing in SPA: {missing_in_spa}."
        )
    return plain_results, spa_results


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plain_results = collect_method_results(log_dir, "plain")
    spa_results = collect_method_results(log_dir, "spa")
    plain_results, spa_results = validate_results(plain_results, spa_results)
    clients = ordered_clients(plain_results, spa_results)

    saved_paths = []
    for metric in args.metrics:
        saved_paths.append(plot_metric(metric, clients, plain_results, spa_results, output_dir))

    print("Saved client comparison figures:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

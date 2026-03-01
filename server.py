import argparse

import flwr as fl

try:
    import wandb  # optional
except Exception:  # pragma: no cover
    wandb = None


def _weighted_metrics(metrics):
    """Weighted average over a list of (num_examples, metrics_dict)."""
    if not metrics:
        return {}
    total_examples = 0
    agg = {}
    for num_examples, m in metrics:
        total_examples += num_examples
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + num_examples * float(v)
    if total_examples > 0:
        for k in agg:
            agg[k] /= total_examples
    return agg


class WandbFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy with simple wandb logging of aggregated metrics."""

    def __init__(self, use_wandb: bool = False, *args, **kwargs):
        self.use_wandb = use_wandb and wandb is not None
        # Ensure training and validation metrics are aggregated so we can log them
        kwargs.setdefault("fit_metrics_aggregation_fn", _weighted_metrics)
        kwargs.setdefault("evaluate_metrics_aggregation_fn", _weighted_metrics)
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, rnd, results, failures):  # type: ignore[override]
        aggregated_params, metrics = super().aggregate_fit(rnd, results, failures)
        if self.use_wandb and metrics is not None:
            log_data = {f"server/train_{k}": float(v) for k, v in metrics.items()}
            log_data["round"] = rnd
            wandb.log(log_data, step=rnd)
        return aggregated_params, metrics

    def aggregate_evaluate(self, rnd, results, failures):  # type: ignore[override]
        aggregated_loss, metrics = super().aggregate_evaluate(rnd, results, failures)
        if self.use_wandb:
            log_data = {"server/val_loss": float(aggregated_loss)}
            if metrics is not None:
                log_data.update({f"server/val_{k}": float(v) for k, v in metrics.items()})
            log_data["round"] = rnd
            wandb.log(log_data, step=rnd)
        return aggregated_loss, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower server for federated time-series forecasting")
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="gRPC server address, e.g. '0.0.0.0:8080'",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds",
    )
    parser.add_argument(
        "--min_fit_clients",
        type=int,
        default=1,
        help="Minimum number of clients used for training in each round",
    )
    parser.add_argument(
        "--min_available_clients",
        type=int,
        default=1,
        help="Minimum number of clients that need to be connected to start training",
    )

    # wandb options
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable wandb logging on the server")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="federated-time-series-forecasting",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="slife2026-university-of-hong-kong",
        help="Weights & Biases entity (team/user)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Initialize wandb run for the server (optional)
    wb_run = None
    if wandb is not None and getattr(args, "wandb", False):
        wb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name="flwr-server",
            mode="online",
        )
        wandb.config.update(
            {
                "rounds": args.rounds,
                "min_fit_clients": args.min_fit_clients,
                "min_available_clients": args.min_available_clients,
            },
            allow_val_change=True,
        )

    # FedAvg strategy with optional wandb logging.
    strategy = WandbFedAvg(
        use_wandb=getattr(args, "wandb", False),
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    if wb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

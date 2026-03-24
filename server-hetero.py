import argparse
from numbers import Number
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import FitRes, NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

try:
    import wandb  # optional
except Exception:  # pragma: no cover
    wandb = None


def _weighted_numeric_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, float]:
    if not metrics:
        return {}

    total_examples = 0
    aggregated: Dict[str, float] = {}
    for num_examples, metric_dict in metrics:
        total_examples += num_examples
        for key, value in metric_dict.items():
            if isinstance(value, Number):
                aggregated[key] = aggregated.get(key, 0.0) + num_examples * float(value)

    if total_examples > 0:
        for key in aggregated:
            aggregated[key] /= total_examples
    return aggregated


class WandbHeteroFedAvg(fl.server.strategy.FedAvg):
    """Masked aggregation for padded HeteroFL-style client updates."""

    def __init__(self, use_wandb: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_wandb = use_wandb and wandb is not None
        self.latest_parameters: Optional[NDArrays] = None

    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        parameters = super().initialize_parameters(client_manager)
        if parameters is not None:
            ndarrays = parameters_to_ndarrays(parameters)
            self.latest_parameters = self._strip_masks_if_present(ndarrays)
            return ndarrays_to_parameters(self.latest_parameters)
        return None

    @staticmethod
    def _strip_masks_if_present(arrays: NDArrays) -> NDArrays:
        if len(arrays) % 2 != 0:
            return arrays

        half = len(arrays) // 2
        masks = arrays[half:]
        if all(mask.dtype.kind in {"f", "i", "u", "b"} for mask in masks):
            return arrays[:half]
        return arrays

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        split_index = None
        previous_parameters = self.latest_parameters
        weighted_metrics: List[Tuple[int, Dict[str, Scalar]]] = []

        numerators: Optional[List[np.ndarray]] = None
        denominators: Optional[List[np.ndarray]] = None

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if split_index is None:
                if len(arrays) % 2 != 0:
                    raise ValueError(
                        "Heterogeneous client updates must contain padded parameters followed by masks."
                    )
                split_index = len(arrays) // 2
                numerators = [np.zeros_like(arr) for arr in arrays[:split_index]]
                denominators = [np.zeros_like(arr, dtype=np.float32) for arr in arrays[:split_index]]

            local_parameters = arrays[:split_index]
            local_masks = arrays[split_index:]
            num_examples = fit_res.num_examples

            for idx, (param, mask) in enumerate(zip(local_parameters, local_masks)):
                numerators[idx] += num_examples * param * mask
                denominators[idx] += num_examples * mask.astype(np.float32, copy=False)

            weighted_metrics.append((num_examples, fit_res.metrics))

        assert split_index is not None
        assert numerators is not None
        assert denominators is not None

        aggregated: NDArrays = []
        for idx in range(split_index):
            fallback = None
            if previous_parameters is not None and idx < len(previous_parameters):
                fallback = previous_parameters[idx]
            else:
                fallback = np.zeros_like(numerators[idx])

            denom = denominators[idx]
            aggregated_array = np.where(denom > 0, numerators[idx] / denom, fallback)
            aggregated.append(aggregated_array.astype(fallback.dtype, copy=False))

        self.latest_parameters = aggregated
        metrics = _weighted_numeric_metrics(weighted_metrics)

        if self.use_wandb and metrics:
            log_data = {f"server/train_{key}": float(value) for key, value in metrics.items()}
            log_data["round"] = rnd
            wandb.log(log_data, step=rnd)

        return ndarrays_to_parameters(aggregated), metrics

    def aggregate_evaluate(self, rnd, results, failures):  # type: ignore[override]
        aggregated_loss, metrics = super().aggregate_evaluate(rnd, results, failures)
        if self.use_wandb and aggregated_loss is not None:
            log_data = {"server/val_loss": float(aggregated_loss)}
            if metrics is not None:
                numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, Number)}
                log_data.update({f"server/val_{k}": v for k, v in numeric_metrics.items()})
            log_data["round"] = rnd
            wandb.log(log_data, step=rnd)
        return aggregated_loss, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower server for heterogeneous federated time-series forecasting")
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="gRPC server address, e.g. '0.0.0.0:8080'",
    )
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated learning rounds")
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

    wb_run = None
    if wandb is not None and getattr(args, "wandb", False):
        wb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name="flwr-hetero-server",
            mode="online",
        )
        wandb.config.update(
            {
                "rounds": args.rounds,
                "min_fit_clients": args.min_fit_clients,
                "min_available_clients": args.min_available_clients,
                "aggregation": "heterofl_masked_fedavg",
            },
            allow_val_change=True,
        )

    strategy = WandbHeteroFedAvg(
        use_wandb=getattr(args, "wandb", False),
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        fit_metrics_aggregation_fn=_weighted_numeric_metrics,
        evaluate_metrics_aggregation_fn=_weighted_numeric_metrics,
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

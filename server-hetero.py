import argparse
import csv
import json
import time
from numbers import Number
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import EvaluateIns, FitIns, FitRes, NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

try:
    import wandb  # optional
except Exception:  # pragma: no cover
    wandb = None

from src.spa_hfl import (
    AlignmentProjector,
    aggregate_ndarrays,
    cluster_pattern_descriptors,
    update_centroid,
    update_cluster_centroids,
)
from ml.models.gru import GRU
from ml.models.lstm import LSTM
from ml.models.rnn import RNN

SERVER_METRIC_FIELDNAMES = [
    "round",
    "split",
    "loss",
    "forecast_accuracy",
    "global_num_layers",
    "local_num_layers",
    "model_rate",
    "hetero_fedavg",
    "inclusive_fl",
    "spa_hfl",
    "spc",
    "pattern_weighted_residual_heads",
    "fedprox",
    "fedprox_mu",
    "inclusive_group_count",
    "inclusive_beta",
    "mse",
    "rmse",
    "mae",
    "r2",
    "nrmse",
    "round_train_time_seconds",
    "round_evaluate_time_seconds",
    "round_time_seconds",
    "align_train_loss",
    "align_train_rmse",
    "latent_mean_norm",
]

SPC_ASSIGNMENT_FIELDNAMES = ["round", "client_id", "cluster_id"]
RESIDUAL_HEAD_WEIGHT_FIELDNAMES = ["round", "client_id", "weights", "dominant_head_id", "entropy"]


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


def build_recurrent_model(
    model_name: str,
    input_dim: int,
    out_dim: int,
    num_layers: int,
) -> torch.nn.Module:
    if model_name == "rnn":
        return RNN(
            input_dim=input_dim,
            rnn_hidden_size=128,
            num_rnn_layers=num_layers,
            rnn_dropout=0.0,
            layer_units=[128],
            num_outputs=out_dim,
            matrix_rep=True,
            exogenous_dim=0,
        )
    if model_name == "lstm":
        return LSTM(
            input_dim=input_dim,
            lstm_hidden_size=128,
            num_lstm_layers=num_layers,
            lstm_dropout=0.0,
            layer_units=[128],
            num_outputs=out_dim,
            matrix_rep=True,
            exogenous_dim=0,
        )
    if model_name == "gru":
        return GRU(
            input_dim=input_dim,
            gru_hidden_size=128,
            num_gru_layers=num_layers,
            gru_dropout=0.0,
            layer_units=[128],
            num_outputs=out_dim,
            matrix_rep=True,
            exogenous_dim=0,
        )
    raise NotImplementedError("Heterogeneous FL is currently implemented for ['rnn', 'lstm', 'gru'].")


def build_initial_parameters(
    model_name: str,
    input_dim: int,
    out_dim: int,
    global_num_layers: int,
) -> Parameters:
    model = build_recurrent_model(
        model_name=model_name,
        input_dim=input_dim,
        out_dim=out_dim,
        num_layers=global_num_layers,
    )
    arrays = [value.detach().cpu().numpy() for _, value in model.state_dict().items()]
    return ndarrays_to_parameters(arrays)


def get_model_parameter_arrays_and_keys(
    model_name: str,
    input_dim: int,
    out_dim: int,
    global_num_layers: int,
) -> Tuple[NDArrays, List[str]]:
    model = build_recurrent_model(
        model_name=model_name,
        input_dim=input_dim,
        out_dim=out_dim,
        num_layers=global_num_layers,
    )
    state = model.state_dict()
    return [value.detach().cpu().numpy() for _, value in state.items()], list(state.keys())


def infer_spc_head_keys(reference_keys: List[str]) -> List[str]:
    return [key for key in reference_keys if key.startswith("MLP_layers.")]


def _normalize_np_vector(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm <= 1e-8:
        return x.astype(np.float32, copy=False)
    return (x / norm).astype(np.float32, copy=False)


def compute_soft_head_weights(
    pattern_descriptor: Optional[np.ndarray],
    prototypes: Optional[List[np.ndarray]],
    temperature: float,
    num_heads: int,
) -> np.ndarray:
    if num_heads <= 0:
        raise ValueError("num_heads must be positive.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if pattern_descriptor is None or prototypes is None or len(prototypes) < num_heads:
        return np.full(num_heads, 1.0 / num_heads, dtype=np.float32)

    pattern = _normalize_np_vector(np.asarray(pattern_descriptor, dtype=np.float32))
    centers = np.stack([_normalize_np_vector(np.asarray(proto, dtype=np.float32)) for proto in prototypes[:num_heads]], axis=0)
    similarities = centers @ pattern
    logits = similarities / temperature
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    denom = np.clip(exp_logits.sum(), a_min=1e-8, a_max=None)
    return (exp_logits / denom).astype(np.float32, copy=False)


def mix_residual_heads(residual_head_outputs: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if len(residual_head_outputs) != len(weights):
        raise ValueError(f"Expected {len(weights)} residual outputs, got {len(residual_head_outputs)}")
    mixed = np.zeros_like(residual_head_outputs[0])
    for weight, output in zip(weights, residual_head_outputs):
        mixed += float(weight) * output
    return mixed


def aggregate_weighted_residual_heads(
    residual_results: List[Tuple[List[List[np.ndarray]], np.ndarray, int]],
    previous_residual_heads: Optional[List[List[np.ndarray]]],
    num_heads: int,
    head_param_count: int,
    server_weight_power: float = 0.0,
) -> List[List[np.ndarray]]:
    if previous_residual_heads is None:
        if not residual_results:
            return []
        previous_residual_heads = [
            [np.zeros_like(param) for param in residual_results[0][0][head_idx]]
            for head_idx in range(num_heads)
        ]

    updated: List[List[np.ndarray]] = []
    for head_idx in range(num_heads):
        numerators = [np.zeros_like(previous_residual_heads[head_idx][param_idx]) for param_idx in range(head_param_count)]
        denominators = [0.0 for _ in range(head_param_count)]
        for residual_heads, weights, num_examples in residual_results:
            head_weight = float(weights[head_idx])
            if head_weight <= 0.0:
                continue
            weighted_examples = num_examples * (head_weight ** server_weight_power)
            for param_idx in range(head_param_count):
                numerators[param_idx] += weighted_examples * residual_heads[head_idx][param_idx]
                denominators[param_idx] += weighted_examples

        updated_head = []
        for param_idx in range(head_param_count):
            fallback = previous_residual_heads[head_idx][param_idx]
            if denominators[param_idx] > 0.0:
                value = numerators[param_idx] / denominators[param_idx]
            else:
                value = fallback
            updated_head.append(value.astype(fallback.dtype, copy=False))
        updated.append(updated_head)
    return updated


def update_residual_head_prototypes(
    descriptor_results: List[Tuple[np.ndarray, np.ndarray, int]],
    previous_prototypes: Optional[List[np.ndarray]],
    num_heads: int,
) -> List[np.ndarray]:
    if not descriptor_results:
        return previous_prototypes if previous_prototypes is not None else []

    descriptors = np.stack(
        [_normalize_np_vector(np.asarray(descriptor, dtype=np.float32)) for descriptor, _, _ in descriptor_results],
        axis=0,
    )
    if previous_prototypes is None or len(previous_prototypes) < num_heads:
        selected = [0]
        while len(selected) < min(num_heads, descriptors.shape[0]):
            similarity = descriptors @ descriptors[selected].T
            nearest_similarity = similarity.max(axis=1)
            next_idx = int(np.argmin(nearest_similarity))
            if next_idx in selected:
                break
            selected.append(next_idx)
        prototypes = [descriptors[idx].astype(np.float32, copy=False) for idx in selected]
        while len(prototypes) < num_heads:
            prototypes.append(prototypes[-1].copy())
        return prototypes

    descriptor_dim = descriptor_results[0][0].shape[0]
    prototypes: List[np.ndarray] = []
    for head_idx in range(num_heads):
        numerator = np.zeros(descriptor_dim, dtype=np.float32)
        denominator = 0.0
        for descriptor, weights, num_examples in descriptor_results:
            weight = num_examples * float(weights[head_idx])
            numerator += weight * np.asarray(descriptor, dtype=np.float32)
            denominator += weight
        if denominator > 0.0:
            prototypes.append(_normalize_np_vector(numerator / denominator))
        elif previous_prototypes is not None and head_idx < len(previous_prototypes):
            prototypes.append(previous_prototypes[head_idx].astype(np.float32, copy=False))
        else:
            prototypes.append(np.zeros(descriptor_dim, dtype=np.float32))
    return prototypes


class WandbHeteroFedAvg(fl.server.strategy.FedAvg):
    """Masked aggregation for padded HeteroFL-style client updates."""

    def __init__(self, use_wandb: bool = False, *args, **kwargs):
        self.spa_hfl = kwargs.pop("spa_hfl", False)
        self.hetero_fedavg = kwargs.pop("hetero_fedavg", False)
        self.inclusive_fl = kwargs.pop("inclusive_fl", False)
        self.inclusive_beta = kwargs.pop("inclusive_beta", 0.5)
        self.spc = kwargs.pop("spc", False)
        self.pattern_weighted_residual_heads = kwargs.pop("pattern_weighted_residual_heads", False)
        self.align_dim = kwargs.pop("align_dim", 32)
        self.centroid_momentum = kwargs.pop("centroid_momentum", 0.9)
        self.pattern_cluster_count = kwargs.pop("pattern_cluster_count", 1)
        self.pattern_cluster_iters = kwargs.pop("pattern_cluster_iters", 10)
        self.spc_cluster_count = kwargs.pop("spc_cluster_count", 2)
        self.spc_cluster_iters = kwargs.pop("spc_cluster_iters", 10)
        self.spc_assignment_log_path = kwargs.pop("spc_assignment_log_path", "")
        self.num_residual_heads = kwargs.pop("num_residual_heads", 6)
        self.residual_head_temperature = kwargs.pop("residual_head_temperature", 0.1)
        self.residual_head_entropy_lambda = kwargs.pop("residual_head_entropy_lambda", 0.0)
        self.residual_head_init_scale = kwargs.pop("residual_head_init_scale", 0.01)
        self.residual_head_weight_log_path = kwargs.pop("residual_head_weight_log_path", "")
        self.residual_head_server_weight_power = kwargs.pop("residual_head_server_weight_power", 0.0)
        self.reference_keys = kwargs.pop("reference_keys", None)
        self.metrics_log_path = kwargs.pop("metrics_log_path", "")
        self.round_start_times: Dict[int, float] = {}
        self.round_evaluate_start_times: Dict[int, float] = {}
        super().__init__(*args, **kwargs)
        self.use_wandb = use_wandb and wandb is not None
        self.latest_parameters: Optional[NDArrays] = None
        self.latest_spc_backbone: Optional[NDArrays] = None
        self.latest_spc_default_head: Optional[NDArrays] = None
        self.latest_spc_cluster_heads: Dict[int, NDArrays] = {}
        self.latest_pwrh_backbone: Optional[NDArrays] = None
        self.latest_pwrh_global_head: Optional[NDArrays] = None
        self.latest_pwrh_residual_heads: Optional[List[NDArrays]] = None
        self.latest_pwrh_prototypes: Optional[List[np.ndarray]] = None
        self.latest_client_pattern_descriptors: Dict[str, np.ndarray] = {}
        self.latest_client_residual_weights: Dict[str, np.ndarray] = {}
        self.proxy_to_real_cid: Dict[str, str] = {}
        self.latest_projector: Optional[NDArrays] = None
        self.latest_centroid: Optional[np.ndarray] = None
        self.latest_cluster_centroids: Dict[int, np.ndarray] = {}
        self.latest_pattern_cluster_centers: List[np.ndarray] = []
        self.client_cluster_assignments: Dict[str, int] = {}
        self.projector_template = AlignmentProjector(input_dim=128, align_dim=self.align_dim)
        self.projector_param_count = len(list(self.projector_template.state_dict().keys()))
        self.head_keys = infer_spc_head_keys(self.reference_keys or [])
        self.backbone_keys = [key for key in (self.reference_keys or []) if key not in self.head_keys]
        if (self.spc or self.pattern_weighted_residual_heads) and not self.head_keys:
            raise ValueError("Could not identify prediction-head parameters from server model state_dict.")
        self.backbone_param_count = len(self.backbone_keys)
        self.head_param_count = len(self.head_keys)
        if self.pattern_weighted_residual_heads and self.num_residual_heads <= 0:
            raise ValueError("num_residual_heads must be positive.")
        if self.pattern_weighted_residual_heads and self.residual_head_temperature <= 0.0:
            raise ValueError("residual_head_temperature must be positive.")
        if not (0.0 <= self.inclusive_beta <= 1.0):
            raise ValueError("inclusive_beta must be in [0, 1].")

    def configure_fit(self, server_round, parameters, client_manager):  # type: ignore[override]
        """Start the wall-clock timer immediately before a round is sent to clients."""
        self.round_start_times[int(server_round)] = time.perf_counter()
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):  # type: ignore[override]
        """Start the evaluation-phase timer before requests are sent to clients."""
        self.round_evaluate_start_times[int(server_round)] = time.perf_counter()
        return super().configure_evaluate(server_round, parameters, client_manager)

    def _aggregate_masked_layer_updates(
        self,
        split_index: int,
        results: List[Tuple[ClientProxy, FitRes]],
        previous_parameters: Optional[NDArrays],
    ) -> Tuple[NDArrays, List[Tuple[int, Dict[str, Scalar]]]]:
        if previous_parameters is None:
            raise ValueError("Heterogeneous FedAvg requires initialized server parameters.")

        numerators = [np.zeros_like(arr) for arr in previous_parameters[:split_index]]
        denominators = [np.zeros_like(arr, dtype=np.float32) for arr in previous_parameters[:split_index]]
        weighted_metrics: List[Tuple[int, Dict[str, Scalar]]] = []

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            local_parameters = arrays[:split_index]
            local_masks = arrays[split_index : 2 * split_index]
            num_examples = fit_res.num_examples
            for idx, (param, mask) in enumerate(zip(local_parameters, local_masks)):
                update = (param - previous_parameters[idx]) * mask
                numerators[idx] += num_examples * update
                denominators[idx] += num_examples * mask.astype(np.float32, copy=False)
            weighted_metrics.append((num_examples, fit_res.metrics))

        aggregated: NDArrays = []
        for idx in range(split_index):
            previous = previous_parameters[idx]
            denom = denominators[idx]
            averaged_update = np.where(denom > 0, numerators[idx] / denom, 0.0)
            aggregated.append((previous + averaged_update).astype(previous.dtype, copy=False))
        return aggregated, weighted_metrics

    def _aggregate_inclusive_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        previous_parameters = self.latest_parameters
        if previous_parameters is None:
            raise ValueError("InclusiveFL requires initialized server parameters.")

        split_index = len(previous_parameters)
        groups: Dict[Tuple[int, float], Dict[str, object]] = {}
        weighted_metrics: List[Tuple[int, Dict[str, Scalar]]] = []

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            local_parameters = arrays[:split_index]
            local_masks = arrays[split_index : 2 * split_index]
            metrics = fit_res.metrics
            local_layers = int(metrics.get("local_num_layers", 1))
            model_rate = round(float(metrics.get("model_rate", 1.0)), 6)
            group_key = (local_layers, model_rate)
            group = groups.setdefault(
                group_key,
                {
                    "examples": 0,
                    "numerators": [np.zeros_like(arr) for arr in previous_parameters],
                    "denominators": [np.zeros_like(arr, dtype=np.float32) for arr in previous_parameters],
                },
            )
            group["examples"] = int(group["examples"]) + fit_res.num_examples
            numerators = group["numerators"]
            denominators = group["denominators"]
            for idx, (param, mask) in enumerate(zip(local_parameters, local_masks)):
                numerators[idx] += fit_res.num_examples * param * mask
                denominators[idx] += fit_res.num_examples * mask.astype(np.float32, copy=False)
            weighted_metrics.append((fit_res.num_examples, metrics))

        group_states: Dict[Tuple[int, float], Dict[str, object]] = {}
        for group_key, group in groups.items():
            params: NDArrays = []
            masks: NDArrays = []
            for idx, previous in enumerate(previous_parameters):
                denom = group["denominators"][idx]
                mask = (denom > 0).astype(np.float32, copy=False)
                aggregate = previous.copy()
                np.divide(group["numerators"][idx], denom, out=aggregate, where=denom > 0)
                params.append(aggregate.astype(previous.dtype, copy=False))
                masks.append(mask)
            group_states[group_key] = {
                "examples": int(group["examples"]),
                "params": params,
                "masks": masks,
            }

        sorted_keys = sorted(group_states, key=lambda key: (key[0], key[1]))
        for pos, group_key in enumerate(sorted_keys[:-1]):
            larger_key = sorted_keys[pos + 1]
            current = group_states[group_key]
            larger = group_states[larger_key]
            for idx in range(split_index):
                overlap = (current["masks"][idx] > 0) & (larger["masks"][idx] > 0)
                if np.any(overlap):
                    blended = current["params"][idx].copy()
                    blended[overlap] = (
                        self.inclusive_beta * larger["params"][idx][overlap]
                        + (1.0 - self.inclusive_beta) * current["params"][idx][overlap]
                    )
                    current["params"][idx] = blended.astype(previous_parameters[idx].dtype, copy=False)

        final_numerators = [np.zeros_like(arr) for arr in previous_parameters]
        final_denominators = [np.zeros_like(arr, dtype=np.float32) for arr in previous_parameters]
        for group in group_states.values():
            examples = int(group["examples"])
            for idx in range(split_index):
                final_numerators[idx] += examples * group["params"][idx] * group["masks"][idx]
                final_denominators[idx] += examples * group["masks"][idx].astype(np.float32, copy=False)

        aggregated: NDArrays = []
        for idx, previous in enumerate(previous_parameters):
            denom = final_denominators[idx]
            aggregate = previous.copy()
            np.divide(final_numerators[idx], denom, out=aggregate, where=denom > 0)
            aggregated.append(aggregate.astype(previous.dtype, copy=False))

        self.latest_parameters = aggregated
        metrics = _weighted_numeric_metrics(weighted_metrics)
        metrics["inclusive_group_count"] = float(len(group_states))
        metrics["inclusive_beta"] = float(self.inclusive_beta)
        if self.use_wandb and metrics:
            log_data = {f"server/train_{key}": float(value) for key, value in metrics.items()}
            log_data["round"] = rnd
            wandb.log(log_data, step=rnd, commit=False)
        self._append_metric_row(rnd, "train", metrics)
        return ndarrays_to_parameters(list(aggregated)), metrics

    def _append_metric_row(self, rnd: int, split: str, metrics: Dict[str, Scalar]) -> None:
        if not self.metrics_log_path:
            return
        path = Path(self.metrics_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {key: "" for key in SERVER_METRIC_FIELDNAMES}
        row["round"] = rnd
        row["split"] = split
        for key, value in metrics.items():
            if key in row and isinstance(value, Number):
                row[key] = float(value)
        needs_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SERVER_METRIC_FIELDNAMES)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)

    def _append_spc_assignment_rows(self, rnd: int, assignments: Dict[str, int]) -> None:
        if not self.spc_assignment_log_path:
            return
        path = Path(self.spc_assignment_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SPC_ASSIGNMENT_FIELDNAMES)
            if needs_header:
                writer.writeheader()
            for cid, cluster_idx in sorted(assignments.items()):
                writer.writerow({"round": rnd, "client_id": cid, "cluster_id": int(cluster_idx)})

    def _append_residual_head_weight_rows(self, rnd: int, weights_by_client: Dict[str, np.ndarray]) -> None:
        if not self.residual_head_weight_log_path:
            return
        path = Path(self.residual_head_weight_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESIDUAL_HEAD_WEIGHT_FIELDNAMES)
            if needs_header:
                writer.writeheader()
            for cid, weights in sorted(weights_by_client.items()):
                weights = weights.astype(np.float32, copy=False)
                entropy = -float(np.sum(weights * np.log(np.clip(weights, a_min=1e-8, a_max=None))))
                writer.writerow(
                    {
                        "round": rnd,
                        "client_id": cid,
                        "weights": json.dumps([float(v) for v in weights]),
                        "dominant_head_id": int(np.argmax(weights)),
                        "entropy": entropy,
                    }
                )

    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        parameters = super().initialize_parameters(client_manager)
        if parameters is not None:
            ndarrays = parameters_to_ndarrays(parameters)
            if self.pattern_weighted_residual_heads:
                if not self.reference_keys:
                    raise ValueError("Pattern-weighted residual heads require reference_keys.")
                if len(ndarrays) != len(self.reference_keys):
                    raise ValueError(
                        f"Expected {len(self.reference_keys)} initial model tensors for PWRH, got {len(ndarrays)}"
                    )
                key_to_array = dict(zip(self.reference_keys, ndarrays))
                self.latest_pwrh_backbone = [key_to_array[key] for key in self.backbone_keys]
                self.latest_pwrh_global_head = [key_to_array[key] for key in self.head_keys]
                self.latest_pwrh_residual_heads = [
                    [
                        np.random.normal(
                            loc=0.0,
                            scale=self.residual_head_init_scale,
                            size=key_to_array[key].shape,
                        ).astype(key_to_array[key].dtype, copy=False)
                        for key in self.head_keys
                    ]
                    for _ in range(self.num_residual_heads)
                ]
                uniform_weights = np.full(self.num_residual_heads, 1.0 / self.num_residual_heads, dtype=np.float32)
                return ndarrays_to_parameters(
                    list(self.latest_pwrh_backbone)
                    + list(self.latest_pwrh_global_head)
                    + [param for head in self.latest_pwrh_residual_heads for param in head]
                    + [uniform_weights]
                )
            if self.spc:
                if not self.reference_keys:
                    raise ValueError("SPC requires reference_keys to split backbone/head parameters.")
                if len(ndarrays) != len(self.reference_keys):
                    raise ValueError(
                        f"Expected {len(self.reference_keys)} initial model tensors for SPC, got {len(ndarrays)}"
                    )
                key_to_array = dict(zip(self.reference_keys, ndarrays))
                self.latest_spc_backbone = [key_to_array[key] for key in self.backbone_keys]
                self.latest_spc_default_head = [key_to_array[key] for key in self.head_keys]
                self.latest_spc_cluster_heads = {
                    cluster_idx: list(self.latest_spc_default_head)
                    for cluster_idx in range(max(1, self.spc_cluster_count))
                }
                return ndarrays_to_parameters(list(self.latest_spc_backbone) + list(self.latest_spc_default_head))
            self.latest_parameters = self._strip_masks_if_present(ndarrays)
            payload = list(self.latest_parameters)
            if self.spa_hfl:
                self.latest_projector = [
                    value.detach().cpu().numpy() for _, value in self.projector_template.state_dict().items()
                ]
                self.latest_centroid = np.zeros(self.align_dim, dtype=np.float32)
                payload = payload + self.latest_projector + [self.latest_centroid]
            return ndarrays_to_parameters(payload)
        return None

    def _build_spc_payload_for_client(self, client_proxy: ClientProxy) -> Parameters:
        if self.latest_spc_backbone is None or self.latest_spc_default_head is None:
            return ndarrays_to_parameters([])
        cluster_idx = self.client_cluster_assignments.get(client_proxy.cid)
        head = None if cluster_idx is None else self.latest_spc_cluster_heads.get(cluster_idx)
        if head is None:
            head = self.latest_spc_default_head
        payload = list(self.latest_spc_backbone) + list(head)
        return ndarrays_to_parameters(payload)

    def _build_pwrh_payload_for_client(self, client_proxy: ClientProxy) -> Parameters:
        if (
            self.latest_pwrh_backbone is None
            or self.latest_pwrh_global_head is None
            or self.latest_pwrh_residual_heads is None
        ):
            return ndarrays_to_parameters([])
        real_cid = self.proxy_to_real_cid.get(client_proxy.cid, client_proxy.cid)
        descriptor = self.latest_client_pattern_descriptors.get(real_cid)
        weights = compute_soft_head_weights(
            pattern_descriptor=descriptor,
            prototypes=self.latest_pwrh_prototypes,
            temperature=self.residual_head_temperature,
            num_heads=self.num_residual_heads,
        )
        self.latest_client_residual_weights[real_cid] = weights
        payload = (
            list(self.latest_pwrh_backbone)
            + list(self.latest_pwrh_global_head)
            + [param for residual_head in self.latest_pwrh_residual_heads for param in residual_head]
            + [weights]
        )
        return ndarrays_to_parameters(payload)

    def _build_spa_payload_for_client(self, client_proxy: ClientProxy) -> Parameters:
        payload = list(self.latest_parameters or [])
        projector = self.latest_projector or [
            value.detach().cpu().numpy() for _, value in self.projector_template.state_dict().items()
        ]
        payload.extend(projector)

        centroid = self.latest_centroid
        if self.client_cluster_assignments and self.latest_cluster_centroids:
            cluster_idx = self.client_cluster_assignments.get(client_proxy.cid)
            if cluster_idx is not None:
                centroid = self.latest_cluster_centroids.get(cluster_idx, centroid)
        if centroid is None:
            centroid = np.zeros(self.align_dim, dtype=np.float32)
        payload.append(centroid.astype(np.float32, copy=False))
        return ndarrays_to_parameters(payload)

    @staticmethod
    def _strip_masks_if_present(arrays: NDArrays) -> NDArrays:
        if len(arrays) % 2 != 0:
            return arrays

        half = len(arrays) // 2
        parameters = arrays[:half]
        masks = arrays[half:]
        has_mask_shapes = all(param.shape == mask.shape for param, mask in zip(parameters, masks))
        has_binary_masks = all(
            mask.dtype.kind in {"f", "i", "u", "b"} and np.all((mask == 0) | (mask == 1))
            for mask in masks
        )
        if has_mask_shapes and has_binary_masks:
            return arrays[:half]
        return arrays

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        if self.pattern_weighted_residual_heads and self.latest_pwrh_backbone is not None:
            config = {} if self.on_fit_config_fn is None else self.on_fit_config_fn(server_round)
            fit_ins_by_client: List[Tuple[ClientProxy, FitIns]] = []

            sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
            clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

            for client in clients:
                fit_ins = FitIns(self._build_pwrh_payload_for_client(client), config)
                fit_ins_by_client.append((client, fit_ins))
            return fit_ins_by_client

        if self.spc and self.latest_spc_backbone is not None:
            config = {} if self.on_fit_config_fn is None else self.on_fit_config_fn(server_round)
            fit_ins_by_client: List[Tuple[ClientProxy, FitIns]] = []

            sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
            clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

            for client in clients:
                fit_ins = FitIns(self._build_spc_payload_for_client(client), config)
                fit_ins_by_client.append((client, fit_ins))
            return fit_ins_by_client

        if not self.spa_hfl or self.pattern_cluster_count <= 1 or self.latest_parameters is None:
            return super().configure_fit(server_round, parameters, client_manager)

        config = {} if self.on_fit_config_fn is None else self.on_fit_config_fn(server_round)
        fit_ins_by_client: List[Tuple[ClientProxy, FitIns]] = []

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        for client in clients:
            fit_ins = FitIns(self._build_spa_payload_for_client(client), config)
            fit_ins_by_client.append((client, fit_ins))
        return fit_ins_by_client

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        if self.pattern_weighted_residual_heads and self.latest_pwrh_backbone is not None:
            config = {} if self.on_evaluate_config_fn is None else self.on_evaluate_config_fn(server_round)
            evaluate_ins_by_client: List[Tuple[ClientProxy, EvaluateIns]] = []

            sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
            clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

            for client in clients:
                evaluate_ins = EvaluateIns(self._build_pwrh_payload_for_client(client), config)
                evaluate_ins_by_client.append((client, evaluate_ins))
            return evaluate_ins_by_client

        if self.spc and self.latest_spc_backbone is not None:
            config = {} if self.on_evaluate_config_fn is None else self.on_evaluate_config_fn(server_round)
            evaluate_ins_by_client: List[Tuple[ClientProxy, EvaluateIns]] = []

            sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
            clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

            for client in clients:
                evaluate_ins = EvaluateIns(self._build_spc_payload_for_client(client), config)
                evaluate_ins_by_client.append((client, evaluate_ins))
            return evaluate_ins_by_client
        return super().configure_evaluate(server_round, parameters, client_manager)

    def _aggregate_spc_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        backbone_count = self.backbone_param_count
        head_count = self.head_param_count
        expected_total = 2 * backbone_count + head_count + 1
        if backbone_count <= 0 or head_count <= 0:
            raise ValueError("SPC requires non-empty backbone and head parameter splits.")

        previous_backbone = self.latest_spc_backbone
        if previous_backbone is None:
            raise ValueError("SPC server backbone is not initialized.")

        numerators: Optional[List[np.ndarray]] = None
        denominators: Optional[List[np.ndarray]] = None
        head_results: List[Tuple[str, List[np.ndarray], np.ndarray, int]] = []
        weighted_metrics: List[Tuple[int, Dict[str, Scalar]]] = []

        for client_proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) != expected_total:
                raise ValueError(
                    f"Expected SPC client payload of {expected_total} arrays "
                    f"({backbone_count} backbone + {backbone_count} masks + {head_count} head + pattern), "
                    f"got {len(arrays)} from client {client_proxy.cid}"
                )

            local_backbone = arrays[:backbone_count]
            local_masks = arrays[backbone_count : 2 * backbone_count]
            local_head = arrays[2 * backbone_count : 2 * backbone_count + head_count]
            pattern_mean = arrays[-1].astype(np.float32, copy=False)

            if numerators is None:
                numerators = [np.zeros_like(arr) for arr in local_backbone]
                denominators = [np.zeros_like(arr, dtype=np.float32) for arr in local_backbone]

            num_examples = fit_res.num_examples
            for idx, (param, mask) in enumerate(zip(local_backbone, local_masks)):
                numerators[idx] += num_examples * param * mask
                denominators[idx] += num_examples * mask.astype(np.float32, copy=False)
            
            real_cid = str(fit_res.metrics.get("cid", client_proxy.cid))
            head_results.append((real_cid, local_head, pattern_mean, num_examples))
            weighted_metrics.append((num_examples, fit_res.metrics))

        assert numerators is not None
        assert denominators is not None

        aggregated_backbone: NDArrays = []
        for idx in range(backbone_count):
            fallback = previous_backbone[idx] if idx < len(previous_backbone) else np.zeros_like(numerators[idx])
            denom = denominators[idx]
            aggregated_array = np.where(denom > 0, numerators[idx] / denom, fallback)
            aggregated_backbone.append(aggregated_array.astype(fallback.dtype, copy=False))
        self.latest_spc_backbone = aggregated_backbone

        client_ids = [cid for cid, _, _, _ in head_results]
        pattern_means = [pattern_mean for _, _, pattern_mean, _ in head_results]
        assignment_map, pattern_centers = cluster_pattern_descriptors(
            client_ids=client_ids,
            pattern_descriptors=pattern_means,
            num_clusters=self.spc_cluster_count,
            num_iters=self.spc_cluster_iters,
            previous_centers=self.latest_pattern_cluster_centers,
        )
        self.client_cluster_assignments.update(assignment_map)
        self.latest_pattern_cluster_centers = pattern_centers
        print(f"[SPC][Round {rnd}] assignments: {self.client_cluster_assignments}")
        self._append_spc_assignment_rows(rnd,  self.client_cluster_assignments)

        all_head_entries = [(head, num_examples) for _, head, _, num_examples in head_results]
        self.latest_spc_default_head = aggregate_ndarrays(all_head_entries, previous=self.latest_spc_default_head)

        updated_cluster_heads: Dict[int, NDArrays] = {}
        for cluster_idx in range(max(1, self.spc_cluster_count)):
            cluster_entries = [
                (head, num_examples)
                for cid, head, _, num_examples in head_results
                if assignment_map.get(cid, 0) == cluster_idx
            ]
            previous_head = self.latest_spc_cluster_heads.get(cluster_idx, self.latest_spc_default_head)
            updated_cluster_heads[cluster_idx] = aggregate_ndarrays(cluster_entries, previous=previous_head)
        self.latest_spc_cluster_heads = updated_cluster_heads

        metrics = _weighted_numeric_metrics(weighted_metrics)
        if self.use_wandb and metrics:
            log_data = {f"server/train_{key}": float(value) for key, value in metrics.items()}
            log_data["round"] = rnd
            wandb.log(log_data, step=rnd, commit=False)
        self._append_metric_row(rnd, "train", metrics)

        return ndarrays_to_parameters(list(self.latest_spc_backbone) + list(self.latest_spc_default_head)), metrics

    def _aggregate_pwrh_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        backbone_count = self.backbone_param_count
        head_count = self.head_param_count
        num_heads = self.num_residual_heads
        expected_total = 2 * backbone_count + head_count + num_heads * head_count + 1
        if backbone_count <= 0 or head_count <= 0:
            raise ValueError("PWRH requires non-empty backbone and head parameter splits.")
        if self.latest_pwrh_backbone is None or self.latest_pwrh_global_head is None:
            raise ValueError("PWRH server state is not initialized.")

        numerators: Optional[List[np.ndarray]] = None
        denominators: Optional[List[np.ndarray]] = None
        global_head_results: List[Tuple[List[np.ndarray], int]] = []
        residual_results: List[Tuple[List[List[np.ndarray]], np.ndarray, int]] = []
        descriptor_results: List[Tuple[np.ndarray, np.ndarray, int]] = []
        weights_by_client: Dict[str, np.ndarray] = {}
        weighted_metrics: List[Tuple[int, Dict[str, Scalar]]] = []

        for client_proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) != expected_total:
                raise ValueError(
                    f"Expected PWRH client payload of {expected_total} arrays "
                    f"({backbone_count} backbone + {backbone_count} masks + {head_count} global head + "
                    f"{num_heads}*{head_count} residual head + pattern), got {len(arrays)} from client {client_proxy.cid}"
                )

            cursor = 0
            local_backbone = arrays[cursor : cursor + backbone_count]
            cursor += backbone_count
            local_masks = arrays[cursor : cursor + backbone_count]
            cursor += backbone_count
            local_global_head = arrays[cursor : cursor + head_count]
            cursor += head_count
            local_residual_heads = []
            for _ in range(num_heads):
                local_residual_heads.append(arrays[cursor : cursor + head_count])
                cursor += head_count
            pattern_mean = arrays[cursor].astype(np.float32, copy=False)

            if numerators is None:
                numerators = [np.zeros_like(arr) for arr in local_backbone]
                denominators = [np.zeros_like(arr, dtype=np.float32) for arr in local_backbone]

            num_examples = fit_res.num_examples
            for idx, (param, mask) in enumerate(zip(local_backbone, local_masks)):
                numerators[idx] += num_examples * param * mask
                denominators[idx] += num_examples * mask.astype(np.float32, copy=False)

            real_cid = str(fit_res.metrics.get("cid", client_proxy.cid))
            self.proxy_to_real_cid[client_proxy.cid] = real_cid
            self.latest_client_pattern_descriptors[real_cid] = pattern_mean
            weights = compute_soft_head_weights(
                pattern_descriptor=pattern_mean,
                prototypes=self.latest_pwrh_prototypes,
                temperature=self.residual_head_temperature,
                num_heads=num_heads,
            )
            weights_by_client[real_cid] = weights
            self.latest_client_residual_weights[real_cid] = weights

            global_head_results.append((local_global_head, num_examples))
            residual_results.append((local_residual_heads, weights, num_examples))
            descriptor_results.append((pattern_mean, weights, num_examples))
            weighted_metrics.append((num_examples, fit_res.metrics))

        assert numerators is not None
        assert denominators is not None

        aggregated_backbone: NDArrays = []
        for idx in range(backbone_count):
            fallback = self.latest_pwrh_backbone[idx]
            denom = denominators[idx]
            aggregated_array = np.where(denom > 0, numerators[idx] / denom, fallback)
            aggregated_backbone.append(aggregated_array.astype(fallback.dtype, copy=False))
        self.latest_pwrh_backbone = aggregated_backbone
        self.latest_pwrh_global_head = aggregate_ndarrays(
            global_head_results,
            previous=self.latest_pwrh_global_head,
        )
        self.latest_pwrh_residual_heads = aggregate_weighted_residual_heads(
            residual_results=residual_results,
            previous_residual_heads=self.latest_pwrh_residual_heads,
            num_heads=num_heads,
            head_param_count=head_count,
            server_weight_power=self.residual_head_server_weight_power,
        )
        self.latest_pwrh_prototypes = update_residual_head_prototypes(
            descriptor_results=descriptor_results,
            previous_prototypes=self.latest_pwrh_prototypes,
            num_heads=num_heads,
        )
        print(
            f"[PWRH][Round {rnd}] weights: "
            f"{ {cid: weights.tolist() for cid, weights in sorted(weights_by_client.items())} }"
        )
        self._append_residual_head_weight_rows(rnd, weights_by_client)

        metrics = _weighted_numeric_metrics(weighted_metrics)
        if self.use_wandb and metrics:
            log_data = {f"server/train_{key}": float(value) for key, value in metrics.items()}
            log_data["round"] = rnd
            wandb.log(log_data, step=rnd, commit=False)
        self._append_metric_row(rnd, "train", metrics)

        uniform_weights = np.full(num_heads, 1.0 / num_heads, dtype=np.float32)
        payload = (
            list(self.latest_pwrh_backbone)
            + list(self.latest_pwrh_global_head)
            + [param for residual_head in self.latest_pwrh_residual_heads for param in residual_head]
            + [uniform_weights]
        )
        return ndarrays_to_parameters(payload), metrics

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        if self.pattern_weighted_residual_heads:
            return self._aggregate_pwrh_fit(rnd, results)

        if self.spc:
            return self._aggregate_spc_fit(rnd, results)

        if self.inclusive_fl:
            return self._aggregate_inclusive_fit(rnd, results)

        split_index = None
        previous_parameters = self.latest_parameters
        weighted_metrics: List[Tuple[int, Dict[str, Scalar]]] = []

        numerators: Optional[List[np.ndarray]] = None
        denominators: Optional[List[np.ndarray]] = None
        projector_results: List[Tuple[List[np.ndarray], int]] = []
        centroid_results: List[Tuple[Dict[str, np.ndarray], int]] = []
        pattern_results: List[Tuple[str, np.ndarray, np.ndarray, int]] = []

        for client_proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if split_index is None:
                if self.latest_parameters is not None:
                    split_index = len(self.latest_parameters)
                elif self.spa_hfl:
                    extra_stats = 2 if len(arrays) >= self.projector_param_count + 2 else 1
                    split_index = (len(arrays) - self.projector_param_count - extra_stats) // 2
                else:
                    split_index = len(arrays) // 2
                numerators = [np.zeros_like(arr) for arr in arrays[:split_index]]
                denominators = [np.zeros_like(arr, dtype=np.float32) for arr in arrays[:split_index]]

            if self.hetero_fedavg:
                assert split_index is not None
                aggregated, weighted_metrics = self._aggregate_masked_layer_updates(
                    split_index=split_index,
                    results=results,
                    previous_parameters=previous_parameters,
                )
                self.latest_parameters = aggregated
                metrics = _weighted_numeric_metrics(weighted_metrics)
                if self.use_wandb and metrics:
                    log_data = {f"server/train_{key}": float(value) for key, value in metrics.items()}
                    log_data["round"] = rnd
                    wandb.log(log_data, step=rnd, commit=False)
                self._append_metric_row(rnd, "train", metrics)
                return ndarrays_to_parameters(list(aggregated)), metrics

            local_parameters = arrays[:split_index]
            local_masks = arrays[split_index : 2 * split_index]
            num_examples = fit_res.num_examples

            for idx, (param, mask) in enumerate(zip(local_parameters, local_masks)):
                numerators[idx] += num_examples * param * mask
                denominators[idx] += num_examples * mask.astype(np.float32, copy=False)

            if self.spa_hfl:
                projector_start = 2 * split_index
                projector_end = projector_start + self.projector_param_count
                projector_results.append((arrays[projector_start:projector_end], num_examples))
                centroid_results.append(({"latent_mean": arrays[projector_end]}, num_examples))
                if len(arrays) > projector_end + 1:
                    pattern_results.append(
                        (
                            client_proxy.cid,
                            arrays[projector_end].astype(np.float32, copy=False),
                            arrays[projector_end + 1].astype(np.float32, copy=False),
                            num_examples,
                        )
                    )

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
        payload = list(aggregated)
        if self.spa_hfl:
            self.latest_projector = aggregate_ndarrays(projector_results, previous=self.latest_projector)
            if self.pattern_cluster_count > 1 and pattern_results:
                client_ids = [cid for cid, _, _, _ in pattern_results]
                latent_means = [latent_mean for _, latent_mean, _, _ in pattern_results]
                pattern_means = [pattern_mean for _, _, pattern_mean, _ in pattern_results]
                assignment_map, pattern_centers = cluster_pattern_descriptors(
                    client_ids=client_ids,
                    pattern_descriptors=pattern_means,
                    num_clusters=self.pattern_cluster_count,
                    num_iters=self.pattern_cluster_iters,
                    previous_centers=self.latest_pattern_cluster_centers,
                )
                self.client_cluster_assignments = assignment_map
                self.latest_pattern_cluster_centers = pattern_centers

                clustered_stats: Dict[int, List[Tuple[np.ndarray, int]]] = {}
                for cid, latent_mean, _, num_examples in pattern_results:
                    cluster_idx = assignment_map.get(cid, 0)
                    clustered_stats.setdefault(cluster_idx, []).append((latent_mean, num_examples))
                self.latest_cluster_centroids = update_cluster_centroids(
                    clustered_stats,
                    previous_centroids=self.latest_cluster_centroids,
                    momentum=self.centroid_momentum,
                )

                global_cluster_results = []
                for entries in clustered_stats.values():
                    total_examples = sum(num_examples for _, num_examples in entries)
                    weighted_latent = sum(num_examples * latent_mean for latent_mean, num_examples in entries)
                    global_cluster_results.append(
                        ({"latent_mean": (weighted_latent / max(total_examples, 1)).astype(np.float32, copy=False)}, total_examples)
                    )
                self.latest_centroid = update_centroid(
                    global_cluster_results,
                    previous_centroid=self.latest_centroid,
                    momentum=self.centroid_momentum,
                )
            else:
                self.client_cluster_assignments = {}
                self.latest_pattern_cluster_centers = []
                self.latest_cluster_centroids = {}
                self.latest_centroid = update_centroid(
                    centroid_results,
                    previous_centroid=self.latest_centroid,
                    momentum=self.centroid_momentum,
                )
            payload = payload + self.latest_projector + [self.latest_centroid]
        metrics = _weighted_numeric_metrics(weighted_metrics)

        if self.use_wandb and metrics:
            log_data = {f"server/train_{key}": float(value) for key, value in metrics.items()}
            log_data["round"] = rnd
            wandb.log(log_data, step=rnd, commit=False)
        self._append_metric_row(rnd, "train", metrics)

        return ndarrays_to_parameters(payload), metrics

    def aggregate_evaluate(self, rnd, results, failures):  # type: ignore[override]
        evaluate_start_time = self.round_evaluate_start_times.pop(int(rnd), time.perf_counter())
        aggregated_loss, metrics = super().aggregate_evaluate(rnd, results, failures)
        evaluate_time = time.perf_counter() - evaluate_start_time
        round_start_time = self.round_start_times.pop(int(rnd), None)
        round_time = time.perf_counter() - round_start_time if round_start_time is not None else evaluate_time
        if metrics is None:
            metrics = {}
        metrics["round_evaluate_time_seconds"] = float(evaluate_time)
        metrics["round_time_seconds"] = float(round_time)
        if self.use_wandb and aggregated_loss is not None:
            log_data = {"server/val_loss": float(aggregated_loss)}
            if metrics is not None:
                numeric_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, Number)}
                log_data.update({f"server/val_{k}": v for k, v in numeric_metrics.items()})
            log_data["round"] = rnd
            wandb.log(log_data, step=rnd, commit=True)
        if aggregated_loss is not None:
            eval_metrics = {"loss": float(aggregated_loss)}
            if metrics is not None:
                for k, v in metrics.items():
                    if isinstance(v, Number):
                        eval_metrics[k] = float(v)
            self._append_metric_row(rnd, "val", eval_metrics)
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
        default=2,
        help="Minimum number of clients used for training in each round",
    )
    parser.add_argument(
        "--min_available_clients",
        type=int,
        default=2,
        help="Minimum number of clients that need to be connected to start training",
    )
    parser.add_argument(
        "--min_evaluate_clients",
        type=int,
        default=None,
        help="Minimum number of clients used for evaluation in each round. Defaults to min_fit_clients.",
    )
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable wandb logging on the server")
    parser.add_argument("--model_name", type=str, default="lstm", choices=["rnn", "lstm", "gru"])
    parser.add_argument("--input_dim", type=int, default=9, help="Number of input features used by each client model")
    parser.add_argument("--out_dim", type=int, default=4, help="Number of forecast outputs")
    parser.add_argument("--global_num_layers", type=int, default=3, help="Number of layers in the global supernet")
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
    parser.add_argument("--spa_hfl", action="store_true", default=False, help="Enable SPA-HFL aggregation state")
    parser.add_argument(
        "--hetero_fedavg",
        action="store_true",
        default=False,
        help="Use heterogeneous FedAvg aggregation over sample-weighted layer updates/deltas.",
    )
    parser.add_argument("--inclusive_fl", action="store_true", default=False, help="Enable InclusiveFL group transfer aggregation.")
    parser.add_argument("--inclusive_beta", type=float, default=0.5, help="InclusiveFL blend weight for larger-model shared knowledge.")
    parser.add_argument("--spc", action="store_true", default=False, help="Enable SPC-HeteroFL clustered heads")
    parser.add_argument(
        "--pattern_weighted_residual_heads",
        action="store_true",
        default=False,
        help="Enable pattern-weighted residual heads.",
    )
    parser.add_argument("--align_dim", type=int, default=32)
    parser.add_argument("--centroid_momentum", type=float, default=0.9)
    parser.add_argument(
        "--pattern_cluster_count",
        type=int,
        default=1,
        help="Number of pattern clusters for cluster-aware SPA aggregation. Use 1 to keep the original global centroid.",
    )
    parser.add_argument(
        "--pattern_cluster_iters",
        type=int,
        default=10,
        help="Maximum clustering iterations for pattern-aware SPA aggregation.",
    )
    parser.add_argument("--spc_cluster_count", type=int, default=2, help="Number of SPC sequence-pattern clusters.")
    parser.add_argument("--spc_cluster_iters", type=int, default=10, help="Maximum SPC clustering iterations.")
    parser.add_argument("--num_residual_heads", type=int, default=6)
    parser.add_argument("--residual_head_temperature", type=float, default=0.1)
    parser.add_argument("--residual_head_entropy_lambda", type=float, default=0.0)
    parser.add_argument("--residual_head_init_scale", type=float, default=0.01)
    parser.add_argument(
        "--residual_head_weight_log_path",
        type=str,
        default="",
        help="Optional CSV path for pattern-weighted residual head weights.",
    )
    parser.add_argument("--residual_head_server_weight_power", type=float, default=0.0)
    parser.add_argument(
        "--spc_assignment_log_path",
        type=str,
        default="",
        help="Optional CSV path for SPC round/client cluster assignments.",
    )
    parser.add_argument(
        "--metrics_log_path",
        type=str,
        default="./benchmark_logs/server_hetero_metrics.csv",
        help="CSV path for per-round aggregated metrics logging.",
    )
    args = parser.parse_args()
    mode_count = sum(
        [
            bool(args.spa_hfl),
            bool(args.hetero_fedavg),
            bool(args.inclusive_fl),
            bool(args.spc),
            bool(args.pattern_weighted_residual_heads),
        ]
    )
    if mode_count > 1:
        raise ValueError("--inclusive_fl, --hetero_fedavg, --spa_hfl, --spc, and --pattern_weighted_residual_heads are mutually exclusive modes.")
    if not (0.0 <= args.inclusive_beta <= 1.0):
        raise ValueError("--inclusive_beta must be in [0, 1].")
    if args.spc_cluster_count <= 0:
        raise ValueError("--spc_cluster_count must be positive.")
    if args.num_residual_heads <= 0:
        raise ValueError("--num_residual_heads must be positive.")
    if args.residual_head_temperature <= 0.0:
        raise ValueError("--residual_head_temperature must be positive.")
    if args.residual_head_init_scale < 0.0:
        raise ValueError("--residual_head_init_scale must be non-negative.")
    return args


def main() -> None:
    args = parse_args()
    min_evaluate_clients = args.min_fit_clients if args.min_evaluate_clients is None else args.min_evaluate_clients
    initial_parameters = build_initial_parameters(
        model_name=args.model_name,
        input_dim=args.input_dim,
        out_dim=args.out_dim,
        global_num_layers=args.global_num_layers,
    )
    _, reference_keys = get_model_parameter_arrays_and_keys(
        model_name=args.model_name,
        input_dim=args.input_dim,
        out_dim=args.out_dim,
        global_num_layers=args.global_num_layers,
    )

    wb_run = None
    if wandb is not None and getattr(args, "wandb", False):
        wb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=(
                f"flwr-hetero-server"
                f"{'-pwrh' if args.pattern_weighted_residual_heads else '-spc' if args.spc else '-spa' if args.spa_hfl else '-inclusive-fl' if args.inclusive_fl else '-hetero-fedavg' if args.hetero_fedavg else 'hfl'}"
            ),
            mode="online",
        )
        wandb.config.update(
            {
                "rounds": args.rounds,
                "min_fit_clients": args.min_fit_clients,
                "min_evaluate_clients": min_evaluate_clients,
                "min_available_clients": args.min_available_clients,
                "aggregation": (
                    "pattern_weighted_residual_heads"
                    if args.pattern_weighted_residual_heads
                    else "spc_heterofl_clustered_heads"
                    if args.spc
                    else "spa_hfl_masked_fedavg"
                    if args.spa_hfl
                    else "inclusive_fl_group_transfer"
                    if args.inclusive_fl
                    else "heterogeneous_fedavg_layer_updates"
                    if args.hetero_fedavg
                    else "heterofl_masked_fedavg"
                ),
                "hetero_fedavg": args.hetero_fedavg,
                "inclusive_fl": args.inclusive_fl,
                "inclusive_beta": args.inclusive_beta,
                "spa_hfl": args.spa_hfl,
                "spc": args.spc,
                "pattern_weighted_residual_heads": args.pattern_weighted_residual_heads,
                "pattern_cluster_count": args.pattern_cluster_count,
                "spc_cluster_count": args.spc_cluster_count,
                "num_residual_heads": args.num_residual_heads,
                "strict_synchronous": True,
            },
            allow_val_change=True,
        )

    strategy = WandbHeteroFedAvg(
        use_wandb=getattr(args, "wandb", False),
        spa_hfl=args.spa_hfl,
        hetero_fedavg=args.hetero_fedavg,
        inclusive_fl=args.inclusive_fl,
        inclusive_beta=args.inclusive_beta,
        spc=args.spc,
        pattern_weighted_residual_heads=args.pattern_weighted_residual_heads,
        align_dim=args.align_dim,
        centroid_momentum=args.centroid_momentum,
        pattern_cluster_count=args.pattern_cluster_count,
        pattern_cluster_iters=args.pattern_cluster_iters,
        spc_cluster_count=args.spc_cluster_count,
        spc_cluster_iters=args.spc_cluster_iters,
        spc_assignment_log_path=args.spc_assignment_log_path,
        num_residual_heads=args.num_residual_heads,
        residual_head_temperature=args.residual_head_temperature,
        residual_head_entropy_lambda=args.residual_head_entropy_lambda,
        residual_head_init_scale=args.residual_head_init_scale,
        residual_head_weight_log_path=args.residual_head_weight_log_path,
        residual_head_server_weight_power=args.residual_head_server_weight_power,
        reference_keys=reference_keys,
        metrics_log_path=args.metrics_log_path,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        initial_parameters=initial_parameters,
        min_evaluate_clients=min_evaluate_clients,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        accept_failures=False,
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

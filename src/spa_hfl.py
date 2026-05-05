import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ml.utils.helpers import EarlyStopping, accumulate_metric, get_criterion


class AlignmentProjector(torch.nn.Module):
    def __init__(self, input_dim: int, align_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, align_dim),
            torch.nn.LayerNorm(align_dim),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _squeeze_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and x.size(-1) == 1:
        return x[..., 0]
    return x


def compute_autocorr_summary(x: torch.Tensor, max_lag: int) -> torch.Tensor:
    seq = _squeeze_sequence(x).mean(dim=-1)
    seq = seq - seq.mean(dim=1, keepdim=True)
    seq = seq / (seq.std(dim=1, keepdim=True) + 1e-6)
    summaries = []
    t = seq.size(1)
    for lag in range(1, max_lag + 1):
        if lag >= t:
            summaries.append(torch.zeros(seq.size(0), device=seq.device))
        else:
            summaries.append((seq[:, :-lag] * seq[:, lag:]).mean(dim=1))
    return torch.stack(summaries, dim=1)


def compute_fft_summary(x: torch.Tensor, num_bins: int) -> torch.Tensor:
    seq = _squeeze_sequence(x).mean(dim=-1)
    seq = seq - seq.mean(dim=1, keepdim=True)
    device = seq.device
    # Small batched FFTs can hit cuFFT internal errors on some CUDA/PyTorch builds.
    # Computing this descriptor on CPU keeps SPA pattern summaries stable.
    spec = torch.fft.rfft(seq.detach().cpu(), dim=1).abs().to(device)
    spec = spec[:, 1 : 1 + num_bins]
    if spec.size(1) < num_bins:
        pad = torch.zeros(spec.size(0), num_bins - spec.size(1), device=device)
        spec = torch.cat([spec, pad], dim=1)
    return spec


def compute_pattern_summary(x: torch.Tensor, acf_lags: int, fft_bins: int) -> torch.Tensor:
    acf = compute_autocorr_summary(x, acf_lags)
    fft = compute_fft_summary(x, fft_bins)
    summary = torch.cat([acf, fft], dim=1)
    return torch.nn.functional.normalize(summary, dim=1)


def pairwise_cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, dim=1)
    return x @ x.transpose(0, 1)


def alignment_loss(z: torch.Tensor, centroid: Optional[torch.Tensor]) -> torch.Tensor:
    if centroid is None:
        return z.new_tensor(0.0)
    centroid = torch.nn.functional.normalize(centroid.unsqueeze(0), dim=1)
    z_mean = torch.nn.functional.normalize(z.mean(dim=0, keepdim=True), dim=1)
    return 1.0 - torch.sum(z_mean * centroid)


def consistency_loss(z: torch.Tensor, pattern_summary: torch.Tensor) -> torch.Tensor:
    if z.size(0) <= 1:
        return z.new_tensor(0.0)
    latent_sim = pairwise_cosine_matrix(z)
    raw_sim = pairwise_cosine_matrix(pattern_summary)
    return torch.nn.functional.mse_loss(latent_sim, raw_sim)


def pack_state_dict(module: torch.nn.Module) -> List[np.ndarray]:
    return [param.detach().cpu().numpy() for _, param in module.state_dict().items()]


def load_state_dict_from_ndarrays(module: torch.nn.Module, arrays: List[np.ndarray]) -> None:
    params_dict = zip(module.state_dict().keys(), arrays)
    state_dict = {k: torch.tensor(v, dtype=module.state_dict()[k].dtype) for k, v in params_dict}
    module.load_state_dict(state_dict, strict=True)


def aggregate_ndarrays(results: List[Tuple[List[np.ndarray], int]], previous: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
    if not results:
        return previous if previous is not None else []
    total_examples = sum(num_examples for _, num_examples in results)
    aggregated = []
    for idx in range(len(results[0][0])):
        acc = sum(num_examples * arrays[idx] for arrays, num_examples in results)
        aggregated.append((acc / total_examples).astype(results[0][0][idx].dtype, copy=False))
    return aggregated


def update_centroid(
    stats_results: List[Tuple[Dict[str, np.ndarray], int]],
    previous_centroid: Optional[np.ndarray],
    momentum: float,
) -> Optional[np.ndarray]:
    if not stats_results:
        return previous_centroid
    total_examples = sum(num_examples for _, num_examples in stats_results)
    weighted = sum(num_examples * stats["latent_mean"] for stats, num_examples in stats_results) / max(total_examples, 1)
    if previous_centroid is None:
        return weighted.astype(np.float32, copy=False)
    return (momentum * previous_centroid + (1.0 - momentum) * weighted).astype(np.float32, copy=False)


def _normalize_np_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, a_min=1e-8, a_max=None)


def _normalize_np_vector(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm <= 1e-8:
        return x.astype(np.float32, copy=False)
    return (x / norm).astype(np.float32, copy=False)


def cluster_pattern_descriptors(
    client_ids: List[str],
    pattern_descriptors: List[np.ndarray],
    num_clusters: int,
    num_iters: int = 10,
    previous_centers: Optional[List[np.ndarray]] = None,
) -> Tuple[Dict[str, int], List[np.ndarray]]:
    if not client_ids or not pattern_descriptors:
        return {}, []

    patterns = np.stack([_normalize_np_vector(np.asarray(p, dtype=np.float32)) for p in pattern_descriptors], axis=0)
    num_clients = patterns.shape[0]
    k = max(1, min(num_clusters, num_clients))

    if previous_centers is not None and len(previous_centers) >= k:
        centers = np.stack(
            [_normalize_np_vector(np.asarray(center, dtype=np.float32)) for center in previous_centers[:k]],
            axis=0,
        )
    else:
        selected = [0]
        while len(selected) < k:
            similarity = patterns @ patterns[selected].T
            min_similarity = similarity.max(axis=1)
            next_idx = int(np.argmin(min_similarity))
            if next_idx in selected:
                break
            selected.append(next_idx)
        while len(selected) < k:
            selected.append(len(selected))
        centers = patterns[selected].copy()

    assignments = np.zeros(num_clients, dtype=np.int64)
    for _ in range(max(1, num_iters)):
        similarity = patterns @ centers.T
        new_assignments = similarity.argmax(axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments

        new_centers = []
        for cluster_idx in range(k):
            members = patterns[assignments == cluster_idx]
            if len(members) == 0:
                new_centers.append(centers[cluster_idx])
            else:
                new_centers.append(_normalize_np_vector(members.mean(axis=0)))
        centers = np.stack(new_centers, axis=0)

    assignment_map = {cid: int(cluster_idx) for cid, cluster_idx in zip(client_ids, assignments)}
    center_list = [centers[idx].astype(np.float32, copy=False) for idx in range(k)]
    return assignment_map, center_list


def update_cluster_centroids(
    clustered_stats: Dict[int, List[Tuple[np.ndarray, int]]],
    previous_centroids: Optional[Dict[int, np.ndarray]],
    momentum: float,
) -> Dict[int, np.ndarray]:
    updated: Dict[int, np.ndarray] = {}
    for cluster_idx, entries in clustered_stats.items():
        if not entries:
            continue
        total_examples = sum(num_examples for _, num_examples in entries)
        weighted = sum(num_examples * latent_mean for latent_mean, num_examples in entries) / max(total_examples, 1)
        weighted = weighted.astype(np.float32, copy=False)
        previous = None if previous_centroids is None else previous_centroids.get(cluster_idx)
        if previous is None:
            updated[cluster_idx] = weighted
        else:
            updated[cluster_idx] = (
                momentum * previous + (1.0 - momentum) * weighted
            ).astype(np.float32, copy=False)
    return updated


def evaluate_spa(model, projector, data_loader, criterion, device="cuda"):
    model.to(device)
    projector.to(device)
    model.eval()
    projector.eval()
    y_true, y_pred = [], []
    loss = 0.0
    with torch.no_grad():
        for x, exogenous, y_hist, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_hist = y_hist.to(device)
            if exogenous is not None and len(exogenous) > 0:
                exogenous = exogenous.to(device)
            else:
                exogenous = None
            out = model(x, exogenous, device, y_hist, return_features=True)
            pred = out["prediction"]
            loss += criterion(pred, y).item()
            y_true.extend(y)
            y_pred.extend(pred)

    loss /= len(data_loader.dataset)
    y_true = torch.stack(y_true)
    y_pred = torch.stack(y_pred)
    mse, rmse, mae, r2, nrmse = accumulate_metric(y_true.cpu(), y_pred.cpu())
    return loss, mse, rmse, mae, r2, nrmse


def train_spa_hfl(
    model: torch.nn.Module,
    projector: AlignmentProjector,
    train_loader,
    val_loader,
    epochs: int,
    optimizer_name: str,
    lr: float,
    criterion_name: str,
    device: str,
    lambda_align: float,
    lambda_cons: float,
    global_centroid: Optional[np.ndarray],
    acf_lags: int,
    fft_bins: int,
    early_stopping: bool = False,
    patience: int = 50,
    max_grad_norm: float = 0.0,
) -> Tuple[torch.nn.Module, AlignmentProjector, Dict[str, np.ndarray], Dict[str, float]]:
    params = list(model.parameters()) + list(projector.parameters())
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr)
    else:
        raise NotImplementedError(f"Unsupported optimizer: {optimizer_name}")

    criterion = get_criterion(criterion_name)
    centroid_tensor = None
    if global_centroid is not None:
        centroid_tensor = torch.tensor(global_centroid, dtype=torch.float32, device=device)

    best_bundle = None
    best_val_loss = np.inf
    monitor = EarlyStopping(patience, trace=False) if early_stopping else None

    for _ in range(epochs):
        model.to(device)
        projector.to(device)
        model.train()
        projector.train()
        for x, exogenous, y_hist, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_hist = y_hist.to(device)
            if exogenous is not None and len(exogenous) > 0:
                exogenous = exogenous.to(device)
            else:
                exogenous = None

            optimizer.zero_grad()
            out = model(x, exogenous, device, y_hist, return_features=True)
            pred = out["prediction"]
            z = projector(out["last_hidden"])
            pattern = compute_pattern_summary(x, acf_lags=acf_lags, fft_bins=fft_bins)

            forecast_loss = criterion(pred, y)
            align_reg = alignment_loss(z, centroid_tensor)
            cons_reg = consistency_loss(z, pattern)
            loss = forecast_loss + lambda_align * align_reg + lambda_cons * cons_reg
            loss.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()

        val_loss, *_ = evaluate_spa(model, projector, val_loader, criterion, device=device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_bundle = (copy.deepcopy(model), copy.deepcopy(projector))

        if monitor is not None:
            monitor(val_loss, model)
            if monitor.early_stop:
                break

    if best_bundle is not None:
        model, projector = best_bundle

    model.to(device)
    projector.to(device)
    model.eval()
    projector.eval()

    latent_means = []
    pattern_means = []
    with torch.no_grad():
        for x, exogenous, y_hist, _ in train_loader:
            x = x.to(device)
            y_hist = y_hist.to(device)
            if exogenous is not None and len(exogenous) > 0:
                exogenous = exogenous.to(device)
            else:
                exogenous = None
            out = model(x, exogenous, device, y_hist, return_features=True)
            z = projector(out["last_hidden"])
            pattern = compute_pattern_summary(x, acf_lags=acf_lags, fft_bins=fft_bins)
            latent_means.append(z.mean(dim=0))
            pattern_means.append(pattern.mean(dim=0))

    stats = {
        "latent_mean": torch.stack(latent_means).mean(dim=0).detach().cpu().numpy().astype(np.float32, copy=False),
        "pattern_mean": torch.stack(pattern_means).mean(dim=0).detach().cpu().numpy().astype(np.float32, copy=False),
    }

    train_loss, train_mse, train_rmse, train_mae, train_r2, train_nrmse = evaluate_spa(
        model, projector, train_loader, criterion, device=device
    )
    metrics = {
        "train_loss": float(train_loss),
        "train_mse": float(train_mse),
        "train_rmse": float(train_rmse),
        "train_mae": float(train_mae),
        "train_r2": float(train_r2),
        "train_nrmse": float(train_nrmse),
    }
    return model, projector, stats, metrics

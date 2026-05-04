import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch

try:
    import wandb  # optional
except Exception:  # pragma: no cover
    wandb = None

from client import prepare_client_data, save_client_model, save_last_lags_predictions
from ml.models.gru import GRU
from ml.models.lstm import LSTM
from ml.models.rnn import RNN
from ml.utils.helpers import get_criterion
from ml.utils.train_utils import test, train
from src.spa_hfl import (
    AlignmentProjector,
    evaluate_spa,
    load_state_dict_from_ndarrays,
    pack_state_dict,
    train_spa_hfl,
)

CLIENT_METRIC_FIELDNAMES = [
    "round",
    "split",
    "cid",
    "local_num_layers",
    "global_num_layers",
    "spa_hfl",
    "loss",
    "mse",
    "rmse",
    "mae",
    "r2",
    "nrmse",
    "round_train_time_seconds",
    "align_train_loss",
    "align_train_rmse",
    "latent_mean_norm",
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class HeteroModelAdapter:
    """Maps a smaller local recurrent model into a padded global supernet."""

    def __init__(
        self,
        model_name: str,
        input_dim: int,
        out_dim: int,
        local_num_layers: int,
        global_num_layers: int,
    ) -> None:
        if local_num_layers > global_num_layers:
            raise ValueError("local_num_layers must be <= global_num_layers")

        self.model_name = model_name
        self.local_num_layers = local_num_layers
        self.global_num_layers = global_num_layers
        self.local_model = build_recurrent_model(model_name, input_dim, out_dim, local_num_layers)
        self.reference_model = build_recurrent_model(model_name, input_dim, out_dim, global_num_layers)
        self.reference_keys = list(self.reference_model.state_dict().keys())

    def export_parameters_with_masks(self) -> List[np.ndarray]:
        local_state = self.local_model.state_dict()
        reference_state = self.reference_model.state_dict()

        full_parameters: List[np.ndarray] = []
        full_masks: List[np.ndarray] = []
        for key in self.reference_keys:
            ref_tensor = reference_state[key].detach().cpu().numpy()
            if key in local_state:
                value = local_state[key].detach().cpu().numpy()
                full_parameters.append(value.astype(ref_tensor.dtype, copy=False))
                full_masks.append(np.ones_like(ref_tensor, dtype=np.float32))
            else:
                full_parameters.append(np.zeros_like(ref_tensor))
                full_masks.append(np.zeros_like(ref_tensor, dtype=np.float32))

        return full_parameters + full_masks

    def load_global_parameters(self, parameters: List[np.ndarray]) -> None:
        if not parameters:
            return

        num_reference_tensors = len(self.reference_keys)
        if len(parameters) >= 2 * num_reference_tensors:
            parameters = parameters[:num_reference_tensors]
        elif len(parameters) != num_reference_tensors:
            raise ValueError(
                f"Expected {num_reference_tensors} global tensors or {2 * num_reference_tensors} "
                f"padded tensors+masks, got {len(parameters)}"
            )

        current_state = self.local_model.state_dict()
        updated_state = {}
        for key, global_value in zip(self.reference_keys, parameters):
            if key not in current_state:
                continue
            updated_state[key] = torch.tensor(global_value, dtype=current_state[key].dtype)

        current_state.update(updated_state)
        self.local_model.load_state_dict(current_state, strict=True)


class FlowerHeteroTimeSeriesClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        adapter: HeteroModelAdapter,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        args: argparse.Namespace,
    ) -> None:
        self.cid = cid
        self.adapter = adapter
        self.model = adapter.local_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
        self.criterion = get_criterion(args.criterion)
        self.wandb_round = 0
        self.projector = AlignmentProjector(input_dim=self.model.hidden_dim, align_dim=args.align_dim)

    def _append_metric_row(
        self,
        split: str,
        loss: float,
        metrics: Dict[str, float],
        round_train_time_seconds: float = None,
        alignment_metrics: Dict[str, float] = None,
        alignment_stats: Dict[str, np.ndarray] = None,
    ) -> None:
        if not getattr(self.args, "metrics_log_path", ""):
            return

        path = Path(self.args.metrics_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists()

        row: Dict[str, Any] = {key: "" for key in CLIENT_METRIC_FIELDNAMES}
        row.update(
            {
                "round": int(self.wandb_round),
                "split": split,
                "cid": self.cid,
                "local_num_layers": int(self.args.local_num_layers),
                "global_num_layers": int(self.args.global_num_layers),
                "spa_hfl": 1.0 if getattr(self.args, "spa_hfl", False) else 0.0,
                "loss": float(loss),
                "mse": float(metrics["mse"]),
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "r2": float(metrics["r2"]),
                "nrmse": float(metrics["nrmse"]),
            }
        )
        if round_train_time_seconds is not None:
            row["round_train_time_seconds"] = float(round_train_time_seconds)
        if alignment_metrics is not None:
            row["align_train_loss"] = float(alignment_metrics["train_loss"])
            row["align_train_rmse"] = float(alignment_metrics["train_rmse"])
        if alignment_stats is not None:
            row["latent_mean_norm"] = float(np.linalg.norm(alignment_stats["latent_mean"]))

        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CLIENT_METRIC_FIELDNAMES)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)

    def get_parameters(self, config):  # type: ignore[override]
        if getattr(self.args, "spa_hfl", False):
            return self._get_spa_parameters()
        return self.adapter.export_parameters_with_masks()

    def _set_parameters(self, parameters: List[np.ndarray]) -> None:
        if getattr(self.args, "spa_hfl", False):
            self._set_spa_parameters(parameters)
            return
        self.adapter.load_global_parameters(parameters)
        self.model = self.adapter.local_model

    def _get_spa_parameters(self) -> List[np.ndarray]:
        centroid = getattr(self, "current_global_centroid", np.zeros(self.args.align_dim, dtype=np.float32))
        return self.adapter.export_parameters_with_masks() + pack_state_dict(self.projector) + [centroid]

    def _set_spa_parameters(self, parameters: List[np.ndarray]) -> None:
        model_param_count = len(self.adapter.reference_keys)
        projector_param_count = len(list(self.projector.state_dict().keys()))
        min_total = model_param_count + projector_param_count + 1
        raw_total = 2 * model_param_count + projector_param_count + 1
        if len(parameters) < min_total:
            raise ValueError(
                f"Expected at least {min_total} tensors for SPA-HFL state, got {len(parameters)}"
            )
        if len(parameters) >= raw_total:
            model_parameters = parameters[:model_param_count]
            projector_start = 2 * model_param_count
            projector_parameters = parameters[projector_start : projector_start + projector_param_count]
            centroid = parameters[projector_start + projector_param_count]
        else:
            model_parameters = parameters[:model_param_count]
            projector_parameters = parameters[model_param_count : model_param_count + projector_param_count]
            centroid = parameters[model_param_count + projector_param_count]

        self.adapter.load_global_parameters(model_parameters)
        self.model = self.adapter.local_model
        load_state_dict_from_ndarrays(self.projector, projector_parameters)
        self.current_global_centroid = centroid

    def fit(self, parameters, config):  # type: ignore[override]
        if parameters is not None:
            self._set_parameters(parameters)

        self.wandb_round += 1

        start_time = time.time()
        if getattr(self.args, "spa_hfl", False):
            centroid = getattr(self, "current_global_centroid", None)
            self.model, self.projector, alignment_stats, alignment_metrics = train_spa_hfl(
                model=self.model,
                projector=self.projector,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                epochs=self.args.epochs,
                optimizer_name=self.args.optimizer,
                lr=self.args.lr,
                criterion_name=self.args.criterion,
                device=self.device,
                lambda_align=self.args.lambda_align,
                lambda_cons=self.args.lambda_cons,
                global_centroid=centroid,
                acf_lags=self.args.acf_lags,
                fft_bins=self.args.fft_bins,
                early_stopping=self.args.local_early_stopping,
                patience=self.args.local_patience,
                max_grad_norm=self.args.max_grad_norm,
            )
        else:
            alignment_stats = None
            alignment_metrics = None
            self.model = train(
                model=self.model,
                train_loader=self.train_loader,
                test_loader=self.val_loader,
                epochs=self.args.epochs,
                optimizer=self.args.optimizer,
                lr=self.args.lr,
                reg1=self.args.reg1,
                reg2=self.args.reg2,
                max_grad_norm=self.args.max_grad_norm,
                criterion=self.args.criterion,
                early_stopping=self.args.local_early_stopping,
                patience=self.args.local_patience,
                plot_history=False,
                device=self.device,
                fedprox_mu=0.0,
                log_per=1,
                use_carbontracker=self.args.use_carbontracker,
            )
        self.adapter.local_model = self.model
        round_train_time = time.time() - start_time

        if getattr(self.args, "spa_hfl", False):
            loss, mse, rmse, mae, r2, nrmse = evaluate_spa(
                self.model,
                self.projector,
                self.train_loader,
                self.criterion,
                device=self.device,
            )
        else:
            loss, mse, rmse, mae, r2, nrmse = test(
                self.model,
                self.train_loader,
                self.criterion,
                device=self.device,
            )

        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "nrmse": float(nrmse),
            "local_num_layers": int(self.args.local_num_layers),
            "global_num_layers": int(self.args.global_num_layers),
            "spa_hfl": 1.0 if getattr(self.args, "spa_hfl", False) else 0.0,
        }
        if alignment_metrics is not None:
            metrics["align_train_loss"] = float(alignment_metrics["train_loss"])
            metrics["align_train_rmse"] = float(alignment_metrics["train_rmse"])
            metrics["latent_mean_norm"] = float(np.linalg.norm(alignment_stats["latent_mean"]))

        self._append_metric_row(
            split="train",
            loss=float(loss),
            metrics=metrics,
            round_train_time_seconds=float(round_train_time),
            alignment_metrics=alignment_metrics,
            alignment_stats=alignment_stats,
        )

        if wandb is not None and getattr(self.args, "wandb", False):
            log_data = {
                "client/train_loss": float(loss),
                "client/train_mse": float(mse),
                "client/train_rmse": float(rmse),
                "client/train_mae": float(mae),
                "client/train_r2": float(r2),
                "client/train_nrmse": float(nrmse),
                "client/round_train_time_seconds": float(round_train_time),
                "client/local_num_layers": int(self.args.local_num_layers),
                "client/spa_hfl": 1.0 if getattr(self.args, "spa_hfl", False) else 0.0,
                "round": int(self.wandb_round),
            }
            if alignment_metrics is not None:
                log_data["client/align_train_loss"] = float(alignment_metrics["train_loss"])
                log_data["client/align_train_rmse"] = float(alignment_metrics["train_rmse"])
            wandb.log(log_data, step=int(self.wandb_round), commit=False)

        if getattr(self.args, "spa_hfl", False):
            payload = self.adapter.export_parameters_with_masks() + pack_state_dict(self.projector) + [
                alignment_stats["latent_mean"].astype(np.float32, copy=False),
                alignment_stats["pattern_mean"].astype(np.float32, copy=False),
            ]
        else:
            payload = self.get_parameters(config)
        return payload, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):  # type: ignore[override]
        if parameters is not None:
            self._set_parameters(parameters)

        if getattr(self.args, "spa_hfl", False):
            loss, mse, rmse, mae, r2, nrmse = evaluate_spa(
                self.model,
                self.projector,
                self.val_loader,
                self.criterion,
                device=self.device,
            )
        else:
            loss, mse, rmse, mae, r2, nrmse = test(
                self.model,
                self.val_loader,
                self.criterion,
                device=self.device,
            )

        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "nrmse": float(nrmse),
            "local_num_layers": int(self.args.local_num_layers),
            "spa_hfl": 1.0 if getattr(self.args, "spa_hfl", False) else 0.0,
        }

        self._append_metric_row(split="val", loss=float(loss), metrics=metrics)

        if wandb is not None and getattr(self.args, "wandb", False):
            wandb.log(
                {
                    "client/val_loss": float(loss),
                    "client/val_mse": float(mse),
                    "client/val_rmse": float(rmse),
                    "client/val_mae": float(mae),
                    "client/val_r2": float(r2),
                    "client/val_nrmse": float(nrmse),
                    "client/local_num_layers": int(self.args.local_num_layers),
                    "client/spa_hfl": 1.0 if getattr(self.args, "spa_hfl", False) else 0.0,
                    "round": int(self.wandb_round),
                },
                step=int(self.wandb_round),
                commit=True,
            )

        return float(loss), len(self.val_loader.dataset), metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower client for heterogeneous federated time-series forecasting")

    parser.add_argument("--cid", type=str, default="12167-0", help="Client ID (e.g. District name)")
    parser.add_argument(
        "--server_address",
        type=str,
        default="127.0.0.1:8080",
        help="Flower server address, e.g. '127.0.0.1:8080'",
    )

    parser.add_argument("--data_path", type=str, default="./dataset/5G-2y-bs12167.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=[
            "BBU Energy (W)",
        ],
    )
    parser.add_argument("--idxs", nargs="+", type=int, default=[4])
    parser.add_argument("--num_lags", type=int, default=48)
    parser.add_argument("--prediction_steps", type=int, default=4)
    parser.add_argument("--identifier", type=str, default="District")
    parser.add_argument("--nan_constant", type=float, default=0.0)
    parser.add_argument("--x_scaler", type=str, default="minmax")
    parser.add_argument("--y_scaler", type=str, default="minmax")
    parser.add_argument("--outlier_detection", type=str, default=None)

    parser.add_argument("--model_name", type=str, default="lstm")
    parser.add_argument("--local_num_layers", type=int, default=1)
    parser.add_argument("--global_num_layers", type=int, default=3)
    parser.add_argument("--criterion", type=str, default="mse")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--local_early_stopping", action="store_true", default=False)
    parser.add_argument("--local_patience", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--reg1", type=float, default=0.0)
    parser.add_argument("--reg2", type=float, default=0.0)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_carbontracker", action="store_true", default=False)
    parser.add_argument("--spa_hfl", action="store_true", default=False, help="Enable SPA-HFL alignment training")
    parser.add_argument("--align_dim", type=int, default=32)
    parser.add_argument("--lambda_align", type=float, default=0.1)
    parser.add_argument("--lambda_cons", type=float, default=0.1)
    parser.add_argument("--acf_lags", type=int, default=8)
    parser.add_argument("--fft_bins", type=int, default=8)

    parser.add_argument("--wandb", action="store_true", default=False, help="Enable wandb logging for this client")
    parser.add_argument(
        "--metrics_log_path",
        type=str,
        default="",
        help="Optional CSV path for stable per-round client metrics logging.",
    )
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
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./saved_models/{cid}_{model_name}_final.pt",
        help="Output path template for saving final client model after FL ends.",
    )
    parser.add_argument(
        "--prediction_save_path",
        type=str,
        default="./saved_predictions/{cid}_{model_name}_last_{num_lags}.csv",
        help="Output path template for final predictions CSV.",
    )

    args = parser.parse_args()

    if args.outlier_detection is not None:
        args.outlier_columns = ["rb_down", "rb_up", "down", "up"]
        args.outlier_kwargs = {"ElBorn": (10, 90), "LesCorts": (10, 90), "PobleSec": (5, 95)}

    if isinstance(args.idxs, list) and len(args.idxs) > 0 and isinstance(args.idxs[0], str):
        args.idxs = [int(i) for i in args.idxs]

    if args.model_name not in {"rnn", "lstm", "gru"}:
        raise ValueError("client-hetero.py currently supports heterogeneous recurrent models: ['rnn', 'lstm', 'gru']")
    if args.local_num_layers <= 0 or args.global_num_layers <= 0:
        raise ValueError("Both local_num_layers and global_num_layers must be positive integers.")
    if args.local_num_layers > args.global_num_layers:
        raise ValueError("local_num_layers cannot exceed global_num_layers.")

    return args


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    wb_run = None
    if wandb is not None and getattr(args, "wandb", False):
        wb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"flwr-hetclient-{args.cid}-{args.model_name}-L{args.local_num_layers}{'-spa' if args.spa_hfl else 'hfl'}",
            mode="online",
        )
        wandb.config.update(
            {
                "cid": args.cid,
                "model_name": args.model_name,
                "local_num_layers": args.local_num_layers,
                "global_num_layers": args.global_num_layers,
                "spa_hfl": args.spa_hfl,
            },
            allow_val_change=True,
        )

    train_loader, val_loader, num_features, out_dim, prediction_artifacts = prepare_client_data(args)

    adapter = HeteroModelAdapter(
        model_name=args.model_name,
        input_dim=num_features,
        out_dim=out_dim,
        local_num_layers=args.local_num_layers,
        global_num_layers=args.global_num_layers,
    )

    client = FlowerHeteroTimeSeriesClient(
        cid=args.cid,
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
    )

    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    saved_path = save_client_model(client.model, args)
    print(f"Saved final client model to: {saved_path}")

    prediction_path = save_last_lags_predictions(client.model, client.device, args, prediction_artifacts)
    print(f"Saved final predictions CSV to: {prediction_path}")

    if wb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

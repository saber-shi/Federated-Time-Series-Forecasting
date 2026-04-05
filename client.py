import argparse
import random
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

import flwr as fl
import numpy as np
import pandas as pd
import torch

try:
    import wandb  # optional
except Exception:  # pragma: no cover
    wandb = None

from ml.utils.data_utils import (
    read_data,
    handle_nans,
    to_train_val,
    handle_outliers,
    to_Xy,
    scale_features,
    generate_time_lags,
    remove_identifiers,
    to_timeseries_rep,
    to_torch_dataset,
)
from ml.utils.train_utils import train, test
from ml.utils.helpers import get_criterion
from ml.models.mlp import MLP
from ml.models.rnn import RNN
from ml.models.lstm import LSTM
from ml.models.gru import GRU
from ml.models.cnn import CNN
from ml.models.rnn_autoencoder import DualAttentionAutoEncoder


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(
    model_name: str,
    input_dim: int,
    out_dim: int,
    lags: int,
    exogenous_dim: int = 0,
) -> torch.nn.Module:
    if model_name == "mlp":
        return MLP(input_dim=input_dim, layer_units=[256, 128, 64], num_outputs=out_dim)
    if model_name == "rnn":
        return RNN(
            input_dim=input_dim,
            rnn_hidden_size=128,
            num_rnn_layers=1,
            rnn_dropout=0.0,
            layer_units=[128],
            num_outputs=out_dim,
            matrix_rep=True,
            exogenous_dim=exogenous_dim,
        )
    if model_name == "lstm":
        return LSTM(
            input_dim=input_dim,
            lstm_hidden_size=128,
            num_lstm_layers=1,
            lstm_dropout=0.0,
            layer_units=[128],
            num_outputs=out_dim,
            matrix_rep=True,
            exogenous_dim=exogenous_dim,
        )
    if model_name == "gru":
        return GRU(
            input_dim=input_dim,
            gru_hidden_size=128,
            num_gru_layers=1,
            gru_dropout=0.0,
            layer_units=[128],
            num_outputs=out_dim,
            matrix_rep=True,
            exogenous_dim=exogenous_dim,
        )
    if model_name == "cnn":
        return CNN(num_features=input_dim, lags=lags, exogenous_dim=exogenous_dim, out_dim=out_dim)
    if model_name == "da_encoder_decoder":
        return DualAttentionAutoEncoder(input_dim=input_dim, architecture="lstm", matrix_rep=True)
    raise NotImplementedError(
        "Specified model is not implemented. Choose one from ['mlp', 'rnn', 'lstm', 'gru', 'cnn', 'da_encoder_decoder']."
    )


def prepare_client_data(
    args: argparse.Namespace,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, int, Dict[str, Any]]:
    # Load only the data for this client (District)
    df = read_data(args.data_path, filter_data=args.cid)

    # Handle NaNs
    df = handle_nans(train_data=df, constant=args.nan_constant, identifier=args.identifier)

    # Train/validation split (per time order)
    train_data, val_data = to_train_val(df, train_size=1.0 - args.test_size, identifier=args.identifier)

    # Optional outlier handling
    if args.outlier_detection is not None:
        train_data = handle_outliers(
            df=train_data,
            columns=args.outlier_columns,
            identifier=args.identifier,
            kwargs=args.outlier_kwargs,
        )

    # Features/targets
    X_train, X_val, y_train, y_val = to_Xy(
        train_data=train_data,
        val_data=val_data,
        targets=args.targets,
        identifier=args.identifier,
    )

    # Scale features and targets (single area, but reuse per_area logic)
    X_train, X_val, _ = scale_features(
        train_data=X_train,
        val_data=X_val,
        scaler=args.x_scaler,
        per_area=True,
        identifier=args.identifier,
    )
    y_train, y_val, y_scalers = scale_features(
        train_data=y_train,
        val_data=y_val,
        scaler=args.y_scaler,
        per_area=True,
        identifier=args.identifier,
    )

    # Time lags
    X_train = generate_time_lags(
        X_train,
        args.num_lags,
        identifier=args.identifier,
        prediction_steps=args.prediction_steps,
    )
    X_val = generate_time_lags(
        X_val,
        args.num_lags,
        identifier=args.identifier,
        prediction_steps=args.prediction_steps,
    )
    y_train = generate_time_lags(
        y_train,
        args.num_lags,
        identifier=args.identifier,
        is_y=True,
        prediction_steps=args.prediction_steps,
    )
    y_val = generate_time_lags(
        y_val,
        args.num_lags,
        identifier=args.identifier,
        is_y=True,
        prediction_steps=args.prediction_steps,
    )

    # Remove identifier column and convert to time-series representation
    X_train, y_train, X_val, y_val = remove_identifiers(
        X_train,
        y_train,
        X_val,
        y_val,
        identifier=args.identifier,
    )

    num_features = len(X_train.columns) // args.num_lags
    X_train_ts = to_timeseries_rep(X_train.to_numpy(), num_lags=args.num_lags, num_features=num_features)
    X_val_ts = to_timeseries_rep(X_val.to_numpy(), num_lags=args.num_lags, num_features=num_features)

    y_train_np, y_val_np = y_train.to_numpy(), y_val.to_numpy()

    # Build PyTorch DataLoaders (no exogenous data by default)
    train_loader = to_torch_dataset(
        X_train_ts,
        y_train_np,
        num_lags=args.num_lags,
        num_features=num_features,
        indices=args.idxs,
        batch_size=args.batch_size,
        exogenous_data=None,
        shuffle=True,
    )
    val_loader = to_torch_dataset(
        X_val_ts,
        y_val_np,
        num_lags=args.num_lags,
        num_features=num_features,
        indices=args.idxs,
        batch_size=args.batch_size,
        exogenous_data=None,
        shuffle=False,
    )

    y_scaler = None
    if isinstance(y_scalers, dict) and len(y_scalers) > 0:
        y_scaler = next(iter(y_scalers.values()))

    prediction_artifacts: Dict[str, Any] = {
        "X_val_ts": X_val_ts,
        "y_val_np": y_val_np,
        "y_val_index": y_val.index,
        "num_features": num_features,
        "target_names": list(y_val.columns),
        "base_target_names": list(args.targets),
        "prediction_steps": args.prediction_steps,
        "y_scaler": y_scaler,
    }

    return train_loader, val_loader, num_features, y_train_np.shape[1], prediction_artifacts


class FlowerTimeSeriesClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        args: argparse.Namespace,
    ) -> None:
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
        self.criterion = get_criterion(args.criterion)

    def get_parameters(self, config):  # type: ignore[override]
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def _set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):  # type: ignore[override]
        if parameters is not None:
            self._set_parameters(parameters)

        start_time = time.time()
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
        round_train_time = time.time() - start_time

        loss, mse, rmse, mae, r2, nrmse = test(
            self.model,
            self.train_loader,
            self.criterion,
            device=self.device,
        )

        # Cast metrics to plain Python floats so Flower can serialize them
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "nrmse": float(nrmse),
        }

        # Simple wandb logging for local training metrics
        if wandb is not None and getattr(self.args, "wandb", False):
            wandb.log(
                {
                    "client/train_loss": float(loss),
                    "client/train_mse": float(mse),
                    "client/train_rmse": float(rmse),
                    "client/train_mae": float(mae),
                    "client/train_r2": float(r2),
                    "client/train_nrmse": float(nrmse),
                    "client/round_train_time_seconds": float(round_train_time),
                }
            )

        return self.get_parameters(config), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):  # type: ignore[override]
        if parameters is not None:
            self._set_parameters(parameters)

        loss, mse, rmse, mae, r2, nrmse = test(
            self.model,
            self.val_loader,
            self.criterion,
            device=self.device,
        )

        # Cast metrics to plain Python floats so Flower can serialize them
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "nrmse": float(nrmse),
        }

        # Simple wandb logging for local validation metrics
        if wandb is not None and getattr(self.args, "wandb", False):
            wandb.log(
                {
                    "client/val_loss": float(loss),
                    "client/val_mse": float(mse),
                    "client/val_rmse": float(rmse),
                    "client/val_mae": float(mae),
                    "client/val_r2": float(r2),
                    "client/val_nrmse": float(nrmse)
                }
            )

        return float(loss), len(self.val_loader.dataset), metrics


def save_client_model(model: torch.nn.Module, args: argparse.Namespace) -> Path:
    save_path = Path(args.model_save_path.format(cid=args.cid, model_name=args.model_name)).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cpu_state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    checkpoint = {
        "cid": args.cid,
        "model_name": args.model_name,
        "num_lags": args.num_lags,
        "targets": args.targets,
        "state_dict": cpu_state_dict,
    }
    torch.save(checkpoint, save_path)
    return save_path


def save_last_lags_predictions(
    model: torch.nn.Module,
    device: str,
    args: argparse.Namespace,
    prediction_artifacts: Dict[str, Any],
) -> Path:
    X_val_ts = prediction_artifacts["X_val_ts"]
    y_val_np = prediction_artifacts["y_val_np"]
    y_val_index = prediction_artifacts["y_val_index"]
    num_features = prediction_artifacts["num_features"]
    target_names = prediction_artifacts["target_names"]
    base_target_names = prediction_artifacts.get("base_target_names", target_names)
    prediction_steps = prediction_artifacts.get("prediction_steps", 1)
    y_scaler = prediction_artifacts["y_scaler"]

    if len(X_val_ts) == 0:
        raise ValueError("Validation data is empty. Cannot produce final predictions.")

    horizon = min(args.num_lags, len(y_val_np))
    if horizon <= 0:
        raise ValueError("Not enough validation samples to produce forecasts.")

    y_pred_seq = []
    model.eval()
    with torch.no_grad():
        for sample_idx in range(horizon):
            current_window = np.array(X_val_ts[sample_idx], dtype=np.float32)
            x_tensor = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)

            y_hist_np = np.zeros((args.num_lags, len(args.idxs)), dtype=np.float32)
            if args.num_lags > 1 and len(args.idxs) > 0:
                y_hist_np[1:, :] = current_window[:-1, args.idxs, 0]
            y_hist_tensor = torch.tensor(y_hist_np, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(x_tensor, None, device, y_hist_tensor)
            pred_np = pred.detach().cpu().numpy().reshape(-1)
            y_pred_seq.append(pred_np)

    y_pred = np.vstack(y_pred_seq)
    y_true = y_val_np[:horizon]
    idx_last = y_val_index[:horizon]

    if y_scaler is not None:
        if prediction_steps <= 1:
            y_pred = y_scaler.inverse_transform(y_pred)
            y_true = y_scaler.inverse_transform(y_true)
        else:
            num_base_targets = len(base_target_names)
            expected_width = num_base_targets * prediction_steps
            if y_pred.shape[1] != expected_width or y_true.shape[1] != expected_width:
                raise ValueError(
                    f"Expected {expected_width} outputs before inverse scaling, "
                    f"got y_true={y_true.shape[1]} and y_pred={y_pred.shape[1]}."
                )

            # The scaler is fit on base targets only (e.g., 4 cols). For multi-step outputs
            # (e.g., 16 cols), inverse-transform per step by flattening to base-target width.
            y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, num_base_targets)).reshape(
                -1, expected_width
            )
            y_true = y_scaler.inverse_transform(y_true.reshape(-1, num_base_targets)).reshape(
                -1, expected_width
            )

    records = {"time": idx_last}
    if prediction_steps <= 1:
        for col_idx, target_name in enumerate(target_names):
            records[f"true_{target_name}"] = y_true[:, col_idx]
            records[f"pred_{target_name}"] = y_pred[:, col_idx]
    else:
        num_base_targets = len(base_target_names)
        expected_width = num_base_targets * prediction_steps
        if y_true.shape[1] != expected_width or y_pred.shape[1] != expected_width:
            raise ValueError(
                f"Expected {expected_width} forecast outputs for {num_base_targets} targets and "
                f"{prediction_steps} steps, got y_true={y_true.shape[1]} and y_pred={y_pred.shape[1]}."
            )
        for step in range(prediction_steps):
            offset = step * num_base_targets
            for target_idx, target_name in enumerate(base_target_names):
                col_idx = offset + target_idx
                records[f"true_{target_name}_step+{step + 1}"] = y_true[:, col_idx]
                records[f"pred_{target_name}_step+{step + 1}"] = y_pred[:, col_idx]

    predictions_df = pd.DataFrame(records)

    save_path = Path(
        args.prediction_save_path.format(
            cid=args.cid,
            model_name=args.model_name,
            num_lags=horizon,
        )
    ).expanduser()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(save_path, index=False)
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower client for federated time-series forecasting")

    # Federated settings
    parser.add_argument("--cid", type=str, default="12167-0", help="Client ID (e.g. District name)")
    parser.add_argument(
        "--server_address",
        type=str,
        default="127.0.0.1:8080",
        help="Flower server address, e.g. '127.0.0.1:8080'",
    )

    # Data / preprocessing (mirrors main.py defaults where possible)
    parser.add_argument("--data_path", type=str, default="./dataset/5G-2y-bs12167.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=[
            "PRB Usage Ratio (%)",
            "Traffic Volume (KByte)",
            "Number of Users",
            "BBU Energy (W)",
        ],
    )
    parser.add_argument("--idxs", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--num_lags", type=int, default=48)
    parser.add_argument("--prediction_steps", type=int, default=4)
    parser.add_argument("--identifier", type=str, default="District")
    parser.add_argument("--nan_constant", type=float, default=0.0)
    parser.add_argument("--x_scaler", type=str, default="minmax")
    parser.add_argument("--y_scaler", type=str, default="minmax")
    parser.add_argument("--outlier_detection", type=str, default=None)

    # Model / training
    parser.add_argument("--model_name", type=str, default="lstm")
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

    # wandb options
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable wandb logging for this client")
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

    # Outlier configuration (kept compatible with main.py when enabled)
    if args.outlier_detection is not None:
        args.outlier_columns = ["rb_down", "rb_up", "down", "up"]
        args.outlier_kwargs = {"ElBorn": (10, 90), "LesCorts": (10, 90), "PobleSec": (5, 95)}

    # Convert idxs list from strings if needed
    if isinstance(args.idxs, list) and len(args.idxs) > 0 and isinstance(args.idxs[0], str):
        args.idxs = [int(i) for i in args.idxs]

    return args


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    # Initialize wandb run for this client (optional)
    wb_run = None
    if wandb is not None and getattr(args, "wandb", False):
        wb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"flwr-client-{args.cid}-{args.model_name}",
            mode="online",
        )
        wandb.config.update({"cid": args.cid, "model_name": args.model_name}, allow_val_change=True)

    train_loader, val_loader, num_features, out_dim, prediction_artifacts = prepare_client_data(args)

    # Determine model input dimension
    if args.model_name == "mlp":
        input_dim = num_features * args.num_lags
    else:
        input_dim = num_features

    model = get_model(
        model_name=args.model_name,
        input_dim=input_dim,
        out_dim=out_dim,
        lags=args.num_lags,
        exogenous_dim=0,
    )

    client = FlowerTimeSeriesClient(
        cid=args.cid,
        model=model,
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

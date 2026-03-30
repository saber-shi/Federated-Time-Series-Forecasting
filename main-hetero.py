import sys
import os

from pathlib import Path

parent = Path(os.path.abspath("")).resolve().parents[0]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

import copy
import importlib.util
import random
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from ml.fl.client.client import Client
from ml.fl.client_proxy import SimpleClientProxy
from ml.fl.history.history import History
from ml.fl.server.client_manager import SimpleClientManager
from ml.models.cnn import CNN
from ml.models.gru import GRU
from ml.models.lstm import LSTM
from ml.models.mlp import MLP
from ml.models.rnn import RNN
from ml.models.rnn_autoencoder import DualAttentionAutoEncoder
from src.spa_hfl import (
    AlignmentProjector,
    aggregate_ndarrays,
    evaluate_spa,
    load_state_dict_from_ndarrays,
    pack_state_dict,
    train_spa_hfl,
    update_centroid,
)
from ml.utils.data_utils import (
    assign_statistics,
    generate_time_lags,
    get_data_by_area,
    get_exogenous_data_by_area,
    handle_nans,
    handle_outliers,
    read_data,
    remove_identifiers,
    scale_features,
    time_to_feature,
    to_Xy,
    to_timeseries_rep,
    to_torch_dataset,
    to_train_val,
)
from ml.utils.helpers import get_criterion
from ml.utils.train_utils import test

try:
    import wandb  # optional
except Exception:  # pragma: no cover
    wandb = None


def _load_hetero_adapter():
    module_path = Path(__file__).with_name("client-hetero.py")
    spec = importlib.util.spec_from_file_location("client_hetero_file", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load heterogeneous client helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HeteroModelAdapter


HeteroModelAdapter = _load_hetero_adapter()


def seed_all() -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_preprocessing():
    df = read_data(args.data_path)
    df = handle_nans(train_data=df, constant=args.nan_constant, identifier=args.identifier)
    train_data, val_data = to_train_val(df)

    if args.outlier_detection is not None:
        train_data = handle_outliers(
            df=train_data,
            columns=args.outlier_columns,
            identifier=args.identifier,
            kwargs=args.outlier_kwargs,
        )

    X_train, X_val, y_train, y_val = to_Xy(train_data=train_data, val_data=val_data, targets=args.targets)

    X_train, X_val, x_scalers = scale_features(
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

    X_train = generate_time_lags(X_train, args.num_lags)
    X_val = generate_time_lags(X_val, args.num_lags)
    y_train = generate_time_lags(y_train, args.num_lags, is_y=True)
    y_val = generate_time_lags(y_val, args.num_lags, is_y=True)

    date_time_df_train = time_to_feature(X_train, args.use_time_features, identifier=args.identifier)
    date_time_df_val = time_to_feature(X_val, args.use_time_features, identifier=args.identifier)
    stats_df_train = assign_statistics(
        X_train, args.assign_stats, args.num_lags, targets=args.targets, identifier=args.identifier
    )
    stats_df_val = assign_statistics(
        X_val, args.assign_stats, args.num_lags, targets=args.targets, identifier=args.identifier
    )

    if date_time_df_train is not None or stats_df_train is not None:
        exogenous_data_train = pd.concat([date_time_df_train, stats_df_train], axis=1)
        exogenous_data_train = exogenous_data_train.loc[:, ~exogenous_data_train.columns.duplicated()].copy()
    else:
        exogenous_data_train = None

    if date_time_df_val is not None or stats_df_val is not None:
        exogenous_data_val = pd.concat([date_time_df_val, stats_df_val], axis=1)
        exogenous_data_val = exogenous_data_val.loc[:, ~exogenous_data_val.columns.duplicated()].copy()
    else:
        exogenous_data_val = None

    return X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers


def make_postprocessing(X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers):
    if X_train[args.identifier].nunique() != 1:
        area_X_train, area_X_val, area_y_train, area_y_val = get_data_by_area(
            X_train, X_val, y_train, y_val, identifier=args.identifier
        )
    else:
        area_X_train, area_X_val, area_y_train, area_y_val = None, None, None, None

    if exogenous_data_train is not None:
        exogenous_data_train, exogenous_data_val = get_exogenous_data_by_area(exogenous_data_train, exogenous_data_val)

    if area_X_train is not None:
        for area in area_X_train:
            tmp_X_train, tmp_y_train, tmp_X_val, tmp_y_val = remove_identifiers(
                area_X_train[area], area_y_train[area], area_X_val[area], area_y_val[area]
            )
            area_X_train[area] = tmp_X_train.to_numpy()
            area_X_val[area] = tmp_X_val.to_numpy()
            area_y_train[area] = tmp_y_train.to_numpy()
            area_y_val[area] = tmp_y_val.to_numpy()

    if exogenous_data_train is not None:
        for area in exogenous_data_train:
            exogenous_data_train[area] = exogenous_data_train[area].to_numpy()
            exogenous_data_val[area] = exogenous_data_val[area].to_numpy()

    X_train, y_train, X_val, y_val = remove_identifiers(X_train, y_train, X_val, y_val)
    num_features = len(X_train.columns) // args.num_lags

    X_train = to_timeseries_rep(X_train.to_numpy(), num_lags=args.num_lags, num_features=num_features)
    X_val = to_timeseries_rep(X_val.to_numpy(), num_lags=args.num_lags, num_features=num_features)

    if area_X_train is not None:
        area_X_train = to_timeseries_rep(area_X_train, num_lags=args.num_lags, num_features=num_features)
        area_X_val = to_timeseries_rep(area_X_val, num_lags=args.num_lags, num_features=num_features)

    y_train, y_val = y_train.to_numpy(), y_val.to_numpy()

    if exogenous_data_train is not None:
        exogenous_data_train_combined, exogenous_data_val_combined = [], []
        for area in exogenous_data_train:
            exogenous_data_train_combined.extend(exogenous_data_train[area])
            exogenous_data_val_combined.extend(exogenous_data_val[area])
        exogenous_data_train["all"] = np.stack(exogenous_data_train_combined)
        exogenous_data_val["all"] = np.stack(exogenous_data_val_combined)

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        area_X_train,
        area_X_val,
        area_y_train,
        area_y_val,
        exogenous_data_train,
        exogenous_data_val,
    )


def get_input_dims(X_train, exogenous_data_train):
    if args.model_name == "mlp":
        input_dim = X_train.shape[1] * X_train.shape[2]
    else:
        input_dim = X_train.shape[2]

    if exogenous_data_train is not None:
        if len(exogenous_data_train) == 1:
            cid = next(iter(exogenous_data_train.keys()))
            exogenous_dim = exogenous_data_train[cid].shape[1]
        else:
            exogenous_dim = exogenous_data_train["all"].shape[1]
    else:
        exogenous_dim = 0

    return input_dim, exogenous_dim


def get_model(model: str, input_dim: int, out_dim: int, lags: int = 10, exogenous_dim: int = 0):
    if model == "mlp":
        return MLP(input_dim=input_dim, layer_units=[256, 128, 64], num_outputs=out_dim)
    if model == "rnn":
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
    if model == "lstm":
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
    if model == "gru":
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
    if model == "cnn":
        return CNN(num_features=input_dim, lags=lags, exogenous_dim=exogenous_dim, out_dim=out_dim)
    if model == "da_encoder_decoder":
        return DualAttentionAutoEncoder(input_dim=input_dim, architecture="lstm", matrix_rep=True)
    raise NotImplementedError("Choose one from ['mlp', 'rnn', 'lstm', 'gru', 'cnn', 'da_encoder_decoder']")


class HeteroTorchRegressionClient(Client):
    def __init__(
        self,
        cid: Union[str, int],
        adapter: HeteroModelAdapter,
        train_loader,
        val_loader,
        local_train_params,
    ):
        self.cid = cid
        self.adapter = adapter
        self.net = adapter.local_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.initial_train_params = local_train_params
        self.epochs = None
        self.optimizer = None
        self.lr = None
        self.criterion = None
        self.early_stopping = None
        self.patience = None
        self.device = None
        self.reg1 = None
        self.reg2 = None
        self.max_grad_norm = None
        self.fed_prox_mu = None
        self.projector = AlignmentProjector(input_dim=self.net.hidden_dim, align_dim=self.initial_train_params["align_dim"])
        self._init_local_train_params()

    def _init_local_train_params(self):
        self.epochs = self.initial_train_params["epochs"]
        self.optimizer = self.initial_train_params["optimizer"]
        self.lr = self.initial_train_params["lr"]
        self.criterion = self.initial_train_params["criterion"]
        self.early_stopping = self.initial_train_params["early_stopping"]
        self.patience = self.initial_train_params["patience"]
        self.device = self.initial_train_params["device"]
        self.reg1 = self.initial_train_params.get("reg1", 0.0)
        self.reg2 = self.initial_train_params.get("reg2", 0.0)
        self.max_grad_norm = self.initial_train_params.get("max_grad_norm", 0.0)
        self.fed_prox_mu = self.initial_train_params.get("fedprox_mu", 0.0)

    def get_parameters(self) -> List[np.ndarray]:
        return self.adapter.export_parameters_with_masks()

    def get_alignment_state(self) -> Dict[str, List[np.ndarray]]:
        return {
            "model": self.adapter.export_parameters_with_masks(),
            "projector": pack_state_dict(self.projector),
        }

    def set_train_parameters(self, params, verbose: bool = False):
        self.epochs = params["epochs"] if "epochs" in params else self.epochs
        self.optimizer = params["optimizer"] if "optimizer" in params else self.optimizer
        self.lr = params["lr"] if "lr" in params else self.lr
        self.criterion = params["criterion"] if "criterion" in params else self.criterion
        self.early_stopping = params["early_stopping"] if "early_stopping" in params else self.early_stopping
        self.patience = params["patience"] if "patience" in params else self.patience
        self.device = params["device"] if "device" in params else self.device
        self.reg1 = params["reg1"] if "reg1" in params else self.reg1
        self.reg2 = params["reg2"] if "reg2" in params else self.reg2
        self.max_grad_norm = params["max_grad_norm"] if "max_grad_norm" in params else self.max_grad_norm
        self.fed_prox_mu = params["fedprox_mu"] if "fedprox_mu" in params else self.fed_prox_mu

    def set_parameters(self, parameters):
        if isinstance(parameters, torch.nn.Module):
            self.net.load_state_dict(parameters.state_dict(), strict=True)
        elif isinstance(parameters, dict):
            if "model" in parameters:
                self.adapter.load_global_parameters(parameters["model"])
                self.net = self.adapter.local_model
            if "projector" in parameters and parameters["projector"] is not None:
                load_state_dict_from_ndarrays(self.projector, parameters["projector"])
        else:
            self.adapter.load_global_parameters(parameters)
            self.net = self.adapter.local_model

    def fit(self, model: Optional[Union[torch.nn.Module, List[np.ndarray]]] = None):
        if model is not None:
            self.set_parameters(model)

        global_centroid = model.get("centroid") if isinstance(model, dict) else None

        self.net, self.projector, alignment_stats, alignment_metrics = train_spa_hfl(
            model=self.net,
            projector=self.projector,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.epochs,
            optimizer_name=self.optimizer,
            lr=self.lr,
            criterion_name=self.criterion,
            device=self.device,
            lambda_align=self.initial_train_params["lambda_align"],
            lambda_cons=self.initial_train_params["lambda_cons"],
            global_centroid=global_centroid,
            acf_lags=self.initial_train_params["acf_lags"],
            fft_bins=self.initial_train_params["fft_bins"],
            early_stopping=self.early_stopping,
            patience=self.patience,
            max_grad_norm=self.max_grad_norm,
        )
        self.adapter.local_model = self.net
        _, train_loss, train_metrics = self.evaluate(self.train_loader)
        num_test, test_loss, test_metrics = self.evaluate(self.val_loader)
        train_metrics["local_num_layers"] = float(self.adapter.local_num_layers)
        test_metrics["local_num_layers"] = float(self.adapter.local_num_layers)
        train_metrics["align_train_loss"] = alignment_metrics["train_loss"]
        train_metrics["align_train_rmse"] = alignment_metrics["train_rmse"]
        return (
            {
                "model": self.adapter.export_parameters_with_masks(),
                "projector": pack_state_dict(self.projector),
                "alignment_stats": alignment_stats,
            },
            len(self.train_loader.dataset),
            train_loss,
            train_metrics,
            num_test,
            test_loss,
            test_metrics,
        )

    def evaluate(self, data=None, model=None, params=None, method=None, verbose=False):
        if not params or "criterion" not in params:
            params = {"criterion": get_criterion(self.criterion)}

        if model is not None:
            self.set_parameters(model)

        if data is None and method == "test":
            data = self.val_loader
        if data is None and method == "train":
            data = self.train_loader

        loss, mse, rmse, mae, r2, nrmse = test(self.net, data, params["criterion"], device=self.device)
        metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R^2": r2, "NRMSE": nrmse}
        return len(data.dataset), loss, metrics


class HeteroServer:
    def __init__(self, client_proxies):
        self.client_proxies = client_proxies
        self.client_manager = SimpleClientManager()
        for client_proxy in self.client_proxies:
            self.client_manager.register(client_proxy)
        self.global_model = None
        self.global_projector = None
        self.global_centroid = None
        self.best_model = None
        self.best_loss = np.inf
        self.best_epoch = -1

    def _aggregate_masked(self, results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        previous_parameters = self.global_model
        split_index = len(results[0][0]) // 2
        numerators = [np.zeros_like(arr) for arr in results[0][0][:split_index]]
        denominators = [np.zeros_like(arr, dtype=np.float32) for arr in results[0][0][:split_index]]

        for arrays, num_examples in results:
            local_parameters = arrays[:split_index]
            local_masks = arrays[split_index:]
            for idx, (param, mask) in enumerate(zip(local_parameters, local_masks)):
                numerators[idx] += num_examples * param * mask
                denominators[idx] += num_examples * mask.astype(np.float32, copy=False)

        aggregated = []
        for idx in range(split_index):
            fallback = previous_parameters[idx] if previous_parameters is not None else np.zeros_like(numerators[idx])
            aggregated_array = np.where(denominators[idx] > 0, numerators[idx] / denominators[idx], fallback)
            aggregated.append(aggregated_array.astype(fallback.dtype, copy=False))
        return aggregated

    def _get_initial_model(self) -> List[np.ndarray]:
        random_client = self.client_manager.sample(0.0)[0]
        full = random_client.client.get_alignment_state()
        self.global_projector = full["projector"]
        return full["model"][: len(full["model"]) // 2]

    def evaluate_round(self, fl_round: int, history: History):
        num_train_examples, train_losses, train_metrics = [], {}, {}
        num_test_examples, test_losses, test_metrics = [], {}, {}

        if fl_round == 0:
            self.global_model = self._get_initial_model()

        for cid, client_proxy in self.client_manager.all().items():
            shared_state = {
                "model": self.global_model,
                "projector": self.global_projector,
                "centroid": self.global_centroid,
            }
            num_train_instances, train_loss, train_eval_metrics = client_proxy.evaluate(model=shared_state, method="train")
            num_test_instances, test_loss, test_eval_metrics = client_proxy.evaluate(model=shared_state, method="test")
            num_train_examples.append(num_train_instances)
            num_test_examples.append(num_test_instances)
            train_losses[cid] = train_loss
            test_losses[cid] = test_loss
            train_metrics[cid] = train_eval_metrics
            test_metrics[cid] = test_eval_metrics

        history.add_local_train_loss(train_losses, fl_round)
        history.add_local_train_metrics(train_metrics, fl_round)
        history.add_local_test_loss(test_losses, fl_round)
        history.add_local_test_metrics(test_metrics, fl_round)

        weighted_test_loss = sum(n * loss for n, loss in zip(num_test_examples, test_losses.values())) / sum(num_test_examples)
        if weighted_test_loss < self.best_loss:
            self.best_loss = weighted_test_loss
            self.best_epoch = fl_round
            self.best_model = copy.deepcopy(self.global_model)

    def fit(self, num_rounds: int, fraction: float) -> Tuple[List[np.ndarray], History]:
        history = History()
        self.evaluate_round(fl_round=0, history=history)

        for fl_round in range(1, num_rounds + 1):
            selected_clients = self.client_manager.sample(fraction)
            model_results = []
            projector_results = []
            stats_results = []
            train_losses, test_losses, all_train_metrics, all_test_metrics = {}, {}, {}, {}
            for client in selected_clients:
                shared_state = {
                    "model": self.global_model,
                    "projector": self.global_projector,
                    "centroid": self.global_centroid,
                }
                fit_res = client.fit(model=shared_state)
                state_payload, num_train, train_loss, train_metrics, _, test_loss, test_metrics = fit_res
                model_results.append((state_payload["model"], num_train))
                projector_results.append((state_payload["projector"], num_train))
                stats_results.append((state_payload["alignment_stats"], num_train))
                train_losses[client.cid] = train_loss
                test_losses[client.cid] = test_loss
                all_train_metrics[client.cid] = train_metrics
                all_test_metrics[client.cid] = test_metrics

            history.add_local_train_loss(train_losses, fl_round)
            history.add_local_train_metrics(all_train_metrics, fl_round)
            history.add_local_test_loss(test_losses, fl_round)
            history.add_local_test_metrics(all_test_metrics, fl_round)

            self.global_model = self._aggregate_masked(model_results)
            self.global_projector = aggregate_ndarrays(projector_results, previous=self.global_projector)
            self.global_centroid = update_centroid(
                stats_results,
                previous_centroid=self.global_centroid,
                momentum=args.centroid_momentum,
            )
            if self.best_model is None:
                self.best_model = copy.deepcopy(self.global_model)

            self.evaluate_round(fl_round=fl_round, history=history)

        return self.best_model, history


def fit_hetero(
    client_layer_map: Dict[str, int],
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim: int,
    out_dim: int,
    idxs,
    exogenous_data_train=None,
    exogenous_data_val=None,
    local_train_params=None,
):
    if local_train_params is None:
        local_train_params = {
            "epochs": args.epochs,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "criterion": args.criterion,
            "early_stopping": args.local_early_stopping,
            "patience": args.local_patience,
            "device": device,
            "reg1": args.reg1,
            "reg2": args.reg2,
            "max_grad_norm": args.max_grad_norm,
            "align_dim": args.align_dim,
            "lambda_align": args.lambda_align,
            "lambda_cons": args.lambda_cons,
            "acf_lags": args.acf_lags,
            "fft_bins": args.fft_bins,
        }

    train_loaders, val_loaders = [], []
    cids = [k for k in X_train.keys() if k != "all"]

    for client in cids:
        tmp_exogenous_data_train = exogenous_data_train[client] if exogenous_data_train is not None else None
        tmp_exogenous_data_val = exogenous_data_val[client] if exogenous_data_val is not None else None
        num_features = len(X_train[client][0][0])

        train_loaders.append(
            to_torch_dataset(
                X_train[client],
                y_train[client],
                num_lags=args.num_lags,
                num_features=num_features,
                exogenous_data=tmp_exogenous_data_train,
                indices=idxs,
                batch_size=args.batch_size,
                shuffle=False,
            )
        )
        val_loaders.append(
            to_torch_dataset(
                X_val[client],
                y_val[client],
                num_lags=args.num_lags,
                num_features=num_features,
                exogenous_data=tmp_exogenous_data_val,
                indices=idxs,
                batch_size=args.batch_size,
                shuffle=False,
            )
        )

    clients = []
    for cid, train_loader, val_loader in zip(cids, train_loaders, val_loaders):
        adapter = HeteroModelAdapter(
            model_name=args.model_name,
            input_dim=input_dim,
            out_dim=out_dim,
            local_num_layers=client_layer_map[cid],
            global_num_layers=args.global_num_layers,
        )
        clients.append(
            HeteroTorchRegressionClient(
                cid=cid,
                adapter=adapter,
                train_loader=train_loader,
                val_loader=val_loader,
                local_train_params=local_train_params,
            )
        )

    client_proxies = [SimpleClientProxy(cid, client) for cid, client in zip(cids, clients)]
    server = HeteroServer(client_proxies=client_proxies)
    return server.fit(args.fl_rounds, args.fraction)


if __name__ == "__main__":
    args = Namespace(
        data_path="./dataset/5G-2y-bs12167.csv",
        test_size=0.2,
        targets=["PRB Usage Ratio (%)", "Traffic Volume (KByte)", "Number of Users", "BBU Energy (W)"],
        idxs=[2, 3, 4, 5],
        num_lags=48,
        identifier="District",
        nan_constant=0,
        x_scaler="minmax",
        y_scaler="minmax",
        outlier_detection=None,
        timestamp_column="Timestamp",
        criterion="mse",
        fl_rounds=10,
        fraction=1.0,
        aggregation="heterofl",
        epochs=3,
        lr=0.001,
        optimizer="adam",
        batch_size=128,
        local_early_stopping=False,
        local_patience=50,
        max_grad_norm=0.0,
        reg1=0.0,
        reg2=0.0,
        align_dim=32,
        lambda_align=0.1,
        lambda_cons=0.1,
        acf_lags=8,
        fft_bins=8,
        centroid_momentum=0.9,
        cuda=True,
        seed=0,
        assign_stats=None,
        use_time_features=False,
        model_name="lstm",
        global_num_layers=3,
        client_layer_map={"12167-0": 1},
    )

    print(f"Script arguments: {args}\n")
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    wb_run = None
    if wandb is not None:
        wb_run = wandb.init(
            entity="slife2026-university-of-hong-kong",
            project="federated-time-series-forecasting",
            name=f"time-series-forecasting-hetero-{args.model_name}",
            mode="online",
        )
        wandb.config.update({"device": device, "global_num_layers": args.global_num_layers}, allow_val_change=True)

    if args.outlier_detection is not None:
        args.outlier_columns = ["rb_down", "rb_up", "down", "up"]
        args.outlier_kwargs = {"ElBorn": (10, 90), "LesCorts": (10, 90), "PobleSec": (5, 95)}

    if args.model_name not in {"rnn", "lstm", "gru"}:
        raise ValueError("main-hetero.py currently supports recurrent models only: ['rnn', 'lstm', 'gru']")

    seed_all()

    X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers = make_preprocessing()
    (
        X_train,
        X_val,
        y_train,
        y_val,
        client_X_train,
        client_X_val,
        client_y_train,
        client_y_val,
        exogenous_data_train,
        exogenous_data_val,
    ) = make_postprocessing(
        X_train,
        X_val,
        y_train,
        y_val,
        exogenous_data_train,
        exogenous_data_val,
        x_scalers,
        y_scalers,
    )

    if client_X_train is None:
        raise ValueError("Heterogeneous federated learning requires multiple client partitions.")

    available_clients = [cid for cid in client_X_train.keys() if cid != "all"]
    default_layer_map = {cid: 1 + (idx % args.global_num_layers) for idx, cid in enumerate(sorted(available_clients))}
    layer_map = copy.deepcopy(default_layer_map)
    layer_map.update(args.client_layer_map)
    for cid in available_clients:
        if cid not in layer_map:
            layer_map[cid] = 1
        if layer_map[cid] > args.global_num_layers:
            raise ValueError(f"Client {cid} requests {layer_map[cid]} layers but global_num_layers={args.global_num_layers}")

    for client in client_X_train:
        print(f"\nClient: {client}")
        print(f"X_train shape: {client_X_train[client].shape}, y_train shape: {client_y_train[client].shape}")
        print(f"X_val shape: {client_X_val[client].shape}, y_val shape: {client_y_val[client].shape}")
        print(f"Assigned recurrent layers: {layer_map[client]}")

    input_dim, _ = get_input_dims(X_train, exogenous_data_train)

    local_train_params = {
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "criterion": args.criterion,
        "early_stopping": args.local_early_stopping,
        "patience": args.local_patience,
        "device": device,
        "reg1": args.reg1,
        "reg2": args.reg2,
        "max_grad_norm": args.max_grad_norm,
    }

    global_model, history = fit_hetero(
        client_layer_map=layer_map,
        X_train=client_X_train,
        y_train=client_y_train,
        X_val=client_X_val,
        y_val=client_y_val,
        input_dim=input_dim,
        out_dim=y_train.shape[1],
        idxs=args.idxs,
        exogenous_data_train=exogenous_data_train,
        exogenous_data_val=exogenous_data_val,
        local_train_params=local_train_params,
    )

    print(f"Best heterogeneous global model obtained with {len(global_model)} tensors.")

    if wb_run is not None:
        wandb.finish()

import copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torchmetrics
from torch import nn

from pbim_models.dataset.model import DatasetMetadata
from pbim_models.metrics.metric import (
    TimeSeriesMAE,
    TimeSeriesMSE,
    TimeSeriesRMSE,
    AUROC,
    TimeSeriesMAPE,
    Precision,
    Recall,
    F1,
    ErrorDistributionMetric,
    Accuracy,
)
from pbim_models.models.supervised.lstm import SupervisedLSTMModel
from pbim_models.models.unsupervised.agcrn.net import AGCRN
from pbim_models.models.unsupervised.lstm.net import LSTMModel
from pbim_models.models.unsupervised.mlp.net import MLP
from pbim_models.models.unsupervised.mtgnn.net import MtGNN
from pbim_models.models.unsupervised.stemgnn.net import StemGNN
from pbim_models.models.unsupervised.stfgnn.net import STFGNN
from pbim_models.models.unsupervised.stgcn.net import STGCN
from pbim_models.models.unsupervised.tcn.net import TCN
from pbim_models.models.unsupervised.tcnae.net import TCNAE
from pbim_models.models.unsupervised.tgcn.net import TGCNModel
from pbim_models.models.util import LossWrapper, Stage
from pbim_models.models.unsupervised.vrae.net import VRAE


def _make_full_from_triu(triu: np.ndarray) -> np.ndarray:
    return triu + triu.T - np.diag(triu.diagonal())


def _min_max_normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())


def build_adjacency_matrix(
    config: Dict[str, Any], as_tensor: bool = True
) -> torch.Tensor | np.ndarray:
    adjacency_matrix_config = config["adjacency_matrix"]
    adjacency_matrix_type = adjacency_matrix_config["type"]
    num_nodes = adjacency_matrix_config.get("num_nodes")
    match adjacency_matrix_type:
        case "random":
            adj = _min_max_normalize(
                _make_full_from_triu(np.random.rand(num_nodes, num_nodes))
            )
        case "uniform":
            adj = np.ones((num_nodes, num_nodes)) / num_nodes
        case "file":
            adj = np.load(adjacency_matrix_config["path"])
        case _:
            raise ValueError(
                f"Unknown adjacency matrix type '{adjacency_matrix_type}'."
            )
    if as_tensor:
        device = get_map_location(config["trainer"].get("accelerator", None))
        return torch.from_numpy(adj).to(device=device, dtype=torch.float32)
    return adj


def build_stfgnn_adjacency_matrix(
    config: Dict[str, Any],
) -> torch.Tensor:
    device = get_map_location(config["trainer"].get("accelerator", None))
    k = config["adjacency_matrix"]["k"]
    spatial_adjacency_matrix_config = config["adjacency_matrix"]["spatial"]
    temporal_adjacency_matrix_config = config["adjacency_matrix"]["temporal"]
    spatial_adjacency_matrx = build_adjacency_matrix(
        {"adjacency_matrix": spatial_adjacency_matrix_config},
        as_tensor=False,
    )
    temporal_adjacency_matrx = build_adjacency_matrix(
        {"adjacency_matrix": temporal_adjacency_matrix_config},
        as_tensor=False,
    )
    return torch.from_numpy(
        _build_fusion_matrix(spatial_adjacency_matrx, temporal_adjacency_matrx, k)
    ).to(device=device, dtype=torch.float32)


def _build_fusion_matrix(
    spatial_adj: np.ndarray, temporal_adj: np.ndarray, steps: int = 4
) -> np.ndarray:
    N = len(spatial_adj)
    adj = np.zeros([N * steps] * 2)  # "steps" = 4 !!!

    for i in range(steps):
        if (i == 1) or (i == 2):
            adj[i * N : (i + 1) * N, i * N : (i + 1) * N] = spatial_adj
        else:
            adj[i * N : (i + 1) * N, i * N : (i + 1) * N] = temporal_adj
    # '''
    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1
    # '''
    adj[3 * N : 4 * N, 0:N] = temporal_adj  # adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0:N, 3 * N : 4 * N] = temporal_adj  # adj[0 * N : 1 * N, 1 * N : 2 * N]

    adj[2 * N : 3 * N, 0:N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[0:N, 2 * N : 3 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[1 * N : 2 * N, 3 * N : 4 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]
    adj[3 * N : 4 * N, 1 * N : 2 * N] = adj[0 * N : 1 * N, 1 * N : 2 * N]

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


def build_model(config: Dict[str, Any]) -> nn.Module:
    model_config = config["model"]

    def get(name: str, config: Dict[str, Any] = model_config, default=None):
        value = config.get(name, default)
        if value is None:
            raise ValueError(f"Missing value for key '{name}' in config.")
        return value

    match get("type").lower():
        case "agcrn":
            return AGCRN(
                num_nodes=get("input_size"),
                input_dim=get("input_channels"),
                output_dim=get("output_channels"),
                rnn_units=get("rnn_hidden_dim"),
                window_size=get("window_size", config=config["dataset"]["base"]),
                num_layers=get("num_layers"),
                embedding_dim=get("node_embedding_dim"),
                dropout=get("dropout"),
                cheb_k=get("cheb_k"),
            )
        case "stemgnn":
            return StemGNN(
                node_count=get("nodes"),
                stack_cnt=get("stack_count", default=2),
                multi_layer=get("multi_layer", default=5),
                window_size=get("window_size", config=config["dataset"]["base"]),
            )
        case "mtgnn":
            return MtGNN(
                num_nodes=get("input_size"),
                subgraph_size=get("input_size"),
                in_dim=get("input_channels"),
                window_size=get("window_size", config=config["dataset"]["base"]),
                gcn_depth=get("gcn_depth"),
                dropout=get("dropout"),
                dilation_exponential=get("dilation_exponential"),
                conv_channels=get("conv_channels"),
                residual_channels=get("residual_channels"),
                skip_channels=get("skip_channels"),
                layers=get("num_layers"),
            )
        case "lstm":
            return LSTMModel(
                input_size=get("input_size"),
                input_channels=get("input_channels"),
                hidden_size=get("hidden_size"),
                num_layers=get("num_layers"),
                output_size=get("output_size"),
                window_size=get("window_size", config=config["dataset"]["base"]),
                dropout=get("dropout", default=0.0),
                latent_size=get("latent_size", default=0),
                bidirectional=get("bidirectional", default=False),
            )
        case "tcn":
            channels = [get("hidden_size")] * get("num_layers")
            return TCN(
                input_size=get("input_size"),
                input_channels=get("input_channels"),
                output_size=get("output_size"),
                num_channels=channels,
                kernel_size=get("kernel_size"),
                dropout=get("dropout"),
                horizon=get("horizon_size", config=config["dataset"]["base"]),
            )

        case "tcnae":
            channels = [get("num_filter")] * get("num_layer")
            return TCNAE(
                input_size=get("input_size"),
                input_channels=get("input_channels"),
                output_channels=get("output_channels"),
                num_channels=channels,
                kernel_size=get("kernel_size"),
                dropout=get("dropout"),
                horizon=get("horizon_size", config=config["dataset"]["base"]),
                avg_pool_size=get("avg_pool_size"),
                conv_output_size=get("conv_output_size"),
                hidden_channels=get("hidden_size"),
            )
        case "vrae":
            return VRAE(
                batch_size=get("batch_size", config=config["dataset"]["base"]),
                window_size=get("window_size", config=config["dataset"]["base"]),
                input_size=get("input_size"),
                input_channels=get("input_channels"),
                output_size=get("output_size"),
                hidden_dim=get("hidden_dim"),
                latent_dim=get("latent_dim"),
                num_layers=get("num_layers"),
                dropout_rate=get("dropout"),
                block=get("block"),
                loss=build_loss(config),
            )
        case "stgcn":
            return STGCN(
                num_nodes=get("input_size"),
                dropout=get("dropout", default=0.1),
                window_size=get("window_size", config=config["dataset"]["base"]),
                horizon_size=get("horizon_size", config=config["dataset"]["base"]),
                bias=get("bias", default=True),
                num_blocks=get("num_blocks", default=3),
                kernel_size_gc=get("kernel_size_gc", default=3),
                kernel_size_tc=get("kernel_size_tc", default=3),
                input_channels_per_node=get("input_channels"),
                adjacency_matrix=build_adjacency_matrix(config),
            )
        case "tgcn":
            return TGCNModel(
                input_size=get("input_size"),
                hidden_size=get("hidden_size"),
                horizon_size=get("horizon_size", config=config["dataset"]["base"]),
                dropout=get("dropout", default=0.1),
                input_channels=get("input_channels"),
                output_channels=get("output_channels"),
                adjacency_matrix=build_adjacency_matrix(config),
            )
        case "stfgnn":
            filters = [get("filter")] * get("num_layer")
            return STFGNN(
                num_nodes=get("input_size"),
                window_size=get("window_size", config=config["dataset"]["base"]),
                horizon_size=get("horizon_size", config=config["dataset"]["base"]),
                input_channels=get("input_channels"),
                output_channels=get("output_channels"),
                filters=filters,
                activation=get("activation", default="glu"),
                temporal_emb=get("temporal_emb", default=True),
                spatial_emb=get("spatial_emb", default=True),
                huber_delta=get("huber_delta", default=1.0),
                dropout=get("dropout", default=0.1),
                adjacency_matrix=build_stfgnn_adjacency_matrix(config),
            )
        case "mlp":
            return MLP(
                window_size=get("window_size", config=config["dataset"]["base"]),
                input_size=get("input_size"),
                input_channels=get("input_channels"),
                output_size=get("output_size"),
                output_channels=get("output_channels"),
                hidden_sizes=get("hidden_sizes"),
                dropout=get("dropout"),
            )
        case "supervised_lstm":
            return SupervisedLSTMModel(
                input_size=get("input_size"),
                input_channels=get("input_channels"),
                hidden_size=get("hidden_size"),
                lstm_layers=get("lstm_layers"),
                classifier_layers=get("classifier_layers"),
                window_size=get("window_size", config=config["dataset"]["base"]),
                dropout=get("dropout", default=0.0),
                bidirectional=get("bidirectional", default=False),
            )
        case _:
            raise ValueError(f"Unknown model type {model_config['type']}")


def build_loss(config: Dict[str, Any]) -> nn.Module:
    loss_config = config["loss"]
    name = loss_config["type"]
    reduction = loss_config.get("reduction", "mean")
    multi_step = loss_config.get("multi_step", False)
    return LossWrapper(
        name=name,
        reduction=reduction,
        multi_step=multi_step,
    )


def build_metrics(config: Dict[str, Any]) -> nn.ModuleDict:
    def _build_ks(config: Optional[Dict[str, Any] | List[float]]):
        if config is None:
            return [0.5]
        elif isinstance(config, list):
            if len(config) == 0:
                return [0.5]
            return config
        elif isinstance(config, dict):
            start = config["start"]
            stop = config["stop"]
            step = config["step"]
            return list(np.arange(start, stop, step))
        else:
            raise ValueError(f"Unknown ks config {config}")

    def _build(metric_config: Dict[str, Any]) -> torchmetrics.Metric:
        if isinstance(metric_config, str):
            name = metric_config
        else:
            name = list(metric_config.keys())[0]
            metric_config = metric_config[name]
        match name:
            case "mae":
                return TimeSeriesMAE()
            case "mse":
                return TimeSeriesMSE()
            case "rmse":
                return TimeSeriesRMSE()
            case "mape":
                return TimeSeriesMAPE()
            case "error":
                return ErrorDistributionMetric()
            case "auroc":
                return AUROC(ks=_build_ks(metric_config.get("ks", [])))
            case "precision":
                return Precision(ks=_build_ks(metric_config.get("ks")))
            case "recall":
                return Recall(ks=_build_ks(metric_config.get("ks", [])))
            case "f1":
                return F1(ks=_build_ks(metric_config.get("ks", [])))
            case "accuracy":
                return Accuracy(ks=_build_ks(metric_config.get("ks", [])))

    metrics_config = config.get("metrics")
    if metrics_config is None:
        train_metrics = torchmetrics.MetricCollection([])
        val_metrics = torchmetrics.MetricCollection([])
        test_metrics = torchmetrics.MetricCollection([])
    elif isinstance(metrics_config, list):
        train_metrics = torchmetrics.MetricCollection(
            [_build(metric_config) for metric_config in metrics_config]
        )
        val_metrics = torchmetrics.MetricCollection(
            [_build(metric_config) for metric_config in metrics_config]
        )
        test_metrics = torchmetrics.MetricCollection(
            [_build(metric_config) for metric_config in metrics_config]
        )
    else:
        train_metrics = torchmetrics.MetricCollection(
            [_build(metric_config) for metric_config in metrics_config["train"]]
            if "train" in metrics_config
            else []
        )
        val_metrics = torchmetrics.MetricCollection(
            [_build(metric_config) for metric_config in metrics_config["val"]]
            if "val" in metrics_config
            else []
        )
        test_metrics = torchmetrics.MetricCollection(
            [_build(metric_config) for metric_config in metrics_config["test"]]
            if "test" in metrics_config
            else []
        )
    return nn.ModuleDict(
        {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
    )


def build_optimizers(
    config: Dict[str, Any], model: nn.Module
) -> List[torch.optim.Optimizer | Dict[str, Any]] | Tuple[
    List[torch.optim.Optimizer | Dict[str, Any]],
    List[torch.optim.lr_scheduler.ExponentialLR],
]:
    optimizers_config = config["optimizers"]

    def get(
        name: str,
        config: Dict[str, Any] = optimizers_config,
        default=None,
        allow_none=False,
    ):
        value = config.get(name, default)
        if value is None and not allow_none:
            raise ValueError(f"Missing value for key '{name}' in config.")
        return value

    optimizer_config = get("optimizer")
    match get("name", optimizer_config).lower():
        case "adam":
            optim = torch.optim.Adam(
                params=model.parameters(),
                lr=get("lr", optimizer_config),
                weight_decay=get("weight_decay", optimizer_config),
            )
        case "sgd":
            optim = torch.optim.SGD(
                params=model.parameters(),
                lr=get("lr", optimizer_config),
                momentum=get("momentum", optimizer_config),
                weight_decay=get("weight_decay", optimizer_config),
            )
        case _:
            raise ValueError(f"Unknown optimizer {get('name')}.")

    scheduler_config = get("scheduler", allow_none=True)
    if scheduler_config is None:
        return [optim]

    match get("name", scheduler_config).lower():
        case "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optim,
                gamma=get("gamma", scheduler_config),
            )
        case _:
            raise ValueError(f"Unknown scheduler {get('name', scheduler_config)}.")

    frequency = get("frequency", optimizer_config, default=1)
    return [{"optimizer": optim, "frequency": frequency, "scheduler": scheduler}]


def denormalize(
    x: torch.Tensor, channel_indices: List[int], metadata: DatasetMetadata
) -> torch.Tensor:
    denormalized = x.clone()
    for i, index in enumerate(channel_indices):
        channel = metadata.channel_order[index]
        denormalized[..., i, :] = (
            denormalized[..., i, :] * metadata.statistics[channel].std
            + metadata.statistics[channel].mean
        )
    return denormalized


def get_map_location(accelerator: Optional[str]) -> str:
    match accelerator:
        case None:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        case "cpu":
            return "cpu"
        case "gpu":
            return "cuda:0"
        case "mps":
            return "mps"


def malahanobis_distance(
    x: torch.Tensor, mean: torch.Tensor, inv_cov: torch.Tensor
) -> torch.Tensor:
    delta = x - mean
    return torch.sqrt(torch.einsum("ij,jk,ik->i", delta, inv_cov, delta))

from typing import Literal, List

import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        window_size: int,
        input_size: int,
        input_channels: int,
        use_temporal: bool,
        use_spatial: bool,
    ):
        super().__init__()
        self._window_size = window_size
        self._input_size = input_size
        self._input_channels = input_channels
        self._use_temporal = use_temporal
        self._use_spatial = use_spatial

        if use_temporal:
            self._temporal_embedding = nn.Parameter(
                torch.zeros(1, self._window_size, 1, self._input_channels)
            )
            torch.nn.init.xavier_normal_(self._temporal_embedding)
        if use_spatial:
            self._spatial_embedding = nn.Parameter(
                torch.zeros(1, 1, self._input_size, self._input_channels)
            )
            torch.nn.init.xavier_normal_(self._spatial_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self._use_temporal:
            x += self._temporal_embedding.repeat(batch_size, 1, self._input_size, 1)
        if self._use_spatial:
            x += self._spatial_embedding.repeat(batch_size, self._window_size, 1, 1)
        return x


class GraphConvolution(nn.Module):
    def __init__(
        self,
        activation: Literal["relu", "glu"],
        input_channels: int,
        output_channels: int,
    ):
        super().__init__()
        self._activation = activation
        self._output_channels = output_channels
        self._input_channels = input_channels

        if activation == "relu":
            self._conv = nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.ReLU(),
            )
        elif activation == "glu":
            self._conv = nn.Sequential(
                nn.Linear(input_channels, 2 * output_channels),
                nn.GLU(dim=-1),
            )

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        x = adjacency_matrix @ x
        x = self._conv(x)
        return x


class STSGCM(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        filters: List[int],
        activation: Literal["relu", "glu"],
    ):
        super().__init__()
        # If these filters are not the same size, everything blows up
        self._filters = filters
        self._num_nodes = num_nodes
        self._activation = activation

        self._conv = nn.ModuleList()
        for (in_features, out_features) in zip(filters[:-1], filters[1:]):
            self._conv.append(GraphConvolution(activation, in_features, out_features))

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        xs = []
        for conv in self._conv:
            x = conv(x, adjacency_matrix)
            xs.append(x[:, self._num_nodes : 2 * self._num_nodes, :])

        x = torch.stack(xs, dim=0)
        values, _ = torch.max(x, dim=0)
        return values


class STSGCL(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        window_size: int,
        input_channels: int,
        output_channels: int,
        filters: List[int],
        activation: Literal["relu", "glu"],
        temporal_emb: bool = True,
        spatial_emb: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._filters = filters
        self._window_size = window_size
        self._output_channels = output_channels
        self._input_channels = input_channels
        self._num_nodes = num_nodes
        self._activation = activation
        self._temporal_emb = temporal_emb
        self._spatial_emb = spatial_emb

        self._positional_embedding = PositionalEmbedding(
            window_size, num_nodes, input_channels, temporal_emb, spatial_emb
        )
        self._left_conv = nn.Conv2d(
            input_channels,
            filters[-1],
            kernel_size=(1, 2),
            stride=(1, 1),
            dilation=(1, 3),
        )
        self._right_conv = nn.Conv2d(
            input_channels,
            filters[-1],
            kernel_size=(1, 2),
            stride=(1, 1),
            dilation=(1, 3),
        )
        self._proj_conv = nn.Conv1d(input_channels, filters[0], kernel_size=1)
        self._stsgcm = STSGCM(num_nodes, filters, activation)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        x = self._positional_embedding(x)
        y = x.permute(0, 3, 2, 1)
        y = torch.sigmoid(self._left_conv(y)) * torch.tanh(self._right_conv(y))
        y = y.permute(0, 3, 2, 1)
        out = []
        for i in range(self._window_size - 3):
            t = x[:, i : i + 4, :, :]
            t = t.reshape(-1, 4 * self._num_nodes, self._input_channels)
            t = self._proj_conv(t.transpose(1, 2))
            t = t.permute(0, 2, 1)
            t = self._stsgcm(t, adjacency_matrix)
            out.append(t)
        out = torch.stack(out, dim=1)
        return self._dropout(out + y)


class OutputLayer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        window_size: int,
        horizon_size: int,
        input_channels: int,
        hidden_size: int,
        output_channels: int,
    ):
        super().__init__()
        self._num_nodes = num_nodes
        self._window_size = window_size
        self._horizon_size = horizon_size
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._hidden_size = hidden_size
        self._projection = nn.Sequential(
            nn.Linear(window_size * input_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon_size * output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, self._num_nodes, self._window_size * self._input_channels)
        x = self._projection(x)
        x = x.reshape(-1, self._num_nodes, self._horizon_size, self._output_channels)
        return x.permute(0, 2, 1, 3)

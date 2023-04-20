from typing import List, Literal

import torch
from torch import nn

from pbim_models.models.unsupervised.stfgnn.layer import OutputLayer, STSGCL


class STFGNN(nn.Module):
    def __init__(
        self,
        adjacency_matrix: torch.Tensor,
        num_nodes: int,
        window_size: int,
        horizon_size: int,
        input_channels: int,
        output_channels: int,
        filters: List[List[int]],
        activation: Literal["relu", "glu"],
        temporal_emb: bool = True,
        spatial_emb: bool = True,
        huber_delta: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("adjacency_matrix", adjacency_matrix)
        self._mask = nn.Parameter(
            torch.ones_like(adjacency_matrix, dtype=torch.float32)
        )
        self._num_nodes = num_nodes
        self._window_size = window_size
        self._horizon_size = horizon_size
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._filters = filters
        self._activation = activation
        self._temporal_emb = temporal_emb
        self._spatial_emb = spatial_emb
        self._huber_delta = huber_delta
        self._dropout = dropout

        self._blocks = nn.Sequential(*self._build_blocks())
        self._output_layer = OutputLayer(
            num_nodes,
            window_size - 3 * len(filters),
            horizon_size,
            filters[-1][-1],
            128,
            output_channels,
        )

    def _build_blocks(self) -> List[nn.Module]:
        blocks = []
        for i, block_filter in enumerate(self._filters):
            input_channels = (
                self._input_channels if i == 0 else self._filters[i - 1][-1]
            )
            blocks.append(
                STSGCL(
                    self._num_nodes,
                    self._window_size - 3 * i,
                    input_channels,
                    block_filter[-1],
                    block_filter,
                    self._activation,
                    self._temporal_emb,
                    self._spatial_emb,
                    self._dropout,
                )
            )
        return blocks

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mae = torch.abs(y_hat - y)
        loss = torch.where(
            mae < self._huber_delta,
            0.5 * mae**2,
            self._huber_delta * (mae - 0.5 * self._huber_delta),
        )
        return loss.sum(dim=-1).sum(dim=-1).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._blocks(x, self.adjacency_matrix * self._mask)
        x = self._output_layer(x)
        return x

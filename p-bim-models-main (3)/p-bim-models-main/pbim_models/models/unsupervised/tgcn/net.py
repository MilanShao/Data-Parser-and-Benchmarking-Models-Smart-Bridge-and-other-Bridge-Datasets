import torch
from torch import nn

from pbim_models.models.unsupervised.tgcn.layer import TGCN


class TGCNModel(nn.Module):
    def __init__(
        self,
        adjacency_matrix: torch.Tensor,
        input_size: int,
        hidden_size: int,
        horizon_size: int,
        input_channels: int,
        output_channels: int,
        dropout: float,
    ):
        super().__init__()
        self._horizon_size = horizon_size
        self._input_size = input_size
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._adjacency_matrix = adjacency_matrix
        self.model = TGCN(
            adjacency_matrix, input_size, hidden_size, input_channels, dropout
        )
        self.linear_out = nn.Linear(input_size * input_channels, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, seq_len, _ = x.shape
        x = self.model(x)
        x = x.reshape(batch_size, num_nodes, seq_len, -1)
        x = self.linear_out(x)
        return x[:, -self._horizon_size :, :]

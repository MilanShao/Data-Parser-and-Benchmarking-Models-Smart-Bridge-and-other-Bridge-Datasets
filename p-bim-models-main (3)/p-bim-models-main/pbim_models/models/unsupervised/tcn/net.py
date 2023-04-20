from typing import List

import torch
from torch import nn

from pbim_models.models.unsupervised.tcn.layer import TemporalConvNet


class TCN(nn.Module):
    def __init__(
        self,
        input_size: int,
        input_channels: int,
        output_size: int,
        num_channels: List[int],
        kernel_size: int,
        dropout: float,
        horizon: int,
    ):

        super().__init__()
        self.horizon = horizon
        self.tcn = TemporalConvNet(
            input_size * input_channels, num_channels, kernel_size, dropout=dropout
        )
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        batch_size, seq_len, _, _ = x.shape
        output = self.tcn(x.reshape(batch_size, seq_len, -1).transpose(1, 2)).transpose(
            1, 2
        )
        output = self.linear(output)
        return output.unsqueeze(-1)

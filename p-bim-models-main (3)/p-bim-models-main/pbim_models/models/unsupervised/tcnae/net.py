from typing import List

import torch
from torch import nn

from pbim_models.models.unsupervised.tcnae.layer import TemporalConvNet


class TCNAE(nn.Module):
    def __init__(
        self,
        input_size: int,
        input_channels: int,
        output_channels: int,
        num_channels: List[int],
        conv_output_size: int,
        hidden_channels: int,
        avg_pool_size: int,
        kernel_size: int,
        dropout: float,
        horizon: int,
    ):

        super().__init__()
        self.horizon = horizon
        self.avg_pool_size = avg_pool_size
        self.output_channels = output_channels
        self.encoder = nn.Sequential(
            TemporalConvNet(
                input_size * input_channels,
                conv_output_size,
                num_channels,
                kernel_size,
                dropout=dropout,
            ),
            nn.Conv1d(len(num_channels) * conv_output_size, hidden_channels, 1),
            nn.AvgPool1d(kernel_size=avg_pool_size, stride=avg_pool_size),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=avg_pool_size),
            TemporalConvNet(
                hidden_channels,
                conv_output_size,
                num_channels[::-1],
                kernel_size,
                dropout=dropout,
            ),
            nn.Conv1d(
                len(num_channels) * conv_output_size, input_size * output_channels, 1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has dimension (B, T, N, C)
        # x needs to have dimension (N, C, T) in order to be passed into CNN
        batch_size, seq_len, nodes, channels = x.shape
        hidden = self.encoder(x.reshape(batch_size, seq_len, -1).transpose(1, 2))
        hidden = torch.tanh(hidden)
        dec = self.decoder(hidden).transpose(1, 2)
        return dec.reshape(batch_size, seq_len, nodes, self.output_channels)

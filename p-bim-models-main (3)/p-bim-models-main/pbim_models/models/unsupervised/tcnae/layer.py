import torch
from typing import List

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout=0.2,
    ):
        super(TemporalBlock, self).__init__()
        self.dilated_conv = weight_norm(
            nn.Conv1d(
                input_size,
                intermediate_size,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.projection_conv = weight_norm(
            nn.Conv1d(
                intermediate_size,
                output_size,
                1,
            )
        )
        self.residual_conv = (
            weight_norm(
                nn.Conv1d(
                    input_size,
                    output_size,
                    1,
                )
            )
            if input_size != output_size
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.chomp = Chomp1d(padding)
        self.init_weights()

    def init_weights(self):
        self.dilated_conv.weight.data.normal_(0, 0.01)
        self.projection_conv.weight.data.normal_(0, 0.01)
        if self.residual_conv:
            self.residual_conv.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = F.relu(self.chomp(self.dilated_conv(x)))
        o = self.dropout(o)
        o = self.projection_conv(o)
        if self.residual_conv:
            x = self.residual_conv(x)
        return F.relu(self.dropout(x + o))


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        conv_output_size: int,
        num_channels: List[int],
        kernel_size=2,
        dropout=0.2,
    ):
        super(TemporalConvNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(num_channels)):
            dilation_size = 2**i
            self.layers += [
                TemporalBlock(
                    input_size if i == 0 else conv_output_size,
                    num_channels[i],
                    conv_output_size,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = []
        for layer in self.layers:
            x = layer(x)
            residuals.append(x)
        return torch.cat(residuals, dim=1)

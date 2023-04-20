from typing import List, Optional

import torch
from torch import nn


class ConvolutionalGenerator(nn.Module):
    def __init__(
        self,
        layers: List[int],
        kernel_size: Optional[int],
        noise_channels: int,
        noise_dimension: int,
        noise_length: int,
        window_size: int,
        input_size: int,
        output_size: int,
        output_channels: int,
    ):
        super().__init__()
        self._layers = layers
        self._kernel_size = kernel_size
        self._noise_channels = noise_channels
        self._noise_dimension = noise_dimension
        self._noise_length = noise_length
        self._window_size = window_size
        self._input_size = input_size
        self._output_size = output_size
        self._output_channels = output_channels

        self._conv_layers = nn.Sequential(
            self._make_layer(1, self._layers[0]),
            *[
                self._make_layer(self._layers[i], self._layers[i + 1])
                for i in range(len(self._layers) - 1)
            ],
            self._make_layer(self._layers[-1], self._output_channels)
        )

        self._node_projection = nn.Conv1d(
            in_channels=self._input_size * self._noise_channels,
            out_channels=self._input_size,
            kernel_size=1,
        )

    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self._input_size * self._noise_channels, self._kernel_size),
                padding="same",
                padding_mode="circular",
            ),
            nn.BatchNorm2d(out_channels),
            nn.Upsample(scale_factor=(1, 2)),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x T x N x C
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 1, -1, self._noise_length)
        x = self._conv_layers(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, self._input_size * self._noise_channels, -1)
        x = self._node_projection(x)
        x = x.reshape(
            batch_size, self._window_size, self._output_size, self._output_channels
        )
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        layers: List[int],
        kernel_size: int,
        input_channels: int,
        window_size: int,
        input_size: int,
    ):
        super().__init__()
        self._layers = layers
        self._kernel_size = kernel_size
        self._input_channels = input_channels
        self._window_size = window_size
        self._input_size = input_size

        self._node_projection = nn.Linear(self._input_size, self._layers[0])
        self._convs = nn.Sequential(
            *[
                self._build_layer(in_channels, out_channels)
                for in_channels, out_channels in zip(layers[:-1], layers[1:])
            ]
        )
        self._linear = nn.Linear(
            in_features=self._layers[-1] * self._input_channels * self._window_size,
            out_features=1,
        )

    def _build_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(self._kernel_size, self._input_channels),
                padding="same",
            ),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x T x N x C
        x = torch.transpose(x, 2, 3)  # B x T x N x C -> B x T x C x N
        x = self._node_projection(x)
        x = torch.nn.functional.leaky_relu(x)
        x = torch.permute(x, (0, 3, 1, 2))  # B x T x C x N -> B x N x T x C
        x = self._convs(x)
        x = x.reshape(x.shape[0], -1)
        return self._linear(x)

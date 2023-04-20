from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        window_size: int,
        input_size: int,
        input_channels: int,
        output_size: int,
        output_channels: int,
        hidden_sizes: List[int],
        dropout: float,
    ):
        super().__init__()
        self.window_size = window_size
        self.horizon_size = window_size
        self.input_size = input_size
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_channels = output_channels
        self.hidden_sizes = hidden_sizes

        self.encoder = nn.Sequential(
            nn.Linear(
                self.window_size * self.input_size * self.input_channels,
                hidden_sizes[0],
            ),
            nn.LeakyReLU(),
            *self._build_part(hidden_sizes),
        )
        self.decoder = nn.Sequential(
            *self._build_part(hidden_sizes[::-1]),
            nn.Linear(
                hidden_sizes[0],
                self.horizon_size * self.output_size * self.output_channels,
            ),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_part(hidden_layer: List[int]):
        for i in range(len(hidden_layer) - 1):
            yield nn.Linear(hidden_layer[i], hidden_layer[i + 1])
            yield nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.decoder(x)
        x = x.view(x.size(0), self.horizon_size, self.output_size, self.output_channels)
        return x

from typing import List

import torch
from torch import nn


class SupervisedLSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        input_channels: int,
        hidden_size: int,
        lstm_layers: int,
        classifier_layers: List[int],
        window_size: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.bidirectional = bidirectional
        self.encoder = nn.LSTM(
            input_size=input_size * input_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.lstm_layers = lstm_layers
        self._classifier = nn.Sequential(
            nn.Linear(
                hidden_size * (2 if bidirectional else 1) * lstm_layers,
                classifier_layers[0],
            ),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(classifier_layers[i], classifier_layers[i + 1]), nn.ReLU()
                )
                for i in range(len(classifier_layers) - 1)
            ],
            nn.Linear(classifier_layers[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.window_size, -1)
        _, (h, _) = self.encoder(x)
        h = h.permute(1, 0, 2).reshape(batch_size, -1)
        x = self._classifier(h)
        return x

from typing import List, Tuple

import torch
from torch import nn


class PointNet(nn.Module):
    def __init__(self, input_features: int, layers: List[int]):
        super().__init__()
        self._layer_sizes = [input_features] + layers
        self._layers = nn.ModuleList(
            [
                self._make_layer(self._layer_sizes[i], self._layer_sizes[i + 1])
                for i in range(len(self._layer_sizes) - 1)
            ]
        )

    @staticmethod
    def _make_layer(
        in_features: int, out_features: int, kernel_size: int = 1
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_pts = x.size()[2]
        x = self._layers[0](x)

        point_features = x
        for layer in self._layers[1:]:
            x = layer(x)
        x = torch.nn.functional.max_pool1d(x, n_pts)
        global_features = x.view(-1, self._layer_sizes[-1])
        return global_features, point_features

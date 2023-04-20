from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from pbim_models.models.unsupervised.stgcn.layer import STConvBlock, OutputBlock


def compute_symmetric_normalized_laplacian(
    adjacency_matrix: torch.Tensor,
) -> torch.Tensor:
    num_nodes = adjacency_matrix.shape[0]

    # Symmetrizing the adjacency matrix
    sym_adjacency_matrix = 0.5 * (adjacency_matrix + adjacency_matrix.T) + torch.eye(
        num_nodes, device=adjacency_matrix.device
    )

    row_sum = sym_adjacency_matrix.sum(axis=1)
    row_sum_inv_sqrt = row_sum**-0.5
    row_sum_inv_sqrt = torch.where(
        torch.isinf(row_sum_inv_sqrt),
        torch.zeros_like(row_sum_inv_sqrt),
        row_sum_inv_sqrt,
    )
    deg_inv_sqrt = torch.diag(row_sum_inv_sqrt)
    # A_{sym} = D^{-0.5} * A * D^{-0.5}
    sym_norm_adj = deg_inv_sqrt @ sym_adjacency_matrix @ deg_inv_sqrt
    sym_norm_lap = torch.eye(num_nodes, device=adjacency_matrix.device) - sym_norm_adj
    return sym_norm_lap


def compute_cheb_net_laplacian(laplacian: torch.Tensor) -> torch.Tensor:
    eigenvalues = torch.linalg.eigvals(laplacian).real
    largest_value = torch.max(eigenvalues).item()
    identity = torch.eye(laplacian.shape[0], device=laplacian.device)
    # If the laplacian is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if largest_value >= 2:
        cheb_net_laplacian = laplacian - identity
    else:
        cheb_net_laplacian = 2 * laplacian / largest_value - identity

    return cheb_net_laplacian


def compute_blocks(
    window_size: int,
    horizon_size: int,
    kernel_size_tc: int,
    num_blocks: int,
    input_channels: int,
) -> List[List[int]]:
    output_kernel_size = window_size - (kernel_size_tc - 1) * 2 * num_blocks

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = [[input_channels]]
    for l in range(num_blocks):
        blocks.append([64, 16, 64])
    if output_kernel_size == 0:
        blocks.append([128])
    elif output_kernel_size > 0:
        blocks.append([128, 128])
    blocks.append([horizon_size])

    return blocks


class STGCN(nn.Module):

    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(
        self,
        kernel_size_tc: int,
        kernel_size_gc: int,
        dropout: float,
        adjacency_matrix: torch.Tensor,
        bias: bool,
        num_blocks: int,
        window_size: int,
        horizon_size: int,
        num_nodes: int,
        input_channels_per_node: int,
    ):
        super(STGCN, self).__init__()
        blocks = compute_blocks(
            window_size,
            horizon_size,
            kernel_size_tc,
            num_blocks,
            input_channels_per_node,
        )
        self.register_buffer("adjacency_matrix", adjacency_matrix)
        laplacian = compute_cheb_net_laplacian(
            compute_symmetric_normalized_laplacian(self.adjacency_matrix)
        )
        modules = []
        for i in range(len(blocks) - 3):
            modules.append(
                STConvBlock(
                    kernel_size_tc,
                    kernel_size_gc,
                    num_nodes,
                    blocks[i][-1],
                    blocks[i + 1],
                    laplacian,
                    dropout,
                    bias,
                )
            )
        self.st_blocks = nn.Sequential(*modules)
        self.kernel_size_output = window_size - (len(blocks) - 3) * 2 * (
            kernel_size_tc - 1
        )
        if self.kernel_size_output > 1:
            self.output = OutputBlock(
                self.kernel_size_output,
                blocks[-3][-1],
                blocks[-2],
                blocks[-1][0],
                num_nodes,
                dropout,
                bias,
            )
        else:
            self.fc1 = nn.Linear(
                in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=bias
            )
            self.fc2 = nn.Linear(
                in_features=blocks[-2][0], out_features=blocks[-1][0], bias=bias
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.st_blocks(x)
        if self.kernel_size_output > 1:
            x = self.output(x)
        else:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = F.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        return x

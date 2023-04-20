import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(1, 1)
        )

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat(
                [
                    x,
                    torch.zeros(
                        [batch_size, self.c_out - self.c_in, timestep, n_vertex]
                    ).to(x),
                ],
                dim=1,
            )
        else:
            x = x

        return x


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        enable_padding=False,
        dilation=1,
        groups=1,
        bias=True,
    ):
        if enable_padding == True:
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]

        return result


class CausalConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        enable_padding=False,
        dilation=1,
        groups=1,
        bias=True,
    ):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding:
            self.__padding = [
                int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))
            ]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        if self.__padding != 0:
            x = F.pad(x, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(x)

        return result


class TemporalConvLayer(nn.Module):

    # Temporal Convolution Layer (GLU)
    #
    #        |--------------------------------| * Residual Connection *
    #        |                                |
    #        |    |--->--- CasualConv2d ----- + -------|
    # -------|----|                                   ⊙ ------>
    #             |--->--- CasualConv2d --- Sigmoid ---|
    #

    # param x: tensor, [bs, c_in, ts, n_vertex]

    def __init__(self, kernel_size: int, c_in: int, c_out: int, num_nodes: int):
        super(TemporalConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.c_in = c_in
        self.c_out = c_out
        self.num_nodes = num_nodes
        self.align = Align(c_in, c_out)
        self.causal_conv = CausalConv2d(
            in_channels=c_in,
            out_channels=2 * c_out,
            kernel_size=(kernel_size, 1),
            enable_padding=False,
            dilation=1,
        )

    def forward(self, x):
        # B x C x T x N
        x_in = self.align(x)[:, :, self.kernel_size - 1 :, :]
        x_causal_conv = self.causal_conv(x)
        x_p = x_causal_conv[:, : self.c_out, :, :]
        x_q = x_causal_conv[:, -self.c_out :, :, :]
        x = torch.mul((x_p + x_in), torch.sigmoid(x_q))
        return x


class ChebGraphConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, gso, bias: bool):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(kernel_size, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # bs, c_in, ts, n_vertex = x.shape
        x = torch.permute(x, (0, 2, 3, 1))

        if self.kernel_size - 1 < 0:
            raise ValueError(
                f"ERROR: the graph convolution kernel size has to be a positive integer, but received {self.kernel_size}."
            )
        elif self.kernel_size - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.kernel_size - 1 == 1:
            x_0 = x
            x_1 = torch.einsum("hi,btij->bthj", self.gso, x)
            x_list = [x_0, x_1]
        else:
            x_0 = x
            x_1 = torch.einsum("hi,btij->bthj", self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.kernel_size):
                x_list.append(
                    torch.einsum("hi,btij->bthj", 2 * self.gso, x_list[k - 1])
                    - x_list[k - 2]
                )

        x = torch.stack(x_list, dim=2)

        cheb_graph_conv = torch.einsum("btkhi,kij->bthj", x, self.weight)

        if self.bias is not None:
            cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)

        return cheb_graph_conv


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso, bias):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # bs, c_in, ts, n_vertex = x.shape
        x = torch.permute(x, (0, 2, 3, 1))

        first_mul = torch.einsum("hi,btij->bthj", self.gso, x)
        second_mul = torch.einsum("bthi,ij->bthj", first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul

        return graph_conv


class GraphConvLayer(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, gso, bias: bool = True):
        super(GraphConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.kernel_size = kernel_size
        self.conv = ChebGraphConv(c_out, c_out, kernel_size, gso, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gc_in = self.align(x)
        x_gc = self.conv(x_gc_in)
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)
        return x_gc_out


class STConvBlock(nn.Module):
    # STConv Block contains 'TGTND' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv or GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    def __init__(
        self,
        kernel_size_tc: int,
        kernel_size_gc: int,
        num_nodes: int,
        last_block_channel,
        channels: List[int],
        gso,
        dropout: float,
        bias: bool = True,
    ):
        super(STConvBlock, self).__init__()
        self.tc_1 = TemporalConvLayer(
            kernel_size_tc,
            last_block_channel,
            channels[0],
            num_nodes,
        )
        self.gc = GraphConvLayer(channels[0], channels[1], kernel_size_gc, gso, bias)
        self.tc_2 = TemporalConvLayer(
            kernel_size_tc, channels[1], channels[2], num_nodes
        )
        self.tc2_ln = nn.LayerNorm([num_nodes, channels[2]])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tc_1(x)
        x = self.gc(x)
        x = torch.relu(x)
        x = self.tc_2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)

        return x


class OutputBlock(nn.Module):
    # Output block contains 'TNFF' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(
        self,
        kernel_size: int,
        last_block_channel: int,
        channels: List[int],
        end_channel: int,
        num_nodes: int,
        dropout: float,
        bias: bool = True,
    ):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(
            kernel_size, last_block_channel, channels[0], num_nodes
        )
        self.fc1 = nn.Linear(
            in_features=channels[0], out_features=channels[1], bias=bias
        )
        self.fc2 = nn.Linear(
            in_features=channels[1], out_features=end_channel, bias=bias
        )
        self.tc1_ln = nn.LayerNorm([num_nodes, channels[0]])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x).permute(0, 3, 2, 1)
        return x

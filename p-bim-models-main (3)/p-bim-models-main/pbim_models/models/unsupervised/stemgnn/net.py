import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pbim_models.models.unsupervised.stemgnn.layer import StockBlockLayer


class StemGNN(pl.LightningModule):
    def __init__(
        self,
        node_count: int,
        stack_cnt: int,
        window_size: int,
        multi_layer,
        dropout_rate=0.2,
        leaky_rate=0.2,
        decay_rate=0.5,
    ):
        super(StemGNN, self).__init__()
        self.node_count = node_count
        self.stack_cnt = stack_cnt
        self.alpha = leaky_rate
        self.window_size = window_size
        self.weight_key = nn.Parameter(torch.zeros(size=(self.node_count, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.node_count, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.window_size, self.node_count)
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [
                StockBlockLayer(
                    self.window_size, self.node_count, self.multi_layer, stack_cnt=i
                )
                for i in range(self.stack_cnt)
            ]
        )
        self.fc = nn.Sequential(
            nn.Linear(self.window_size, self.window_size),
            nn.LeakyReLU(),
            nn.Linear(self.window_size, self.window_size),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.decay_rate = decay_rate

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(
                graph.size(0), device=graph.device, dtype=graph.dtype
            ) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros(
            [1, N, N], device=laplacian.device, dtype=torch.float
        )
        second_laplacian = laplacian
        third_laplacian = (
            2 * torch.matmul(laplacian, second_laplacian)
        ) - first_laplacian
        forth_laplacian = (
            2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        )
        multi_order_laplacian = torch.cat(
            [first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0
        )
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(
            diagonal_degree_hat, torch.matmul(degree_l - attention, diagonal_degree_hat)
        )
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x: torch.Tensor):
        assert x.size(3) == 1, "does not support multidimensional nodes"
        x = x.squeeze(3)
        mul_L, attention = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        result = []
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)
        forecast = result[0] + result[1]
        forecast = self.fc(forecast)
        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1)
        else:
            return forecast.permute(0, 2, 1).contiguous().unsqueeze(-1)

import torch
import torch.nn as nn


def calculate_laplacian_with_self_loop(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix + torch.eye(matrix.size(0), device=matrix.device)
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian


class GCN(nn.Module):
    def __init__(self, adj, input_dim: int, output_dim: int):
        super(GCN, self).__init__()
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim  # seq_len for prediction
        self._output_dim = output_dim  # hidden_dim for prediction
        self.weights = nn.Parameter(
            torch.FloatTensor(self._input_dim, self._output_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def forward(self, inputs):
        # (batch_size, seq_len, num_nodes)
        batch_size = inputs.shape[0]
        # (num_nodes, batch_size, seq_len)
        inputs = inputs.transpose(0, 2).transpose(1, 2)
        # (num_nodes, batch_size * seq_len)
        inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))
        # AX (num_nodes, batch_size * seq_len)
        ax = self.laplacian @ inputs
        # (num_nodes, batch_size, seq_len)
        ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
        # (num_nodes * batch_size, seq_len)
        ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
        # act(AXW) (num_nodes * batch_size, output_dim)
        outputs = torch.tanh(ax @ self.weights)
        # (num_nodes, batch_size, output_dim)
        outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
        # (batch_size, num_nodes, output_dim)
        outputs = outputs.transpose(0, 1)
        return outputs


class TGCNGraphConvolution(nn.Module):
    def __init__(
        self,
        adj,
        num_gru_units: int,
        output_dim: int,
        input_channels: int,
        output_channels: int,
        bias: float = 0.0,
    ):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self._input_channels = input_channels
        self._output_channels = output_channels
        self.register_buffer("laplacian", calculate_laplacian_with_self_loop(adj))
        self.weights = nn.Parameter(
            torch.FloatTensor(
                self._num_gru_units + 1,
                self._output_dim,
                input_channels,
                output_channels,
            )
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim, output_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward_v2(self, x: torch.Tensor, h: torch.Tensor):
        batch_size, num_nodes, channels = x.shape
        assert channels == self._input_channels
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1, channels)
        x = x.reshape((batch_size, num_nodes, 1, self._input_channels))
        # hidden_state (batch_size, num_nodes, num_gru_units, channels)
        h = h.reshape(
            (batch_size, num_nodes, self._num_gru_units, self._input_channels)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1, channels)
        concatenation = torch.cat((x, h), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size, channels)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size * channels)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size * self._input_channels)
        )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size * channels)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size, channels)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size, self._input_channels)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1, channels)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1, channels)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1, self._input_channels)
        )
        outputs = torch.zeros(
            batch_size, num_nodes * self._output_dim, self._output_channels
        ).type_as(x)
        for i in range(self._output_channels):
            for j in range(self._input_channels):
                # A[x, h]W (batch_size * num_nodes, num_gru_units + 1, output_dim)
                conv = a_times_concat[:, :, j] @ self.weights[:, :, j, i]
                conv = conv.reshape((batch_size, num_nodes * self._output_dim))
                outputs[:, :, i] += conv
        outputs = outputs.reshape(
            (batch_size * num_nodes, self._output_dim, self._output_channels)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = outputs + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape(
            (batch_size, num_nodes * self._output_dim, self._output_channels)
        )
        return outputs

    def forward(self, inputs, hidden_state):
        return self.forward_v2(inputs, hidden_state)
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs


class TGCNCell(nn.Module):
    def __init__(
        self, adj: torch.Tensor, input_dim: int, hidden_dim: int, channels: int
    ):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._channels = channels
        self.register_buffer("adj", adj)
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj,
            self._hidden_dim,
            self._hidden_dim * 2,
            channels,
            channels,
            bias=1.0,
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim, channels, channels
        )

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes, _ = inputs.shape
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state


class TGCN(nn.Module):
    def __init__(
        self,
        adj: torch.Tensor,
        hidden_dim: int,
        horizon_size: int,
        channels: int,
        dropout: float = 0.1,
    ):
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self._horizon_size = horizon_size
        self._channels = channels
        self._dropout = nn.Dropout(dropout)
        self.register_buffer("adj", adj)
        self._tgcn_cell = TGCNCell(
            self.adj, self._input_dim, self._hidden_dim, channels
        )
        self._projection = nn.Linear(
            self._hidden_dim * self._input_dim, self._horizon_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x T x N x C
        batch_size, seq_len, num_nodes, _ = x.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(
            batch_size, num_nodes * self._hidden_dim, self._channels
        ).type_as(x)
        outputs = []
        for i in range(seq_len):
            output, hidden_state = self._tgcn_cell(x[:, i, :, :], hidden_state)
            hidden_state = self._dropout(hidden_state)
            outputs += [
                output.reshape(
                    (batch_size, num_nodes, self._hidden_dim, self._channels)
                )
            ]
        return torch.stack(outputs, dim=1)

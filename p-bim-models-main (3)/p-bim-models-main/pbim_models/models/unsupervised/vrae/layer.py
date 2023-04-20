import torch
from torch import nn


class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU
    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param num_layers: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """

    def __init__(
        self,
        number_of_features: int,
        hidden_dim: int,
        num_layers: int,
        latent_dim: int,
        dropout: float,
        block="LSTM",
    ):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self._block = block

        if block == "LSTM":
            self.model = nn.LSTM(
                self.number_of_features,
                self.hidden_dim,
                self.num_layers,
                dropout=dropout,
                batch_first=True,
            )
        elif block == "GRU":
            self.model = nn.GRU(
                self.number_of_features,
                self.hidden_dim,
                self.num_layers,
                dropout=dropout,
                batch_first=True,
            )
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """
        if self._block == "LSTM":
            _, (h_end, _) = self.model(x)
        elif self._block == "GRU":
            _, h_end = self.model(x)
        else:
            raise NotImplementedError

        h_end = h_end[-1, :, :]
        return h_end


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector
    :param hidden_dim: hidden size of the encoder
    :param latent_dim: latent vector length
    """

    def __init__(self, hidden_dim: int, latent_dim: int):
        super(Lambda, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.hidden_to_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.hidden_to_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output: torch.Tensor) -> torch.Tensor:
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance
        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean


class Decoder(nn.Module):
    """Converts latent vector into output
    :param sequence_length: length of the input sequence
    :param hidden_dim: hidden size of the RNN
    :param num_layers: number of layers in RNN
    :param latent_dim: latent vector length
    :param output_dim: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    """

    def __init__(
        self,
        batch_size: int,
        sequence_length: int,
        hidden_dim: int,
        num_layers: int,
        latent_dim: int,
        output_dim: int,
        block="LSTM",
    ):

        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self._block = block

        if block == "LSTM":
            self.model = nn.LSTM(1, self.hidden_dim, self.num_layers, batch_first=True)
        elif block == "GRU":
            self.model = nn.GRU(1, self.hidden_dim, self.num_layers, batch_first=True)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.hidden_to_output = nn.Linear(self.hidden_dim, self.output_dim)

        self.decoder_inputs = nn.Parameter(
            torch.zeros(self.batch_size, self.sequence_length, 1)
        )
        self.c_0 = (
            nn.Parameter(
                torch.zeros(
                    self.num_layers,
                    self.batch_size,
                    self.hidden_dim,
                )
            )
            if block == "LSTM"
            else None
        )

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Converts latent to hidden to output
        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)

        if self._block == "LSTM":
            h_0 = torch.stack([h_state for _ in range(self.num_layers)])
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif self._block == "GRU":
            h_0 = torch.stack([h_state for _ in range(self.num_layers)])
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)
        return out

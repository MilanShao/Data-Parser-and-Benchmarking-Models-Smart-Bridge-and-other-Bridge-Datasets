import pytorch_lightning as pl
import torch
from torch import nn

from pbim_models.models.unsupervised.vrae.layer import Encoder, Lambda, Decoder


class VRAE(pl.LightningModule):
    """Variational recurrent auto-encoder. This module is used for dimensionality reduction of timeseries
    :param window_size: length of the input sequence
    :param hidden_dim:  hidden size of the RNN
    :param num_layers: number of layers in RNN
    :param latent_dim: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param block: GRU/LSTM to be used as a basic building block
    :param dropout_rate: The probability of a node being dropped-out
    """

    def __init__(
        self,
        batch_size: int,
        window_size: int,
        input_size: int,
        input_channels: int,
        output_size: int,
        loss: nn.Module,
        hidden_dim=32,
        num_layers=2,
        latent_dim=16,
        block="LSTM",
        dropout_rate=0.1,
    ):

        super(VRAE, self).__init__()
        self.encoder = Encoder(
            number_of_features=input_size * input_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            latent_dim=latent_dim,
            dropout=dropout_rate,
            block=block,
        )

        self.lmbd = Lambda(hidden_dim=hidden_dim, latent_dim=latent_dim)

        self.decoder = Decoder(
            batch_size=batch_size,
            sequence_length=window_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            latent_dim=latent_dim,
            output_dim=output_size,
            block=block,
        )

        self.sequence_length = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.loss = loss
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        batch_size, sequence_length, _, _ = x.shape
        x = x.view(batch_size, sequence_length, -1)
        cell_output = self.encoder(x)
        latent = self.lmbd(self.dropout(cell_output))
        y_hat = self.decoder(latent)

        return y_hat.unsqueeze(-1)

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor):
        """
        Compute the loss given output x decoded, input x and the specified loss function
        :param y_hat: output of the decoder
        :param y: target tensor
        """
        latent_mean, latent_log_var = self.lmbd.latent_mean, self.lmbd.latent_logvar

        kl_loss = -0.5 * torch.mean(
            1 + latent_log_var - latent_mean.pow(2) - latent_log_var.exp()
        )
        reconstruction_loss = self.loss(y_hat, y)

        return kl_loss + reconstruction_loss

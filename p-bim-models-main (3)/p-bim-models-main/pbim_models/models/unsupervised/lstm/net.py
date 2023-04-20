import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        input_channels: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int,
        output_size: int,
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
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.num_layers = num_layers
        self.hidden_to_latent = nn.Linear(
            2 * hidden_size if bidirectional else hidden_size, latent_size
        )
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.decoder_input = nn.Parameter(torch.zeros(1, window_size, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.window_size, -1)
        _, (h, _) = self.encoder(x)
        h = (
            torch.cat(
                [h[0 : self.num_layers], h[self.num_layers : 2 * self.num_layers]],
                dim=-1,
            )
            if self.bidirectional
            else h
        )
        h = self.latent_to_hidden(torch.relu(self.hidden_to_latent(h)))
        decoder_input = self.decoder_input.repeat(batch_size, 1, 1)
        decoded, _ = self.decoder(decoder_input, (h, torch.zeros_like(h)))
        x = self.fc(decoded).unsqueeze(-1)
        return x

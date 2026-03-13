import torch
import torch.nn as nn

from src.config import INPUT_CHANNELS, INPUT_LENGTH, NUM_CLASSES


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]


class TransformerBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_CHANNELS,
        seq_len: int = INPUT_LENGTH,
        num_classes: int = NUM_CLASSES,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim) = (batch, 128, 8)
        x = self.input_projection(x)          # (batch, seq_len, d_model)
        x = self.positional_encoding(x)       # (batch, seq_len, d_model)
        x = self.encoder(x)                   # (batch, seq_len, d_model)
        x = x.mean(dim=1)                     # mean pooling over time
        x = self.classifier(x)                # (batch, num_classes)
        return x
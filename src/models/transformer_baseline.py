"""
Lightweight Transformer encoder classifier for MAFAULDA fault diagnosis.

This is the project's primary model family (config describes F_1layer, the
FPGA-deployed variant). It is kept intentionally small so it can later be
quantized and mapped to the target FPGA:

    input projection -> positional encoding -> N encoder blocks
    -> mean pooling over time -> classification head -> class logits

Input tensor shape is [batch, seq_len, input_dim] (e.g. [B, 1024, 8]) and the
output is raw logits of shape [batch, num_classes]. Softmax is applied by the
loss function during training or separately at inference, not inside the model.
"""

import math

import torch
import torch.nn as nn

from src.config import (
    INPUT_CHANNELS,
    INPUT_LENGTH,
    NUM_CLASSES,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT,
)


class PositionalEncoding(nn.Module):
    """
    Fixed (non-learned) sinusoidal positional encoding.

    A Transformer is order-agnostic on its own, so each time position is given a
    unique signature that is added to the token embedding. Even and odd feature
    dimensions use sine and cosine at geometrically spaced frequencies, matching
    the original "Attention Is All You Need" formulation.

    The table is registered as a buffer (not a trainable parameter): it is fixed
    but still saved and moved with the model.

    Parameters
    ----------
    d_model : int
        Token/feature dimension the encoding is added to.
    max_len : int
        Maximum sequence length the table is precomputed for.
    dropout : float, default=0.1
        Dropout applied after adding the positional encoding.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x`` ([B, seq_len, d_model]); shape is unchanged."""
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    One post-norm Transformer encoder block.

    Two sub-layers, each wrapped in a residual connection followed by
    LayerNorm (i.e. ``norm(x + sublayer(x))``):

    1. Multi-head self-attention, letting each time step attend to all others.
    2. Position-wise feed-forward network (Linear -> GELU -> Dropout -> Linear).

    Input and output shapes are both [B, seq_len, d_model].

    Parameters
    ----------
    d_model : int
        Token/feature dimension.
    nhead : int
        Number of attention heads (must divide ``d_model``).
    dim_feedforward : int
        Hidden size of the feed-forward network.
    dropout : float
        Dropout rate used in attention and the feed-forward network.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(d_model)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run self-attention then the feed-forward network, each with residual + LayerNorm."""
        attention_output, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.attention_norm(x + self.attention_dropout(attention_output))

        feedforward_output = self.feedforward(x)
        x = self.feedforward_norm(x + self.feedforward_dropout(feedforward_output))
        return x


class TransformerBaseline(nn.Module):
    """
    Lightweight Transformer encoder classifier (the project's main model).

    Pipeline: project the raw channels to ``d_model``, add positional encoding,
    pass through ``num_layers`` encoder blocks, mean-pool over time, then a small
    classification head produces one logit per class. Mean pooling and raw-logit
    output are chosen to keep quantization and the FPGA mapping simple.

    Defaults are read from ``src.config`` and describe F_1layer (the deployed
    model). Shapes: input [B, seq_len, input_dim] -> output [B, num_classes].

    Parameters
    ----------
    input_dim : int
        Number of input channels per time step.
    seq_len : int
        Window length (number of time steps); sets the positional-encoding size.
    num_classes : int
        Number of output classes.
    d_model : int
        Token/feature dimension inside the encoder.
    nhead : int
        Number of attention heads (must divide ``d_model``).
    num_layers : int
        Number of stacked encoder blocks.
    dim_feedforward : int
        Hidden size of each block's feed-forward network.
    dropout : float
        Dropout rate used throughout the model.

    Raises
    ------
    ValueError
        If ``d_model`` is not divisible by ``nhead``.
    """

    def __init__(
        self,
        input_dim: int = INPUT_CHANNELS,
        seq_len: int = INPUT_LENGTH,
        num_classes: int = NUM_CLASSES,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = TRANSFORMER_DROPOUT,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=seq_len,
            dropout=dropout,
        )
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify a batch of windows.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape [batch, seq_len, input_dim].

        Returns
        -------
        torch.Tensor
            Raw class logits of shape [batch, num_classes].
        """
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        x = x.mean(dim=1)
        return self.classifier(x)

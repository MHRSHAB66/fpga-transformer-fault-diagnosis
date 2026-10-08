"""
Lightweight Transformer encoder classifier for MAFAULDA fault diagnosis.

Main pipeline:

    raw input
        -> input projection
        -> positional encoding
        -> N Transformer encoder blocks
        -> mean pooling over time
        -> classification head
        -> raw class logits

Expected input:
    [batch_size, seq_len, input_dim]
    e.g. [B, 1024, 8]

Output:
    [batch_size, num_classes]

The model intentionally remains lightweight to simplify later quantization
and FPGA deployment.
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


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.

    Transformer attention itself does not know the order of samples in the
    sequence. Therefore, each time position receives a unique d_model-
    dimensional positional signature.

    For the current defaults (d_model = 32 and max_len = 1024):

        pe shape before batch dimension:
            [1024, 32]

        pe shape after unsqueeze:
            [1, 1024, 32]

    Even dimensions use sine.
    Odd dimensions use cosine.

    The positional encoding is registered as a buffer:
        - it belongs to the model
        - it moves with the model to CPU/GPU
        - it is saved in state_dict
        - it is NOT trainable
    """

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        if max_len <= 0:
            raise ValueError("max_len must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        # ----------------------------------------------------
        # Position indices
        #
        # [0, 1, 2, ..., max_len - 1]
        #
        # shape:
        #     [max_len]
        #
        # after unsqueeze(1):
        #     [max_len, 1]
        #
        # Example:
        #
        # [[0],
        #  [1],
        #  [2],
        #  ...]
        # ----------------------------------------------------
        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        # ----------------------------------------------------
        # Frequencies used by sine/cosine.
        #
        # For d_model = 32:
        #
        # arange:
        #     [0, 2, 4, ..., 30]
        #
        # therefore 16 frequency values are generated.
        #
        # Different feature dimensions use different
        # frequencies so positions receive distinct patterns.
        # ----------------------------------------------------
        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        # ----------------------------------------------------
        # Positional encoding table
        #
        # shape:
        #     [max_len, d_model]
        #
        # With the current defaults this is [1024, 32], but
        # both dimensions follow the constructor arguments.
        # ----------------------------------------------------
        pe = torch.zeros(
            max_len,
            d_model,
            dtype=torch.float32,
        )

        # Even feature dimensions:
        #
        # 0, 2, 4, ...
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd feature dimensions:
        #
        # 1, 3, 5, ...
        #
        # The slice on div_term also makes this safe if
        # d_model happens to be odd.
        num_odd_features = pe[:, 1::2].shape[1]

        pe[:, 1::2] = torch.cos(
            position * div_term[:num_odd_features]
        )

        # Regularization applied after:
        #
        # projected_input + positional_encoding
        self.dropout = nn.Dropout(dropout)

        # Add batch dimension:
        #
        # [max_len, d_model]
        #        ->
        # [1, max_len, d_model]
        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional information to the projected input.

        Input:
            [B, seq_len, d_model]

        Output:
            [B, seq_len, d_model]
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected a 3-D tensor [B, seq_len, d_model], "
                f"but got shape {tuple(x.shape)}."
            )

        seq_len = x.size(1)

        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds "
                f"maximum positional encoding length ({self.pe.size(1)})."
            )

        # Only use PE entries required for this sequence.
        positional_encoding = self.pe[:, :seq_len, :]

        # Keep dtype consistent with x.
        # Useful for mixed precision / later FPGA-related workflows.
        positional_encoding = positional_encoding.to(dtype=x.dtype)

        x = x + positional_encoding

        return self.dropout(x)


# ============================================================
# Transformer Encoder Block
# ============================================================

class TransformerEncoderBlock(nn.Module):
    """
    One post-norm Transformer encoder block.

    Architecture:

        x
        |
        v
        Multi-Head Self-Attention
        |
        v
        Dropout
        |
        + <----- residual x
        |
        v
        LayerNorm
        |
        v
        Feed-Forward Network
        |
        v
        Dropout
        |
        + <----- residual
        |
        v
        LayerNorm

    Input/output:
        [B, seq_len, d_model]

    For the current project configuration:

        d_model = 32
        nhead = 4
        head_dim = 8
        dim_feedforward = 64
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        if nhead <= 0:
            raise ValueError("nhead must be positive.")

        if d_model % nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead."
            )

        if dim_feedforward <= 0:
            raise ValueError(
                "dim_feedforward must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        # ----------------------------------------------------
        # Multi-Head Self-Attention
        #
        # For d_model = 32, nhead = 4:
        #
        # Q: 32 -> 32 -> 4 heads x 8
        # K: 32 -> 32 -> 4 heads x 8
        # V: 32 -> 32 -> 4 heads x 8
        #
        # Each Q/K/V dimension after projection can contain
        # information from all 32 input dimensions.
        # ----------------------------------------------------
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Dropout applied to the attention output before
        # residual addition.
        self.attention_dropout = nn.Dropout(dropout)

        # Post-norm:
        #
        # LayerNorm(x + Attention(x))
        self.attention_norm = nn.LayerNorm(d_model)

        # ----------------------------------------------------
        # Position-wise Feed-Forward Network
        #
        # For the current model:
        #
        # 32
        #  ↓
        # Linear
        #  ↓
        # 64
        #  ↓
        # GELU
        #  ↓
        # Dropout
        #  ↓
        # Linear
        #  ↓
        # 32
        #
        # This is independently applied to every time step,
        # using the same weights at all positions.
        # ----------------------------------------------------
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.feedforward_dropout = nn.Dropout(dropout)

        # Post-norm:
        #
        # LayerNorm(x + FeedForward(x))
        self.feedforward_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention followed by the feed-forward network.

        Shape remains:

            [B, seq_len, d_model]
        """

        # ----------------------------------------------------
        # Self-attention
        #
        # x is passed as query, key and value:
        #
        #     attention(x, x, x)
        #
        # therefore this is SELF-attention.
        #
        # Internally:
        #
        # Q = x W_Q
        # K = x W_K
        # V = x W_V
        #
        # Attention:
        #
        # softmax(Q K^T / sqrt(head_dim)) V
        # ----------------------------------------------------
        attention_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            need_weights=False,
        )

        # ----------------------------------------------------
        # Residual + post LayerNorm
        #
        # x_new =
        #     LayerNorm(
        #         x + Dropout(Attention(x))
        #     )
        # ----------------------------------------------------
        x = self.attention_norm(
            x
            + self.attention_dropout(attention_output)
        )

        # ----------------------------------------------------
        # Feed-forward network
        # ----------------------------------------------------
        feedforward_output = self.feedforward(x)

        # ----------------------------------------------------
        # Second residual + post LayerNorm
        # ----------------------------------------------------
        x = self.feedforward_norm(
            x
            + self.feedforward_dropout(feedforward_output)
        )

        return x


# ============================================================
# Complete Transformer Classifier
# ============================================================

class TransformerBaseline(nn.Module):
    """
    Lightweight Transformer classifier for MAFAULDA fault diagnosis.

    Architecture (current default dimensions shown as an example):

        Raw input
        [B, seq_len, input_dim]  (default: [B, 1024, 8])

              |
              v

        Input projection
        Linear(input_dim -> d_model)  (default: 8 -> 32)

              |
              v

        [B, seq_len, d_model]

              |
              v

        Positional Encoding

              |
              v

        N x TransformerEncoderBlock

              |
              v

        [B, seq_len, d_model]

              |
              v

        Mean Pooling over time

              |
              v

        [B, d_model]

              |
              v

        Classification Head

        LayerNorm(d_model)
        Linear(d_model -> d_model // 2)
        GELU
        Dropout
        Linear(d_model // 2 -> num_classes)

              |
              v

        [B, num_classes]

    Raw logits are returned.
    Softmax is NOT applied inside the model.
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

        # ----------------------------------------------------
        # Parameter validation
        # ----------------------------------------------------
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")

        if num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1."
            )

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        if nhead <= 0:
            raise ValueError("nhead must be positive.")

        if d_model % nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be at least 1."
            )

        if dim_feedforward <= 0:
            raise ValueError(
                "dim_feedforward must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        self.input_dim = input_dim
        self.max_seq_len = seq_len

        # ----------------------------------------------------
        # Raw sensor channels -> transformer representation
        #
        # Current MAFAULDA input:
        #
        # 0 tachometer
        # 1 acc_axial_b1
        # 2 acc_radial_b1
        # 3 acc_tangential_b1
        # 4 acc_axial_b2
        # 5 acc_radial_b2
        # 6 acc_tangential_b2
        # 7 microphone
        #
        # With the current config this is 8 -> 32.
        #
        # Each projected value may depend on every selected
        # input channel.
        # ----------------------------------------------------
        self.input_projection = nn.Linear(
            input_dim,
            d_model,
        )

        # ----------------------------------------------------
        # Add position information.
        #
        # PE shape: [1, seq_len, d_model]
        # ----------------------------------------------------
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=seq_len,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # Stack N independent Transformer encoder blocks.
        #
        # Each block has its OWN trainable weights.
        #
        # If num_layers = 1:
        #
        #     only one encoder block is used.
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Final classification head
        #
        # d_model
        #  ↓
        # LayerNorm
        #  ↓
        # d_model // 2
        #  ↓
        # GELU
        #  ↓
        # Dropout
        #  ↓
        # num_classes
        #
        # No Softmax here because CrossEntropyLoss expects
        # raw logits.
        # ----------------------------------------------------
        classifier_hidden_dim = d_model // 2

        if classifier_hidden_dim < 1:
            raise ValueError(
                "d_model is too small for the classifier head."
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(
                d_model,
                classifier_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                classifier_hidden_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify a batch of MAFAULDA windows.

            Input:
            [B, seq_len, input_dim]

        Example:
            [64, 1024, 8]

        Output:
            [B, num_classes]
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B, seq_len, input_dim], "
                f"but got {tuple(x.shape)}."
            )

        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, "
                f"but got {x.size(-1)}."
            )

        if x.size(1) == 0:
            raise ValueError("Sequence length must be positive.")

        if x.size(1) > self.max_seq_len:
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds "
                f"configured maximum {self.max_seq_len}."
            )

        # ----------------------------------------------------
        # 1. Input projection
        #
        # [B, seq_len, input_dim]
        #        ->
        # [B, seq_len, d_model]
        # ----------------------------------------------------
        x = self.input_projection(x)

        # ----------------------------------------------------
        # 2. Add positional information
        #
        # shape remains:
        #
        # [B, seq_len, d_model]
        # ----------------------------------------------------
        x = self.positional_encoding(x)

        # ----------------------------------------------------
        # 3. Transformer encoder stack
        #
        # shape remains:
        #
        # [B, seq_len, d_model]
        # ----------------------------------------------------
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # ----------------------------------------------------
        # 4. Mean pooling over time
        #
        # [B, seq_len, d_model]
        #        ->
        # [B, d_model]
        #
        # All time positions are averaged into one
        # representation for each input window.
        # ----------------------------------------------------
        x = x.mean(dim=1)

        # ----------------------------------------------------
        # 5. Classification
        #
        # [B, 32]
        #    ->
        # [B, num_classes]
        #
        # Returns raw logits.
        # ----------------------------------------------------
        logits = self.classifier(x)

        return logits

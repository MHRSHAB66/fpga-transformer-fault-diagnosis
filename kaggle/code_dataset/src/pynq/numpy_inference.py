"""
Pure NumPy Transformer inference engine for PYNQ/Zynq deployment.

This module has NO PyTorch dependency at inference time — it runs anywhere
that has NumPy, including the ARM processor on a PYNQ-Z2 board.

Typical usage
-------------
On your dev machine (PyTorch available):
    from src.pynq.export_weights import export
    export()                 # writes artifacts/pynq/numpy_weights.npz

On PYNQ board (copy numpy_weights.npz to SD card):
    from numpy_inference import NumpyTransformer
    model = NumpyTransformer.load("numpy_weights.npz")
    class_id, label, probs, logits = model.predict_raw(raw_window)

Architecture contract (must stay in sync with transformer_baseline.py)
-----------------------------------------------------------------------
    input [1024, 8]
    -> Linear(8, 32)        input_projection
    -> + sinusoidal PE [1024, 32]
    -> TransformerEncoderBlock x2  (post-norm: residual -> LayerNorm)
         self_attention (MHA, 4 heads)
         feedforward    Linear(32,64) -> GELU -> Linear(64,32)
    -> mean over sequence   [32]
    -> LayerNorm(32) -> Linear(32,16) -> GELU -> Linear(16,4)   classifier
    -> logits [4]

Normalization contract (must match FPGA normalization block)
-----------------------------------------------------------
    per-channel z-score computed over the 1024-sample time axis:
        mean = window.mean(axis=0)      # shape [8]
        std  = window.std(axis=0)       # shape [8], ddof=0
        safe_std = where(|std| < 1e-8, 1.0, std)
        normalized = (window - mean) / safe_std
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CLASS_NAMES = {0: "normal", 1: "imbalance", 2: "misalignment", 3: "bearing_fault"}
_EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Primitive ops
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias


def _mha(x: np.ndarray,
         in_proj_w: np.ndarray, in_proj_b: np.ndarray,
         out_w: np.ndarray, out_b: np.ndarray,
         num_heads: int = 4) -> np.ndarray:
    """Scaled dot-product multi-head self-attention. x: [seq, d_model]."""
    seq, d = x.shape
    head_dim = d // num_heads

    qkv = x @ in_proj_w.T + in_proj_b          # [seq, 3*d]
    q, k, v = np.split(qkv, 3, axis=-1)         # each [seq, d]

    # [num_heads, seq, head_dim]
    q = q.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)

    scale = np.float32(np.sqrt(head_dim))
    scores = q @ k.transpose(0, 2, 1) / scale   # [heads, seq, seq]
    scores -= scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn /= attn.sum(axis=-1, keepdims=True)

    out = attn @ v                               # [heads, seq, head_dim]
    out = out.transpose(1, 0, 2).reshape(seq, d)
    return out @ out_w.T + out_b


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NumpyTransformer:
    """
    Float32 Transformer fault classifier in pure NumPy.
    Compatible with PYNQ board ARM Python (numpy only at inference time).
    """

    SEQ_LEN = 1024
    IN_CHANNELS = 8
    D_MODEL = 32
    NUM_HEADS = 4
    NUM_LAYERS = 2
    NUM_CLASSES = 4

    def __init__(self, weights: dict[str, np.ndarray]):
        self._w = weights
        # Positional encoding: loaded from checkpoint buffer (exact match)
        # shape in checkpoint: [1, 1024, 32] → squeeze to [1024, 32]
        if "positional_encoding.pe" in weights:
            self._pe = weights["positional_encoding.pe"].squeeze(0)
        else:
            self._pe = self._make_sinusoidal_pe()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_sinusoidal_pe() -> np.ndarray:
        seq, d = NumpyTransformer.SEQ_LEN, NumpyTransformer.D_MODEL
        pe = np.zeros((seq, d), dtype=np.float32)
        pos = np.arange(seq, dtype=np.float32)[:, None]
        div = np.exp(
            np.arange(0, d, 2, dtype=np.float32) * (-np.log(10000.0) / d)
        )
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        return pe

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "NumpyTransformer":
        """
        Load weights from a PyTorch .pt checkpoint.
        Requires torch — run this on your dev machine, NOT on PYNQ.
        """
        import torch
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        weights = {k: v.numpy().astype(np.float32) for k, v in sd.items()}
        return cls(weights)

    def save(self, path: str | Path) -> None:
        """Save all weights + PE buffer to a portable .npz (no torch needed to load)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **self._w)
        size_kb = Path(path).stat().st_size / 1024
        print(f"[pynq] saved {len(self._w)} arrays to {path}  ({size_kb:.1f} KB)")

    @classmethod
    def load(cls, path: str | Path) -> "NumpyTransformer":
        """Load from .npz weights file — no PyTorch required."""
        data = np.load(str(path))
        weights = {k: data[k].astype(np.float32) for k in data.files}
        return cls(weights)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _encoder_block(self, x: np.ndarray, idx: int) -> np.ndarray:
        p = f"encoder_layers.{idx}"

        # Attention sub-layer  (post-norm: x = LayerNorm(x + attn_out))
        attn_out = _mha(
            x,
            self._w[f"{p}.self_attention.in_proj_weight"],
            self._w[f"{p}.self_attention.in_proj_bias"],
            self._w[f"{p}.self_attention.out_proj.weight"],
            self._w[f"{p}.self_attention.out_proj.bias"],
            self.NUM_HEADS,
        )
        x = _layer_norm(
            x + attn_out,
            self._w[f"{p}.attention_norm.weight"],
            self._w[f"{p}.attention_norm.bias"],
        )

        # FFN sub-layer  (post-norm: x = LayerNorm(x + ffn_out))
        ffn = x @ self._w[f"{p}.feedforward.0.weight"].T + self._w[f"{p}.feedforward.0.bias"]
        ffn = _gelu(ffn)
        ffn = ffn @ self._w[f"{p}.feedforward.3.weight"].T + self._w[f"{p}.feedforward.3.bias"]
        x = _layer_norm(
            x + ffn,
            self._w[f"{p}.feedforward_norm.weight"],
            self._w[f"{p}.feedforward_norm.bias"],
        )
        return x

    def predict(self, normalized_window: np.ndarray):
        """
        Inference on a pre-normalized window.

        Parameters
        ----------
        normalized_window : np.ndarray, shape [1024, 8], float32
            Per-channel z-score applied over the time axis.

        Returns
        -------
        class_id   : int          predicted class (0–3)
        label      : str          class name
        probs      : np.ndarray   softmax probabilities [4]
        logits     : np.ndarray   raw logits [4]
        """
        x = normalized_window.astype(np.float32)        # [1024, 8]

        # Input projection
        x = x @ self._w["input_projection.weight"].T \
              + self._w["input_projection.bias"]         # [1024, 32]

        # Positional encoding (dropout skipped at inference)
        x = x + self._pe                                 # [1024, 32]

        # Encoder layers
        for i in range(self.NUM_LAYERS):
            x = self._encoder_block(x, i)

        # Global average pooling
        x = x.mean(axis=0)                              # [32]

        # Classifier: LayerNorm -> Linear(32,16) -> GELU -> Linear(16,4)
        x = _layer_norm(x,
            self._w["classifier.0.weight"],
            self._w["classifier.0.bias"])
        x = x @ self._w["classifier.1.weight"].T + self._w["classifier.1.bias"]
        x = _gelu(x)
        logits = x @ self._w["classifier.4.weight"].T + self._w["classifier.4.bias"]

        probs = _softmax(logits)
        class_id = int(np.argmax(logits))
        return class_id, CLASS_NAMES[class_id], probs, logits

    def predict_raw(self, raw_window: np.ndarray):
        """
        Accept a raw (un-normalized) window, normalize, then predict.

        Parameters
        ----------
        raw_window : np.ndarray, shape [1024, 8], float32
            Raw sensor readings from the MAFAULDA CSV or a real sensor.

        Returns
        -------
        Same as predict().
        """
        w = raw_window.astype(np.float32)
        mean = w.mean(axis=0, keepdims=True)           # [1, 8]
        std = w.std(axis=0, keepdims=True)             # [1, 8], ddof=0
        safe_std = np.where(np.abs(std) < _EPSILON, 1.0, std)
        return self.predict((w - mean) / safe_std)

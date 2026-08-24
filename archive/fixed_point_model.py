"""
Stage 6b — Fixed-point inference simulator.

Re-implements the trained Transformer's forward pass explicitly (attention is
unrolled) so that reduced numeric precision can be injected exactly where the
hardware would see it. This is the Python reference that the HLS/RTL core will
be built to match, and the source of the Stage 6d golden vectors.

Design (the "hybrid" scheme chosen in Stage 6)
----------------------------------------------
- MATMULs (input projection, Q/K/V, attention output, FFN, classifier) run in
  quantized integer arithmetic: activations and weights are fake-quantized to
  ``act_bits`` / ``weight_bits`` (symmetric per-tensor), the product accumulates
  in high precision, biases are added in float. This mirrors an int16 MAC array
  with an int32 accumulator.
- NONLINEARITIES (softmax, LayerNorm, GELU) are computed in float, standing in
  for the higher-precision fixed-point / CORDIC / LUT units that Vitis HLS
  provides (hls::exp, hls::sqrt, a GELU LUT). Matching these bit-for-bit is not
  attempted; golden-vector checking is tolerance-based (CLAUDE.md Stage 8).
- Weight quantization is STATIC (done once at construction). Activation
  quantization here is DYNAMIC per-tensor (scale = max-abs of the tensor). At
  int16 this is essentially lossless; static activation scales for hardware are
  a later refinement and do not change this reference's structure.

Setting ``weight_bits=None`` and ``act_bits=None`` disables all quantization, so
this same code doubles as a numerical cross-check against the real PyTorch model
(used in ``verify`` below).

This module does not retrain, re-split, or modify the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.quantize.fixed_point import fake_quantize


@dataclass
class FixedPointConfig:
    weight_bits: Optional[int] = 16
    act_bits: Optional[int] = 16
    per_channel_weights: bool = False


def _maybe_quant(x: torch.Tensor, bits: Optional[int], per_channel: bool = False) -> torch.Tensor:
    if bits is None:
        return x
    return fake_quantize(x, bits, per_channel=per_channel, ch_axis=0)


class FixedPointTransformer:
    """Explicit fixed-point forward pass for the trained TransformerBaseline."""

    def __init__(self, state_dict: dict, model_cfg: dict, fp_cfg: FixedPointConfig):
        self.cfg = model_cfg
        self.fp = fp_cfg
        self.d_model = model_cfg["d_model"]
        self.nhead = model_cfg["nhead"]
        self.num_layers = model_cfg["num_layers"]
        self.head_dim = self.d_model // self.nhead

        # Keep a float copy of every parameter; weights are statically quantized.
        self.p = {k: v.detach().clone().float() for k, v in state_dict.items()}

    # --- primitive ops -----------------------------------------------------

    def _linear(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        """Quantized matmul: y = quant(x) @ quant(W)^T + bias(float)."""
        xq = _maybe_quant(x, self.fp.act_bits)
        wq = _maybe_quant(weight, self.fp.weight_bits, per_channel=self.fp.per_channel_weights)
        y = xq @ wq.t()
        if bias is not None:
            y = y + bias
        return y

    def _layernorm(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps=1e-5):
        # Computed in float (stands in for fixed-point mean/var + Newton invsqrt).
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + eps)
        return xhat * weight + bias

    def _gelu(self, x: torch.Tensor):
        # Exact erf GELU (matches nn.GELU default); hardware uses a LUT for this.
        return F.gelu(x)

    def _attention(self, x: torch.Tensor, layer: int):
        """Replicates nn.MultiheadAttention(batch_first=True) for one layer."""
        pfx = f"encoder_layers.{layer}.self_attention."
        in_w = self.p[pfx + "in_proj_weight"]   # (3*d, d)
        in_b = self.p[pfx + "in_proj_bias"]     # (3*d,)
        out_w = self.p[pfx + "out_proj.weight"]
        out_b = self.p[pfx + "out_proj.bias"]

        d = self.d_model
        wq, wk, wv = in_w[:d], in_w[d:2 * d], in_w[2 * d:]
        bq, bk, bv = in_b[:d], in_b[d:2 * d], in_b[2 * d:]

        q = self._linear(x, wq, bq)
        k = self._linear(x, wk, bk)
        v = self._linear(x, wv, bv)

        b, seq, _ = q.shape
        # (b, nhead, seq, head_dim)
        def split_heads(t):
            return t.view(b, seq, self.nhead, self.head_dim).transpose(1, 2)
        qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention. Scores matmul is quantized; softmax float.
        qh_q = _maybe_quant(qh, self.fp.act_bits)
        kh_q = _maybe_quant(kh, self.fp.act_bits)
        scores = (qh_q @ kh_q.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn_q = _maybe_quant(attn, self.fp.act_bits)
        vh_q = _maybe_quant(vh, self.fp.act_bits)
        ctx = attn_q @ vh_q  # (b, nhead, seq, head_dim)

        ctx = ctx.transpose(1, 2).contiguous().view(b, seq, d)
        return self._linear(ctx, out_w, out_b)

    def _encoder_block(self, x: torch.Tensor, layer: int):
        pfx = f"encoder_layers.{layer}."
        attn_out = self._attention(x, layer)
        x = self._layernorm(
            x + attn_out,
            self.p[pfx + "attention_norm.weight"],
            self.p[pfx + "attention_norm.bias"],
        )
        # FFN: Linear -> GELU -> Linear
        h = self._linear(x, self.p[pfx + "feedforward.0.weight"], self.p[pfx + "feedforward.0.bias"])
        h = self._gelu(h)
        h = self._linear(h, self.p[pfx + "feedforward.3.weight"], self.p[pfx + "feedforward.3.bias"])
        x = self._layernorm(
            x + h,
            self.p[pfx + "feedforward_norm.weight"],
            self.p[pfx + "feedforward_norm.bias"],
        )
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, input_dim)
        x = self._linear(x, self.p["input_projection.weight"], self.p["input_projection.bias"])
        x = x + self.p["positional_encoding.pe"][:, : x.size(1), :]  # PE kept float

        for layer in range(self.num_layers):
            x = self._encoder_block(x, layer)

        x = x.mean(dim=1)  # mean pooling over sequence

        # classifier: LayerNorm -> Linear -> GELU -> Linear
        x = self._layernorm(x, self.p["classifier.0.weight"], self.p["classifier.0.bias"])
        x = self._linear(x, self.p["classifier.1.weight"], self.p["classifier.1.bias"])
        x = self._gelu(x)
        x = self._linear(x, self.p["classifier.4.weight"], self.p["classifier.4.bias"])
        return x

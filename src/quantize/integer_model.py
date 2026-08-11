"""
Stage 6f / 8 — Full-integer Transformer inference reference.

Unlike the Stage 6b hybrid simulator (``fixed_point_model.py``, which keeps
softmax / LayerNorm / GELU in float), this reference is **integer-only**: every
tensor that flows through the network is an integer array, input and output
included. The only real numbers are per-operation constants — activation scales,
requant multipliers, and LUT contents — each of which maps to a fixed hardware
constant (a stored scale, an ``M0*2^-shift`` multiplier, or a BRAM table). This
is the exact numeric behaviour the HLS/RTL core will reproduce bit-for-bit, and
the source of the integer golden vectors.

Numeric scheme (see ``integer_ops.py`` for the primitives)
----------------------------------------------------------
- Activations: symmetric int16, one static per-position scale (calibrated on the
  TRAIN split only — never test).
- Weights: symmetric int16, per-tensor scale.
- MAC: int64 accumulator, then a requant multiply ``M = s_in*s_w/s_out`` back to
  int16 (float64 multiply-and-round in this reference; ``quantize_multiplier``
  emits the equivalent ``M0*2^-shift`` for hardware).
- Softmax: integer row-max subtract, exp via LUT, integer normalise to int16
  probabilities.
- LayerNorm: integer mean/variance, fixed-point reciprocal-sqrt (range-reduced
  LUT), per-channel affine folded into the output requant.
- GELU: full int16-domain LUT of the exact erf-GELU.

The scales come from a calibration JSON produced by ``calibrate_integer.py``.
This module never retrains, re-splits, or edits the checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.quantize.integer_ops import (
    qint_max, quantize_sym, requant_float,
    build_gelu_lut, gelu_lookup,
    build_exp_lut, exp_lookup,
    build_rsqrt_lut, rsqrt_fxp,
)

ACT_BITS = 16
W_BITS = 16
EXP_FRAC = 15
ATTN_BITS = 16            # softmax probabilities quantized to int16
LN_EPS = 1e-5


# ---------------------------------------------------------------------------
# Position naming (shared by calibration and integer inference)
# ---------------------------------------------------------------------------

def layer_positions(l: int) -> dict:
    """Names of every calibrated activation position inside encoder layer ``l``."""
    p = f"l{l}"
    return {
        "q": f"{p}.q", "k": f"{p}.k", "v": f"{p}.v",
        "scores": f"{p}.scores", "ctx": f"{p}.ctx", "attn_out": f"{p}.attn_out",
        "res1": f"{p}.res1", "ln1": f"{p}.ln1",
        "ff1": f"{p}.ff1", "ff_act": f"{p}.ff_act", "ff2": f"{p}.ff2",
        "res2": f"{p}.res2", "ln2": f"{p}.ln2",
    }


GLOBAL_POSITIONS = ["input", "embed", "pool", "cls_ln", "cls1", "cls_act", "logits"]


# ---------------------------------------------------------------------------
# Calibration container
# ---------------------------------------------------------------------------

@dataclass
class CalibStats:
    """Running per-position max-abs, turned into scales at the end."""
    maxabs: dict

    @classmethod
    def empty(cls):
        return cls(maxabs={})

    def update(self, name: str, tensor: np.ndarray) -> None:
        m = float(np.abs(tensor).max())
        if name not in self.maxabs or m > self.maxabs[name]:
            self.maxabs[name] = m

    def to_scales(self, act_bits: int = ACT_BITS) -> dict:
        qmax = qint_max(act_bits)
        scales = {n: (m / qmax if m > 0 else 1.0) for n, m in self.maxabs.items()}
        # Softmax output is a probability in [0, 1] -> fixed scale, no calibration.
        for key in list(scales):
            if key.endswith(".attn"):
                scales[key] = 1.0 / qint_max(ATTN_BITS)
        return scales


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class IntegerTransformer:
    def __init__(self, state_dict: dict, model_cfg: dict, scales: dict | None = None):
        self.cfg = model_cfg
        self.d_model = model_cfg["d_model"]
        self.nhead = model_cfg["nhead"]
        self.num_layers = model_cfg["num_layers"]
        self.head_dim = self.d_model // self.nhead

        self.f = {k: np.asarray(v, dtype=np.float64) for k, v in state_dict.items()}
        self.pe = self.f["positional_encoding.pe"][0]        # (seq, d_model)

        # --- quantize every weight matrix (per-tensor symmetric int16) --------
        self.qW: dict = {}
        self.sW: dict = {}
        self.bias: dict = {}
        d = self.d_model
        self._reg("proj", self.f["input_projection.weight"], self.f["input_projection.bias"])
        for l in range(self.num_layers):
            pfx = f"encoder_layers.{l}."
            in_w = self.f[pfx + "self_attention.in_proj_weight"]
            in_b = self.f[pfx + "self_attention.in_proj_bias"]
            self._reg(f"l{l}.wq", in_w[:d],        in_b[:d])
            self._reg(f"l{l}.wk", in_w[d:2*d],     in_b[d:2*d])
            self._reg(f"l{l}.wv", in_w[2*d:],      in_b[2*d:])
            self._reg(f"l{l}.out", self.f[pfx + "self_attention.out_proj.weight"],
                      self.f[pfx + "self_attention.out_proj.bias"])
            self._reg(f"l{l}.ff0", self.f[pfx + "feedforward.0.weight"],
                      self.f[pfx + "feedforward.0.bias"])
            self._reg(f"l{l}.ff3", self.f[pfx + "feedforward.3.weight"],
                      self.f[pfx + "feedforward.3.bias"])
        self._reg("cls1", self.f["classifier.1.weight"], self.f["classifier.1.bias"])
        self._reg("cls4", self.f["classifier.4.weight"], self.f["classifier.4.bias"])

        self.rsqrt_lut = build_rsqrt_lut()
        self.scales = scales
        if scales is not None:
            self._build_luts()

    def _reg(self, name, w, b):
        w = np.asarray(w, dtype=np.float64)
        s = float(np.abs(w).max()) / qint_max(W_BITS)
        s = s if s > 0 else 1.0
        self.sW[name] = s
        self.qW[name] = np.clip(np.round(w / s), -qint_max(W_BITS), qint_max(W_BITS)).astype(np.int64)
        self.bias[name] = np.asarray(b, dtype=np.float64)

    def set_scales(self, scales: dict) -> None:
        self.scales = scales
        self._build_luts()

    def _build_luts(self):
        s = self.scales
        self.gelu_lut = {}
        self.exp_lut = {}
        for l in range(self.num_layers):
            pos = layer_positions(l)
            self.gelu_lut[l] = build_gelu_lut(s[pos["ff1"]], s[pos["ff_act"]], ACT_BITS)
            arg = s[pos["scores"]]
            clamp = min(int(math.ceil(30.0 / arg)) + 1, 2 * qint_max(ACT_BITS))
            self.exp_lut[l] = build_exp_lut(arg, clamp, EXP_FRAC)
        self.cls_gelu_lut = build_gelu_lut(s["cls1"], s["cls_act"], ACT_BITS)

    # =====================================================================
    # Integer operations
    # =====================================================================

    def _linear(self, q_in, name, s_in, s_out):
        """int MAC + requant: codes(s_in) -> codes(s_out)."""
        qW, sW, b = self.qW[name], self.sW[name], self.bias[name]
        acc = q_in @ qW.T                                    # int64
        qb = np.round(b / (s_in * sW)).astype(np.int64)
        acc = acc + qb
        return requant_float(acc, (s_in * sW) / s_out, ACT_BITS)

    def _add(self, qa, sa, qb, sb, s_out):
        """Residual add of two int tensors at different scales -> codes(s_out)."""
        y = np.round(qa.astype(np.float64) * (sa / s_out)
                     + qb.astype(np.float64) * (sb / s_out))
        qmax = qint_max(ACT_BITS)
        return np.clip(y, -qmax, qmax).astype(np.int64)

    def _layernorm(self, q, s_in, gamma, beta, s_out):
        """Integer LayerNorm over the last dim (D=d_model)."""
        D = q.shape[-1]
        s = np.sum(q, axis=-1, keepdims=True)
        mean = np.round(s.astype(np.float64) / D).astype(np.int64)
        centered = q - mean                                  # int
        var = np.round(np.sum(centered.astype(np.int64) ** 2, axis=-1, keepdims=True)
                       .astype(np.float64) / D).astype(np.int64)
        epsq = int(round(LN_EPS / (s_in * s_in)))
        inv, frac = rsqrt_fxp(var + epsq, self.rsqrt_lut)     # 1/sqrt ~ inv*2^-frac
        norm = centered.astype(np.int64) * inv               # scale 2^-frac (real ~ centered/std)
        # out_real = gamma*norm_real + beta ; norm_real = norm * 2^-frac
        gm = gamma / ((1 << frac) * s_out)
        bt = beta / s_out
        y = np.round(norm.astype(np.float64) * gm + bt)
        qmax = qint_max(ACT_BITS)
        return np.clip(y, -qmax, qmax).astype(np.int64)

    def _softmax(self, q_scores, s_out):
        """Row-wise integer softmax over the last dim. Output int16 probs."""
        l = self._cur_layer
        lut, clamp = self.exp_lut[l]
        rowmax = q_scores.max(axis=-1, keepdims=True)
        d = np.clip(q_scores - rowmax, -clamp, 0)
        num = exp_lookup(d, lut, clamp)                      # int, scale 2^-EXP_FRAC
        denom = np.sum(num, axis=-1, keepdims=True)          # int64
        qmax_attn = qint_max(ATTN_BITS)
        q_attn = (num * qmax_attn + denom // 2) // np.maximum(denom, 1)
        return q_attn.astype(np.int64)                        # scale 1/qmax_attn

    def _attention(self, x, s_in, l):
        pos = layer_positions(l)
        qh = self._split(self._linear(x, f"l{l}.wq", s_in, self.scales[pos["q"]]))
        kh = self._split(self._linear(x, f"l{l}.wk", s_in, self.scales[pos["k"]]))
        vh = self._split(self._linear(x, f"l{l}.wv", s_in, self.scales[pos["v"]]))

        # scores = (qh @ kh^T) / sqrt(hd)
        acc = qh @ kh.transpose(0, 1, 3, 2)                  # (N,H,S,S) int64
        s_q, s_k = self.scales[pos["q"]], self.scales[pos["k"]]
        M = (s_q * s_k) / (math.sqrt(self.head_dim) * self.scales[pos["scores"]])
        q_scores = requant_float(acc, M, ACT_BITS)

        self._cur_layer = l
        q_attn = self._softmax(q_scores, None)               # int16 probs, scale 1/qmax_attn
        s_attn = 1.0 / qint_max(ATTN_BITS)

        acc_ctx = q_attn @ vh                                # (N,H,S,hd) int64
        s_v = self.scales[pos["v"]]
        q_ctx = requant_float(acc_ctx, (s_attn * s_v) / self.scales[pos["ctx"]], ACT_BITS)

        ctx = self._merge(q_ctx)                             # (N,S,d)
        return self._linear(ctx, f"l{l}.out", self.scales[pos["ctx"]], self.scales[pos["attn_out"]])

    def _split(self, x):
        N, S, _ = x.shape
        return x.reshape(N, S, self.nhead, self.head_dim).transpose(0, 2, 1, 3)

    def _merge(self, x):
        N, H, S, hd = x.shape
        return x.transpose(0, 2, 1, 3).reshape(N, S, H * hd)

    def _encoder(self, x, s_in, l):
        pos = layer_positions(l)
        pfx = f"encoder_layers.{l}."
        attn = self._attention(x, s_in, l)
        res1 = self._add(x, s_in, attn, self.scales[pos["attn_out"]], self.scales[pos["res1"]])
        ln1 = self._layernorm(res1, self.scales[pos["res1"]],
                              self.f[pfx + "attention_norm.weight"],
                              self.f[pfx + "attention_norm.bias"], self.scales[pos["ln1"]])
        ff1 = self._linear(ln1, f"l{l}.ff0", self.scales[pos["ln1"]], self.scales[pos["ff1"]])
        ff_act = gelu_lookup(ff1, self.gelu_lut[l], ACT_BITS)  # scale ff_act
        ff2 = self._linear(ff_act, f"l{l}.ff3", self.scales[pos["ff_act"]], self.scales[pos["ff2"]])
        res2 = self._add(ln1, self.scales[pos["ln1"]], ff2, self.scales[pos["ff2"]], self.scales[pos["res2"]])
        ln2 = self._layernorm(res2, self.scales[pos["res2"]],
                              self.f[pfx + "feedforward_norm.weight"],
                              self.f[pfx + "feedforward_norm.bias"], self.scales[pos["ln2"]])
        return ln2

    def infer_int(self, x_real: np.ndarray) -> np.ndarray:
        """Full integer forward. x_real: (N, seq, in_dim) normalized. Returns int logits (N,C)."""
        s = self.scales
        q = quantize_sym(x_real, s["input"], ACT_BITS)
        x = self._linear(q, "proj", s["input"], s["embed"])
        # add positional encoding (quantized to the embedding scale)
        q_pe = np.round(self.pe / s["embed"]).astype(np.int64)
        x = self._add(x, s["embed"], q_pe[None], s["embed"], s["embed"])

        s_in = s["embed"]
        for l in range(self.num_layers):
            x = self._encoder(x, s_in, l)
            s_in = s[layer_positions(l)["ln2"]]

        # mean pool over sequence
        pooled = np.sum(x, axis=1)                            # (N,d) int64
        q_pool = requant_float(pooled, s_in / (x.shape[1] * s["pool"]), ACT_BITS)

        # classifier: LN -> Linear -> GELU -> Linear
        cls_ln = self._layernorm(q_pool, s["pool"],
                                 self.f["classifier.0.weight"], self.f["classifier.0.bias"],
                                 s["cls_ln"])
        c1 = self._linear(cls_ln, "cls1", s["cls_ln"], s["cls1"])
        c_act = gelu_lookup(c1, self.cls_gelu_lut, ACT_BITS)
        logits = self._linear(c_act, "cls4", s["cls_act"], s["logits"])
        return logits

    # =====================================================================
    # Float pass with taps (calibration only — never used at inference)
    # =====================================================================

    def collect(self, x_real: np.ndarray, stats: CalibStats) -> None:
        """Run the exact-float forward and record per-position max-abs ranges."""
        f = self.f
        stats.update("input", x_real)
        x = x_real @ f["input_projection.weight"].T + f["input_projection.bias"]
        x = x + self.pe[None]
        stats.update("embed", x)

        for l in range(self.num_layers):
            pos = layer_positions(l)
            pfx = f"encoder_layers.{l}."
            d = self.d_model
            in_w, in_b = f[pfx + "self_attention.in_proj_weight"], f[pfx + "self_attention.in_proj_bias"]
            q = x @ in_w[:d].T + in_b[:d]
            k = x @ in_w[d:2*d].T + in_b[d:2*d]
            v = x @ in_w[2*d:].T + in_b[2*d:]
            stats.update(pos["q"], q); stats.update(pos["k"], k); stats.update(pos["v"], v)
            qh, kh, vh = self._fsplit(q), self._fsplit(k), self._fsplit(v)
            scores = (qh @ kh.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
            stats.update(pos["scores"], scores)
            attn = _softmax_np(scores)
            ctx = attn @ vh
            stats.update(pos["ctx"], ctx)
            ctx = self._fmerge(ctx)
            attn_out = ctx @ f[pfx + "self_attention.out_proj.weight"].T + f[pfx + "self_attention.out_proj.bias"]
            stats.update(pos["attn_out"], attn_out)
            res1 = x + attn_out
            stats.update(pos["res1"], res1)
            ln1 = _ln_np(res1, f[pfx + "attention_norm.weight"], f[pfx + "attention_norm.bias"])
            stats.update(pos["ln1"], ln1)
            h = ln1 @ f[pfx + "feedforward.0.weight"].T + f[pfx + "feedforward.0.bias"]
            stats.update(pos["ff1"], h)
            h_act = _gelu_np(h)
            stats.update(pos["ff_act"], h_act)
            h2 = h_act @ f[pfx + "feedforward.3.weight"].T + f[pfx + "feedforward.3.bias"]
            stats.update(pos["ff2"], h2)
            res2 = ln1 + h2
            stats.update(pos["res2"], res2)
            x = _ln_np(res2, f[pfx + "feedforward_norm.weight"], f[pfx + "feedforward_norm.bias"])
            stats.update(pos["ln2"], x)

        pooled = x.mean(axis=1)
        stats.update("pool", pooled)
        cls_ln = _ln_np(pooled, f["classifier.0.weight"], f["classifier.0.bias"])
        stats.update("cls_ln", cls_ln)
        c1 = cls_ln @ f["classifier.1.weight"].T + f["classifier.1.bias"]
        stats.update("cls1", c1)
        c_act = _gelu_np(c1)
        stats.update("cls_act", c_act)
        logits = c_act @ f["classifier.4.weight"].T + f["classifier.4.bias"]
        stats.update("logits", logits)

    def float_logits(self, x_real: np.ndarray) -> np.ndarray:
        """Exact-float logits (for accuracy reference / cross-check)."""
        stats = CalibStats.empty()
        # reuse collect's math but capture logits
        f = self.f
        x = x_real @ f["input_projection.weight"].T + f["input_projection.bias"]
        x = x + self.pe[None]
        for l in range(self.num_layers):
            pfx = f"encoder_layers.{l}."
            d = self.d_model
            in_w, in_b = f[pfx + "self_attention.in_proj_weight"], f[pfx + "self_attention.in_proj_bias"]
            q = x @ in_w[:d].T + in_b[:d]
            k = x @ in_w[d:2*d].T + in_b[d:2*d]
            v = x @ in_w[2*d:].T + in_b[2*d:]
            qh, kh, vh = self._fsplit(q), self._fsplit(k), self._fsplit(v)
            scores = (qh @ kh.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
            ctx = self._fmerge(_softmax_np(scores) @ vh)
            attn_out = ctx @ f[pfx + "self_attention.out_proj.weight"].T + f[pfx + "self_attention.out_proj.bias"]
            ln1 = _ln_np(x + attn_out, f[pfx + "attention_norm.weight"], f[pfx + "attention_norm.bias"])
            h = _gelu_np(ln1 @ f[pfx + "feedforward.0.weight"].T + f[pfx + "feedforward.0.bias"])
            h2 = h @ f[pfx + "feedforward.3.weight"].T + f[pfx + "feedforward.3.bias"]
            x = _ln_np(ln1 + h2, f[pfx + "feedforward_norm.weight"], f[pfx + "feedforward_norm.bias"])
        pooled = x.mean(axis=1)
        cls_ln = _ln_np(pooled, f["classifier.0.weight"], f["classifier.0.bias"])
        c1 = _gelu_np(cls_ln @ f["classifier.1.weight"].T + f["classifier.1.bias"])
        return c1 @ f["classifier.4.weight"].T + f["classifier.4.bias"]

    def _fsplit(self, x):
        N, S, _ = x.shape
        return x.reshape(N, S, self.nhead, self.head_dim).transpose(0, 2, 1, 3)

    def _fmerge(self, x):
        N, H, S, hd = x.shape
        return x.transpose(0, 2, 1, 3).reshape(N, S, H * hd)


# ---------------------------------------------------------------------------
# float helpers for calibration path
# ---------------------------------------------------------------------------

def _softmax_np(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _gelu_np(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _ln_np(x, gamma, beta, eps=LN_EPS):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta

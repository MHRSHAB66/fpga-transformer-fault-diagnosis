"""
Fixed-point quantization primitives for Stage 6.

The hardware target uses fixed-point integer arithmetic, not float32. This
module implements *symmetric* linear quantization, which maps a real value ``r``
to an integer ``q`` with a single per-tensor (or per-channel) scale and no
zero-point:

    q = round(r / scale)              (clamped to the signed n-bit range)
    r_hat = q * scale                 (dequantized approximation)

Symmetric (zero-point-free) quantization is chosen because it is the cheapest
to implement in hardware: a multiply-accumulate on symmetric int weights needs
no zero-point correction terms. This matches the reference in
``docs/next_steps_quantization_fpga.md`` (int8 symmetric).

The functions here operate on plain ``torch.Tensor`` objects and do not depend
on the model definition, so they can be reused for weight export (Stage 6c) and
golden-vector generation (Stage 6d).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def qint_range(n_bits: int) -> tuple[int, int]:
    """
    Return the inclusive (qmin, qmax) integer range for a signed n-bit value.

    A symmetric range is used (e.g. int8 -> [-127, 127], not -128) so that the
    representation is sign-symmetric and negation never overflows. This costs
    one code point but keeps the hardware sign handling trivial.
    """
    if n_bits < 2:
        raise ValueError("n_bits must be at least 2.")
    qmax = (1 << (n_bits - 1)) - 1  # e.g. 127 for int8, 32767 for int16
    qmin = -qmax
    return qmin, qmax


def compute_scale(
    tensor: torch.Tensor,
    n_bits: int,
    per_channel: bool = False,
    ch_axis: int = 0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute the symmetric quantization scale(s) for ``tensor``.

    Parameters
    ----------
    tensor : torch.Tensor
        The real-valued tensor to be quantized (e.g. a weight matrix).
    n_bits : int
        Target bit width.
    per_channel : bool, default False
        When True, one scale is computed per slice along ``ch_axis`` (typically
        the output-feature axis of a weight matrix). When False, a single scalar
        scale is used for the whole tensor.
    ch_axis : int, default 0
        Axis treated as the channel/output axis for per-channel scaling.
    eps : float
        Floor on the max-abs value so an all-zero slice does not divide by zero.

    Returns
    -------
    torch.Tensor
        A scalar scale (per_channel=False) or a 1-D tensor of scales with length
        equal to ``tensor.shape[ch_axis]`` (per_channel=True).
    """
    _, qmax = qint_range(n_bits)

    if per_channel:
        # Reduce over every axis except ch_axis to get one max-abs per channel.
        reduce_axes = [a for a in range(tensor.dim()) if a != ch_axis]
        max_abs = tensor.abs().amax(dim=reduce_axes)
    else:
        max_abs = tensor.abs().max()

    max_abs = torch.clamp(max_abs, min=eps)
    return max_abs / qmax


def quantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    n_bits: int,
    ch_axis: int = 0,
) -> torch.Tensor:
    """
    Quantize ``tensor`` to signed n-bit integers using the given ``scale``.

    Returns integer-valued codes stored in a float tensor (values are whole
    numbers in the qmin..qmax range). Keeping them in float makes it easy to
    dequantize and to export; the actual dtype cast happens at export time.
    """
    qmin, qmax = qint_range(n_bits)
    if scale.dim() > 0:
        scale = _broadcast_channel(scale, tensor.dim(), ch_axis)
    q = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return q


def dequantize(q: torch.Tensor, scale: torch.Tensor, ch_axis: int = 0) -> torch.Tensor:
    """Map integer codes back to their real-valued approximation ``q * scale``."""
    if scale.dim() > 0:
        scale = _broadcast_channel(scale, q.dim(), ch_axis)
    return q * scale


def fake_quantize(
    tensor: torch.Tensor,
    n_bits: int,
    per_channel: bool = False,
    ch_axis: int = 0,
) -> torch.Tensor:
    """
    Quantize then immediately dequantize ``tensor`` ("fake quant").

    The result is a float tensor that carries exactly the rounding/clamping
    error the fixed-point hardware would see, while staying compatible with the
    existing float model. This is the workhorse for measuring accuracy impact
    without rewriting the model in integer arithmetic.
    """
    scale = compute_scale(tensor, n_bits, per_channel=per_channel, ch_axis=ch_axis)
    q = quantize(tensor, scale, n_bits, ch_axis=ch_axis)
    return dequantize(q, scale, ch_axis=ch_axis)


def _broadcast_channel(scale: torch.Tensor, ndim: int, ch_axis: int) -> torch.Tensor:
    """Reshape a 1-D per-channel scale so it broadcasts against an ndim tensor."""
    shape = [1] * ndim
    shape[ch_axis] = scale.numel()
    return scale.view(shape)


@dataclass
class QuantizedTensor:
    """
    Container pairing integer codes with the scale needed to reconstruct them.

    Used by weight export (Stage 6c): the hardware stores ``int_codes`` in BRAM
    and multiplies by ``scale`` (a per-tensor constant or a per-channel array).
    """

    int_codes: torch.Tensor
    scale: torch.Tensor
    n_bits: int
    per_channel: bool
    ch_axis: int

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        n_bits: int,
        per_channel: bool = False,
        ch_axis: int = 0,
    ) -> "QuantizedTensor":
        scale = compute_scale(tensor, n_bits, per_channel=per_channel, ch_axis=ch_axis)
        codes = quantize(tensor, scale, n_bits, ch_axis=ch_axis)
        return cls(
            int_codes=codes,
            scale=scale,
            n_bits=n_bits,
            per_channel=per_channel,
            ch_axis=ch_axis,
        )

    def dequantized(self) -> torch.Tensor:
        return dequantize(self.int_codes, self.scale, ch_axis=self.ch_axis)

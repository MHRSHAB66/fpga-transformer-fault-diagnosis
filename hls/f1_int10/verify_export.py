"""
Compiler-free verification of the F_1layer @ int10 HLS data contract.

Independent NumPy re-implementation of the C++ core, reading the SAME exported
files under hls/f1_int10/data + hls/f1_int10/golden, asserting it reproduces the
int10 golden logits bit-for-bit. Mirrors hls/verify_export.py but for the
1-layer int10 model (ACT/ATTN qmax = 2^9-1 = 511, LAYERS = 1). It does NOT
import integer_model, so agreement is a genuine second-source check.

Run
---
    python hls/f1_int10/verify_export.py        # from repo root
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
GOLDEN = HERE / "golden"

SEQ, CH, D, NHEAD, DFF, LAYERS, CLS_HID, NUM_CLASSES = 1024, 8, 32, 4, 64, 1, 16, 4
HD = D // NHEAD
ACT_QMAX = (1 << 9) - 1            # int10
ATTN_QMAX = (1 << 9) - 1           # int10
RSQRT_MANT_BITS, RSQRT_FRAC = 8, 15
SQRT1_2 = round((1.0 / np.sqrt(2.0)) * (1 << RSQRT_FRAC))


def read_ints(name):
    vals = (DATA / name).read_text().split()
    return np.array(vals[1:], dtype=np.int64)


def read_floats(name):
    vals = (DATA / name).read_text().split()
    return np.array(vals[1:], dtype=np.float64)


def rint(x):
    return np.rint(x)


def clamp_act(y):
    return np.clip(rint(y), -ACT_QMAX, ACT_QMAX).astype(np.int64)


def requant(acc, M):
    return clamp_act(acc.astype(np.float64) * M)


def rsqrt_fxp(v, lut):
    v = max(int(v), 1)
    e = 0; t = v
    while t > 1:
        t >>= 1; e += 1
    shift = e - RSQRT_MANT_BITS
    mant_idx = (v >> shift) - (1 << RSQRT_MANT_BITS) if shift >= 0 else (v << (-shift)) - (1 << RSQRT_MANT_BITS)
    mant_idx = min(max(mant_idx, 0), (1 << RSQRT_MANT_BITS) - 1)
    y = int(lut[mant_idx])
    if e & 1:
        y = (y * SQRT1_2) >> RSQRT_FRAC
    y = y >> (e >> 1)
    return y


class P:
    def __init__(self):
        self.W, self.b, self.M, self.add = {}, {}, {}, {}
        self.ln_gm, self.ln_bt, self.ln_epsq = {}, {}, {}
        names = ["proj", "cls1", "cls4"]
        for l in range(LAYERS):
            for s in ["wq", "wk", "wv", "out", "ff0", "ff3"]:
                names.append(f"l{l}.{s}")
        for n in names:
            self.W[n] = read_ints(f"W_{n}.txt")
            self.b[n] = read_ints(f"b_{n}.txt")
            self.M[n] = read_floats(f"M_{n}.txt")[0]
        for l in range(LAYERS):
            self.M[f"l{l}.scores"] = read_floats(f"M_l{l}.scores.txt")[0]
            self.M[f"l{l}.ctx"] = read_floats(f"M_l{l}.ctx.txt")[0]
        self.M["pool"] = read_floats("M_pool.txt")[0]
        for l in range(LAYERS):
            self.add[f"l{l}.res1"] = read_floats(f"add_l{l}.res1.txt")
            self.add[f"l{l}.res2"] = read_floats(f"add_l{l}.res2.txt")
        self.pe = read_ints("pe_codes.txt")
        for n in ["cls_ln"] + [f"l{l}.ln{k}" for l in range(LAYERS) for k in (1, 2)]:
            self.ln_gm[n] = read_floats(f"ln_gm_{n}.txt")
            self.ln_bt[n] = read_floats(f"ln_bt_{n}.txt")
            self.ln_epsq[n] = int(read_ints(f"ln_epsq_{n}.txt")[0])
        self.gelu = [read_ints(f"gelu_l{l}.txt") for l in range(LAYERS)]
        self.gelu_cls = read_ints("gelu_cls.txt")
        self.exp = [read_ints(f"exp_l{l}.txt") for l in range(LAYERS)]
        self.exp_clamp = [int(read_ints(f"exp_clamp_l{l}.txt")[0]) for l in range(LAYERS)]
        self.rsqrt = read_ints("rsqrt.txt")


def linear(x, name, p, Dout):
    W = p.W[name].reshape(Dout, -1)
    acc = x @ W.T + p.b[name]
    return requant(acc, p.M[name])


def gelu(x, lut):
    idx = np.clip(x, -ACT_QMAX, ACT_QMAX) + ACT_QMAX
    return lut[idx]


def layernorm(x, name, p):
    gm, bt, epsq = p.ln_gm[name], p.ln_bt[name], p.ln_epsq[name]
    mean = rint(x.sum(-1, keepdims=True) / D).astype(np.int64)
    c = x - mean
    var = rint((c.astype(np.int64) ** 2).sum(-1, keepdims=True) / D).astype(np.int64)
    out = np.empty_like(x)
    for r in range(x.shape[0]):
        inv = rsqrt_fxp(int(var[r, 0]) + epsq, p.rsqrt)
        out[r] = clamp_act(c[r].astype(np.float64) * inv * gm + bt)
    return out


def attention(x, l, p):
    Ln = f"l{l}"
    q = linear(x, f"{Ln}.wq", p, D).reshape(SEQ, NHEAD, HD)
    k = linear(x, f"{Ln}.wk", p, D).reshape(SEQ, NHEAD, HD)
    v = linear(x, f"{Ln}.wv", p, D).reshape(SEQ, NHEAD, HD)
    clamp = p.exp_clamp[l]; elut = p.exp[l]
    ctx = np.zeros((SEQ, NHEAD, HD), dtype=np.int64)
    for h in range(NHEAD):
        sc = requant(q[:, h, :] @ k[:, h, :].T, p.M[f"{Ln}.scores"])
        d = np.clip(sc - sc.max(-1, keepdims=True), -clamp, 0)
        num = elut[d + clamp]
        denom = np.maximum(num.sum(-1, keepdims=True), 1)
        qattn = (num * ATTN_QMAX + denom // 2) // denom
        ctx[:, h, :] = requant(qattn @ v[:, h, :], p.M[f"{Ln}.ctx"])
    return linear(ctx.reshape(SEQ, D), f"{Ln}.out", p, D)


def forward(inp, p):
    x = linear(inp, "proj", p, D)
    x = clamp_act((x + p.pe.reshape(SEQ, D)).astype(np.float64))
    for l in range(LAYERS):
        Ln = f"l{l}"
        a = attention(x, l, p)
        r0, r1 = p.add[f"{Ln}.res1"]
        res = clamp_act(x.astype(np.float64) * r0 + a.astype(np.float64) * r1)
        ln1 = layernorm(res, f"{Ln}.ln1", p)
        ff = gelu(linear(ln1, f"{Ln}.ff0", p, DFF), p.gelu[l])
        ff2 = linear(ff, f"{Ln}.ff3", p, D)
        r0, r1 = p.add[f"{Ln}.res2"]
        res = clamp_act(ln1.astype(np.float64) * r0 + ff2.astype(np.float64) * r1)
        x = layernorm(res, f"{Ln}.ln2", p)
    pooled = requant(x.sum(0, keepdims=True), p.M["pool"])
    cls_ln = layernorm(pooled, "cls_ln", p)
    c1 = gelu(linear(cls_ln, "cls1", p, CLS_HID), p.gelu_cls)
    return linear(c1, "cls4", p, NUM_CLASSES)[0]


def main():
    p = P()
    lines = (GOLDEN / "golden_in.txt").read_text().splitlines()
    n, seq, ch = map(int, lines[0].split())
    inputs = [np.array(lines[1 + i].split(), dtype=np.int64).reshape(seq, ch) for i in range(n)]
    exp = [ln.split() for ln in (GOLDEN / "golden_expected.txt").read_text().splitlines() if not ln.startswith("#")]

    class_ok = logit_ok = 0
    for i in range(n):
        logits = forward(inputs[i], p)
        pred = int(np.argmax(logits))
        exp_pred = int(exp[i][2])
        exp_logits = np.array(exp[i][3:3 + NUM_CLASSES], dtype=np.int64)
        c = pred == exp_pred
        lm = bool(np.array_equal(logits, exp_logits))
        class_ok += c; logit_ok += lm
        print(f"  {i:>2}  pred={pred}  expected={exp_pred}  "
              f"{'ok ' if c else 'ERR'}  logits {'exact' if lm else 'DIFF'}")
    print(f"\n  class predictions : {class_ok}/{n}")
    print(f"  bit-exact logits  : {logit_ok}/{n}")
    ok = class_ok == n and logit_ok == n
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

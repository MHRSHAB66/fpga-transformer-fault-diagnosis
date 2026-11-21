"""Day-2 numeric-contract tests for the DEPLOYED SEQ=256 INT10 core.

Three references are kept apart on purpose:

* oracle      src/quantize/int10_contract.py -- exact Python-int mirror of the HLS
              arithmetic (Q40 + div_round_even). Width-independent.
* hls         the deployed HLS source itself, compiled with g++ through
              tests/cpp/int10_contract_harness.cpp, in two builds: native wide
              integers (the C-model build used for the 95,404-window comparison)
              and HLS_USE_AP_INT (ap_int<10>/<32>/<20>, the synthesis widths) when
              the Xilinx headers are present.
* reference   src/quantize/integer_model.py / integer_ops.py -- the float64-requant
              Python reference used for accuracy numbers.

"bit-exact" tests compare integers for equality; "approximation" tests compare
against real-valued math with a stated tolerance. Nothing here writes to any
deployed artifact.

    python -m pytest tests/test_int10_numeric_contract.py -q
"""
from __future__ import annotations

import atexit
import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.quantize import int10_contract as c
from src.quantize.integer_ops import (
    qint_max, quantize_sym, requant_float, rsqrt_fxp, build_rsqrt_lut,
    build_gelu_lut, gelu_lookup, build_exp_lut,
)
from src.data.token_aggregation import aggregate_round_div

ROOT = c.PROJECT_ROOT
HARNESS_SRC = ROOT / "tests" / "cpp" / "int10_contract_harness.cpp"
STUBS = ROOT / "hls" / "f1_int10" / "gpp_csim" / "stubs"
VITIS = ROOT / "hls" / "f1_int10_seq256" / "vitis"
XILINX_INC = Path(os.environ.get("XILINX_HLS_INCLUDE", "G:/Xilinx/Vitis_HLS/2020.2/include"))
EXPORT_CONFIG = ROOT / "artifacts" / "quantized" / "f_seq256_int10" / "integer_config.json"
CHECKPOINT = (ROOT / "artifacts" / "experiments" / "transformer_f1_seq256_pool4"
              / "kaggle_v3" / "seq256_pool4" / "scratch" / "best.pt")
INGRESS_GOLDEN = ROOT / "hls" / "f1_int10_seq256" / "golden_ingress1024"

H = c.load_header_constants()
RSQRT_LUT = np.array(H["rsqrt_lut"], dtype=np.int64)
SINGLE_MULTIPLIERS = ["proj", "l0_wq", "l0_wk", "l0_wv", "l0_out", "l0_ff0", "l0_ff3",
                      "cls1", "cls4", "l0_scores", "l0_ctx", "pool"]
LINEAR_DIN = {"proj": 8, "l0_wq": 32, "l0_wk": 32, "l0_wv": 32, "l0_out": 32,
              "l0_ff0": 32, "l0_ff3": 64, "cls1": 32, "cls4": 16}
SEQ, HD, D = 256, 8, 32


def max_abs_acc(name: str) -> int:
    """Worst-case |MAC + bias| for the requant at ``name`` (inputs in [-511, 511])."""
    if name in LINEAR_DIN:
        W = np.array(H["W_" + name], dtype=np.int64).reshape(-1, LINEAR_DIN[name])
        b = np.array(H["b_" + name], dtype=np.int64)
        return int((511 * np.abs(W).sum(1) + np.abs(b)).max())
    return {"l0_scores": HD * 511 * 511,          # q.k over HD
            "l0_ctx": SEQ * 511 * 511,            # loose: sum_j p_j*v_j with p_j <= 511
            "pool": SEQ * 511}[name]


# ---------------------------------------------------------------------------
# g++ harness
# ---------------------------------------------------------------------------

class _Harness:
    _dir = None
    builds: dict = {}

    @classmethod
    def get(cls, kind: str):
        if kind in cls.builds:
            return cls.builds[kind]
        gxx = shutil.which("g++")
        if gxx is None:
            cls.builds[kind] = None
            return None
        if cls._dir is None:
            cls._dir = tempfile.mkdtemp(prefix="int10_harness_")
            atexit.register(shutil.rmtree, cls._dir, True)
        exe = Path(cls._dir) / f"harness_{kind}.exe"
        cmd = [gxx, "-O2", "-std=c++17", "-Wno-unknown-pragmas",
               "-I", str(STUBS), "-I", str(VITIS)]
        if kind == "ap":
            if not (XILINX_INC / "ap_int.h").exists():
                cls.builds[kind] = None
                return None
            cmd += ["-DHLS_USE_AP_INT", "-I", str(XILINX_INC)]
        cmd += [str(HARNESS_SRC), "-o", str(exe)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        cls.builds[kind] = exe
        return exe


def run_harness(kind: str, lines: list[str]) -> list[str]:
    exe = _Harness.get(kind)
    out = subprocess.run([str(exe)], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True).stdout.split("\n")
    out = [o.strip() for o in out if o.strip()]
    assert len(out) == len(lines), (len(out), len(lines))
    assert not any(o.startswith("ERR") for o in out)
    return out


class HarnessMixin:
    KINDS = ("native", "ap")

    def kinds(self):
        ks = [k for k in self.KINDS if _Harness.get(k) is not None]
        if not ks:
            self.skipTest("g++ not available")
        return ks


# ---------------------------------------------------------------------------
# A. INT10 quantisation and saturation
# ---------------------------------------------------------------------------

class QuantizeInt10Tests(unittest.TestCase):
    def test_symmetric_range(self):
        self.assertEqual(qint_max(10), 511)
        self.assertEqual(c.ACT_QMAX, 511)
        q = quantize_sym(np.array([-1e9, 1e9]), 1.0, 10)
        self.assertEqual(q.tolist(), [-511, 511])            # never -512

    def test_halfway_and_boundaries_round_half_even(self):
        x = np.array([0.0, 1e-12, -1e-12, 0.4999, 0.5, 0.5001, 1.5, 2.5, -0.5, -1.5, -2.5,
                      510.4, 510.5, 510.6, 511.0, 511.49, 511.5, 512.0, 600.0,
                      -511.0, -511.5, -600.0])
        want = [0, 0, 0, 0, 0, 1, 2, 2, 0, -2, -2,
                510, 510, 511, 511, 511, 511, 511, 511,
                -511, -511, -511]
        self.assertEqual(quantize_sym(x, 1.0, 10).tolist(), want)

    def test_calibration_max_maps_to_511(self):
        s = json.loads(EXPORT_CONFIG.read_text())["scales"]["input"]
        self.assertEqual(int(quantize_sym(np.array([511 * s]), s, 10)[0]), 511)
        self.assertEqual(int(quantize_sym(np.array([-511 * s]), s, 10)[0]), -511)

    def test_clipping_is_saturation_not_wraparound(self):
        # Quantisation clipping: an out-of-range REAL value saturates to +-511.
        # (A two's-complement ap_int<10> overflow would wrap 512 -> -512; the
        # contract never relies on that -- see the width-bound tests below.)
        q = quantize_sym(np.array([512.0, 1023.0, -512.0, -1023.0]), 1.0, 10)
        self.assertEqual(q.tolist(), [511, 511, -511, -511])
        self.assertEqual([c.clamp_act(v) for v in (512, 1 << 40, -512, -(1 << 40))],
                         [511, 511, -511, -511])


# ---------------------------------------------------------------------------
# B. Q40 requantisation
# ---------------------------------------------------------------------------

class DivRoundEvenOracleTests(unittest.TestCase):
    def test_hand_cases(self):
        one, half = c.Q40_ONE, c.Q40_ONE // 2
        cases = {0: 0, half - 1: 0, half: 0, half + 1: 1,          # 0.5 -> 0 (even)
                 one + half - 1: 1, one + half: 2, one + half + 1: 2,   # 1.5 -> 2
                 2 * one + half: 2, 3 * one + half: 4,              # 2.5 -> 2, 3.5 -> 4
                 -half: 0, -half - 1: -1, -(one + half): -2, -(2 * one + half): -2}
        for n, want in cases.items():
            self.assertEqual(c.div_round_even(n, one), want, n)

    def test_vectorised_matches_scalar(self):
        rng = np.random.default_rng(1)
        n = np.concatenate([rng.integers(-(1 << 62), 1 << 62, 4000),
                            (np.arange(-40, 41) * (c.Q40_ONE // 2))]).astype(np.int64)
        for den in (c.Q40_ONE, 32, 7, 1):
            self.assertEqual(c.div_round_even_np(n, den).tolist(),
                             [c.div_round_even(int(v), den) for v in n])

    def test_header_q40_constants_follow_export_rule(self):
        # every Q40 constant in weights_int10.h == Decimal(text)*2^40, half-even,
        # where text is the exported data/*.txt value (gen_weights_header rule)
        data = {"M_": "M_{}.txt", "add_": "add_{}.txt", "ln_gm_": "ln_gm_{}.txt",
                "ln_bt_": "ln_bt_{}.txt"}
        checked = 0
        for name, vals in H.items():
            if not name.endswith("_q40"):
                continue
            base = name[:-4]
            pre = next(p for p in data if base.startswith(p))
            stem = base[len(pre):].replace("l0_", "l0.").replace("cls_ln", "cls_ln")
            text = c.read_data_file(data[pre].format(stem))
            self.assertEqual(vals, [c.to_q40(t) for t in text], name)
            self.assertEqual(H[base], text, base)              # double array == text
            checked += 1
        self.assertGreaterEqual(checked, 20)


class Q40RequantVsHlsTests(HarnessMixin, unittest.TestCase):
    def _edge_accs(self, mq: int, amax: int) -> list[int]:
        accs = {0, 1, -1, amax, -amax}
        for y in (0, 1, 2, 7, 100, 254, 255, 509, 510, 511, 512, 600):
            for sgn in (1, -1):
                b = sgn * (2 * y + 1) * c.Q40_ONE // (2 * abs(mq))   # near (y+0.5)/M
                accs.update(b + d for d in (-2, -1, 0, 1, 2))
        return sorted(a for a in accs if abs(a) < (1 << 31))

    def test_deployed_multipliers_edge_cases(self):
        lines, want = [], []
        for name in SINGLE_MULTIPLIERS:
            mq = H[f"M_{name}_q40"][0]
            for a in self._edge_accs(mq, max_abs_acc(name)):
                lines.append(f"req {a} {mq}")
                want.append(str(c.requant_q40(a, mq)))
        for k in self.kinds():
            self.assertEqual(run_harness(k, lines), want, k)

    def test_exact_ties_and_signed_multipliers(self):
        # Synthetic multipliers whose products land exactly on .5 (ties) plus
        # negative multipliers; the HLS requant is a signed function.
        lines, want = [], []
        for mq in (1 << 39, 3 << 38, -(1 << 39), -(3 << 38), 1, -1, c.Q40_ONE, -c.Q40_ONE):
            for a in list(range(-9, 10)) + [511, 512, -511, -512, 1023, -1023, 1 << 20]:
                lines.append(f"req {a} {mq}")
                want.append(str(c.requant_q40(a, mq)))
        self.assertEqual(c.requant_q40(1, 1 << 39), 0)       # 0.5  -> 0
        self.assertEqual(c.requant_q40(3, 1 << 39), 2)       # 1.5  -> 2
        self.assertEqual(c.requant_q40(-3, 1 << 39), -2)     # -1.5 -> -2
        self.assertEqual(c.requant_q40(-5, 1 << 39), -2)     # -2.5 -> -2
        self.assertEqual(c.requant_q40(2000, c.Q40_ONE), 511)
        for k in self.kinds():
            self.assertEqual(run_harness(k, lines), want, k)

    def test_random_div_and_mul(self):
        rng = np.random.default_rng(2)
        n = rng.integers(-(1 << 62), 1 << 62, 3000)
        a = rng.integers(-(1 << 31), 1 << 31, 3000)
        m = rng.integers(-(1 << 31), 1 << 31, 3000)          # |a*m| < 2^62: no int64 overflow
        lines = [f"div {v}" for v in n] + [f"mul {x} {y}" for x, y in zip(a, m)]
        want = [str(c.div_round_even(int(v), c.Q40_ONE)) for v in n] + \
               [str(int(x) * int(y)) for x, y in zip(a, m)]
        for k in self.kinds():
            self.assertEqual(run_harness(k, lines), want, k)

    def test_residual_q40(self):
        rng = np.random.default_rng(3)
        pairs = [(a, b) for a in (-511, -510, -1, 0, 1, 510, 511) for b in (-511, -1, 0, 1, 511)]
        pairs += [tuple(int(v) for v in p) for p in rng.integers(-511, 512, (3000, 2))]
        for which in ("res1", "res2"):
            r0q, r1q = H[f"add_l0_{which}_q40"]
            lines = [f"{which} {a} {b}" for a, b in pairs]
            want = [str(c.residual_q40(a, b, r0q, r1q)) for a, b in pairs]
            for k in self.kinds():
                self.assertEqual(run_harness(k, lines), want, (which, k))


class Q40VsPythonReferenceTests(unittest.TestCase):
    """The accuracy numbers come from the float64 reference; show it is
    bit-identical to the Q40 contract over the deployed constants' FULL reachable
    input ranges (not merely on the 95,404 test windows)."""

    def test_requant_exhaustive_reachable_range(self):
        for name in SINGLE_MULTIPLIERS:
            mq, m = H[f"M_{name}_q40"][0], float(H[f"M_{name}"][0])
            amax = max_abs_acc(name)
            for lo in range(-amax, amax + 1, 1 << 24):
                acc = np.arange(lo, min(lo + (1 << 24), amax + 1), dtype=np.int64)
                q = c.requant_q40_np(acc, mq)
                f = requant_float(acc, m, 10)
                bad = np.flatnonzero(q != f)
                self.assertEqual(bad.size, 0,
                                 f"{name}: acc={acc[bad[:3]]} q40={q[bad[:3]]} float={f[bad[:3]]}")

    def test_residual_exhaustive(self):
        a, b = np.meshgrid(np.arange(-511, 512), np.arange(-511, 512))
        a, b = a.ravel().astype(np.int64), b.ravel().astype(np.int64)
        for which in ("add_l0_res1", "add_l0_res2"):
            r0q, r1q = H[which + "_q40"]
            r0, r1 = map(float, H[which])
            q = np.clip(c.div_round_even_np(a * r0q + b * r1q, c.Q40_ONE), -511, 511)
            f = np.clip(np.round(a * r0 + b * r1), -511, 511).astype(np.int64)
            self.assertEqual(int((q != f).sum()), 0, which)

    def test_layernorm_random_rows(self):
        # reference LN (float64 affine, integer mean/var/rsqrt) vs Q40 LN oracle
        rng = np.random.default_rng(9)
        lut = build_rsqrt_lut()
        for n in ("l0_ln1", "l0_ln2", "cls_ln"):
            gq, bq = H[f"ln_gm_{n}_q40"], H[f"ln_bt_{n}_q40"]
            g = np.array(H[f"ln_gm_{n}"], dtype=np.float64)
            bt = np.array(H[f"ln_bt_{n}"], dtype=np.float64)
            for _ in range(5000):
                sc = int(rng.choice([1, 5, 30, 150, 511]))
                x = rng.integers(-sc, sc + 1, D).astype(np.int64)
                mean = int(np.round(x.sum() / D))
                cc = x - mean
                var = int(np.round(float((cc ** 2).sum()) / D))
                inv = int(rsqrt_fxp(np.array([var]), lut)[0][0])
                ref = np.clip(np.round(cc * inv * g + bt), -511, 511).astype(np.int64)
                self.assertEqual(ref.tolist(), c.layernorm_q40(x, gq, bq, 0, RSQRT_LUT), n)

    def test_float_reference_is_not_q40_in_general(self):
        # Documented limitation: the float path is an implementation of the same
        # rounding, not the same arithmetic. With a multiplier whose Q40 code
        # is not exactly representable the two can disagree near a tie; the
        # deployed constants simply never reach such an accumulator (tests above).
        m = 0.5 + 2.0 ** -42                   # Q40 code rounds this to exactly 0.5
        mq = c.to_q40(m)
        self.assertEqual(mq, 1 << 39)
        self.assertEqual(c.requant_q40(1, mq), 0)                      # 0.5 -> 0
        self.assertEqual(int(requant_float(np.array([1]), m, 10)[0]), 1)  # 0.5000..2 -> 1


class WidthBoundTests(unittest.TestCase):
    """Intermediate widths of the deployed HLS source (ap_types.h + top)."""

    def test_mac_accumulators_fit_ap_int32_and_win_long(self):
        # acc_t = ap_int<32>; requant() also takes `long`, which is 32-bit on
        # LLP64 (Windows/MinGW). All reachable |acc| must stay below 2^31.
        for name in SINGLE_MULTIPLIERS:
            self.assertLess(max_abs_acc(name), 1 << 31, name)

    def test_q40_products_fit_int64(self):
        lim = (1 << 63) - 1
        for name in SINGLE_MULTIPLIERS:
            self.assertLess(max_abs_acc(name) * abs(H[f"M_{name}_q40"][0]), lim, name)
        for which in ("add_l0_res1_q40", "add_l0_res2_q40"):
            r0q, r1q = H[which]
            self.assertLess(511 * (abs(r0q) + abs(r1q)), lim, which)
        inv_max = c.rsqrt_fxp_int(1, RSQRT_LUT)            # v<1 clamps to 1
        self.assertEqual(inv_max, 1 << 15)
        for n in ("l0_ln1", "l0_ln2", "cls_ln"):
            gm = max(abs(v) for v in H[f"ln_gm_{n}_q40"])
            bt = max(abs(v) for v in H[f"ln_bt_{n}_q40"])
            self.assertLess(1022 * inv_max * gm + bt, lim, n)   # |c| <= 1022
            self.assertLess(D * 1022 * 1022, 1 << 31, n)        # LN sum of squares

    def test_softmax_and_lut_widths(self):
        exp_lut = H["exp_l0"]
        self.assertLess(max(exp_lut), 1 << 19)                 # lut_t = ap_int<20>
        denom = SEQ * max(exp_lut)
        self.assertLess(max(exp_lut) * 511 + denom // 2, 1 << 31)
        for name in ("gelu_l0", "gelu_cls"):
            self.assertLessEqual(max(abs(v) for v in H[name]), 511)
        self.assertLess(max(H["rsqrt_lut"]) * c.SQRT1_2_Q15, 1 << 31)


# ---------------------------------------------------------------------------
# C. GELU, exp/softmax, rsqrt, LayerNorm
# ---------------------------------------------------------------------------

def _export_scales():
    return json.loads(EXPORT_CONFIG.read_text())["scales"]


class LutContentTests(unittest.TestCase):
    def test_gelu_tables_rebuild_bit_exact(self):
        s = _export_scales()
        self.assertEqual(build_gelu_lut(s["l0.ff1"], s["l0.ff_act"], 10).tolist(), H["gelu_l0"])
        self.assertEqual(build_gelu_lut(s["cls1"], s["cls_act"], 10).tolist(), H["gelu_cls"])
        self.assertEqual(len(H["gelu_l0"]), 1023)

    def test_gelu_approximation_vs_erf(self):
        s = _export_scales()
        for lut, si, so in ((H["gelu_l0"], s["l0.ff1"], s["l0.ff_act"]),
                            (H["gelu_cls"], s["cls1"], s["cls_act"])):
            q = np.arange(-511, 512)
            x = q * si
            true = 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
            got = np.array(lut) * so
            inside = np.abs(true / so) <= 511
            self.assertLessEqual(np.abs(got - true)[inside].max(), 0.5 * so + 1e-12)
            self.assertTrue(np.all(np.array(lut)[~inside] == 511))   # clipped only at top
            self.assertEqual(lut[511], 0)                            # GELU(0)=0
            self.assertLess(min(lut), 0)                             # negative lobe kept

    def test_exp_table(self):
        s = _export_scales()
        arg = s["l0.scores"]
        clamp = min(int(math.ceil(30.0 / arg)) + 1, 2 * 511)
        self.assertEqual(clamp, int(c.read_data_file("exp_clamp_l0.txt")[0]))
        lut, _ = build_exp_lut(arg, clamp, 15)
        self.assertEqual(lut.tolist(), H["exp_l0"])
        self.assertEqual(H["exp_l0"][clamp], 1 << 15)                # exp(0)
        self.assertEqual(H["exp_l0"][0], 0)                          # exp(-30) underflows
        self.assertTrue(np.all(np.diff(H["exp_l0"]) >= 0))
        d = np.arange(-clamp, 1)
        self.assertLessEqual(np.abs(np.array(H["exp_l0"]) / 2 ** 15 - np.exp(d * arg)).max(),
                             0.5 / 2 ** 15 + 1e-12)


class LutLookupVsHlsTests(HarnessMixin, unittest.TestCase):
    def test_gelu_lookup_and_clamp(self):
        qs = list(range(-600, 601)) + [-(1 << 20), 1 << 20]
        for cmd, name in (("gelu", "gelu_l0"), ("gelc", "gelu_cls")):
            lut = H[name]
            want = [str(c.gelu_lut_lookup(q, lut)) for q in qs]
            ref = gelu_lookup(np.array(qs), np.array(lut), 10).tolist()
            self.assertEqual([str(v) for v in ref], want)           # Python ref == oracle
            for k in self.kinds():
                self.assertEqual(run_harness(k, [f"{cmd} {q}" for q in qs]), want, (name, k))

    def test_rsqrt_bit_exact_and_accuracy(self):
        vs = list(range(-3, 70000))
        for e in range(1, 31):
            vs += [(1 << e) - 1, 1 << e, (1 << e) + 1]
        rng = np.random.default_rng(4)
        vs += [int(v) for v in rng.integers(1, (1 << 31) - 1, 5000)]
        want = [c.rsqrt_fxp_int(v, RSQRT_LUT) for v in vs]
        py, frac = rsqrt_fxp(np.array(vs, dtype=np.int64), build_rsqrt_lut())
        self.assertEqual(py.tolist(), want)                        # float-log2 ref == loop oracle
        self.assertEqual(RSQRT_LUT.tolist(), build_rsqrt_lut().tolist())
        for k in self.kinds():
            self.assertEqual(run_harness(k, [f"rsq {v}" for v in vs]), [str(w) for w in want], k)
        v = np.array(vs, dtype=np.float64)
        ok = v >= 1
        rel = np.abs(np.array(want)[ok] / 2 ** frac - 1 / np.sqrt(v[ok])) * np.sqrt(v[ok])
        # 8-bit mantissa + truncating shifts: <1% for moderate v; the integer
        # result has only ~15-e/2 bits left for very large v (approximation only)
        self.assertLess(rel[v[ok] < (1 << 16)].max(), 0.01)
        self.assertEqual(want[:4], [1 << 15] * 4)                  # v<=0 and v=1 -> 1.0

    def test_softmax_oracle_vs_python_reference(self):
        import src.quantize.integer_model as im
        s = _export_scales()
        clamp = int(c.read_data_file("exp_clamp_l0.txt")[0])
        lut = np.array(H["exp_l0"])
        saved = (im.ACT_BITS, im.W_BITS, im.ATTN_BITS)
        im.ACT_BITS = im.W_BITS = im.ATTN_BITS = 10
        try:
            m = im.IntegerTransformer.__new__(im.IntegerTransformer)
            m.exp_lut = {0: (lut, clamp)}
            m._cur_layer = 0
            rng = np.random.default_rng(5)
            rows = [np.zeros(SEQ, int), np.full(SEQ, 511), np.full(SEQ, -511),
                    np.r_[511, np.full(SEQ - 1, -511)], np.r_[0, np.full(SEQ - 1, -clamp)],
                    np.r_[5, 5, np.full(SEQ - 2, -300)]]
            rows += [rng.integers(-511, 512, SEQ) for _ in range(50)]
            rows += [rng.integers(-20, 21, SEQ) for _ in range(50)]
            for r in rows:
                ref = m._softmax(r[None, None, None, :].astype(np.int64), None)[0, 0, 0]
                ora = c.softmax_q(r.tolist(), lut, clamp)
                self.assertEqual(ref.tolist(), ora)
                # approximation vs float softmax: one LUT code + one output code
                fl = np.exp((r - r.max()) * s["l0.scores"]); fl /= fl.sum()
                self.assertLessEqual(np.abs(np.array(ora) / 511 - fl).max(), 2.0 / 511)
        finally:
            im.ACT_BITS, im.W_BITS, im.ATTN_BITS = saved
        eq = c.softmax_q([7] * SEQ, lut, clamp)
        self.assertEqual(set(eq), {2})                 # round(511/256)=2: sum 512, not 511
        one = c.softmax_q([511] + [-511] * (SEQ - 1), lut, clamp)
        self.assertEqual(one[0], 511)
        self.assertEqual(sum(one[1:]), 0)

    def test_attention_head_bit_exact(self):
        mqs, mqc = H["M_l0_scores_q40"][0], H["M_l0_ctx_q40"][0]
        clamp = int(c.read_data_file("exp_clamp_l0.txt")[0])
        lut = H["exp_l0"]
        rng = np.random.default_rng(6)
        cases = []
        for scale in (3, 60, 511):
            for _ in range(3):
                cases.append((rng.integers(-scale, scale + 1, HD),
                              rng.integers(-scale, scale + 1, (SEQ, HD)),
                              rng.integers(-511, 512, (SEQ, HD))))
        cases.append((np.full(HD, 511), np.full((SEQ, HD), 511), np.full((SEQ, HD), -511)))
        cases.append((np.zeros(HD, int), np.zeros((SEQ, HD), int), np.full((SEQ, HD), 511)))
        lines, want = [], []
        for q, k, v in cases:
            lines.append("head " + " ".join(map(str, np.r_[q, k.ravel(), v.ravel()])))
            want.append(" ".join(map(str, c.attention_head_q40(q, k, v, mqs, mqc, lut, clamp))))
        for kind in self.kinds():
            self.assertEqual(run_harness(kind, lines), want, kind)

    def test_layernorm_bit_exact_incl_zero_and_tiny_variance(self):
        rng = np.random.default_rng(7)
        rows = [np.full(D, v) for v in (0, 1, -1, 511, -511, 37)]            # var = 0
        for v in (0, 100, -511):                                             # var tiny
            r = np.full(D, v); r[0] += 1 if v < 511 else -1; rows.append(r)
        rows.append(np.r_[np.full(D // 2, 511), np.full(D // 2, -511)])      # max var
        rows += [rng.integers(-sc, sc + 1, D) for sc in (1, 2, 10, 100, 511) for _ in range(40)]
        for cmd, n in (("ln1", "l0_ln1"), ("ln2", "l0_ln2"), ("lnc", "cls_ln")):
            gq, bq = H[f"ln_gm_{n}_q40"], H[f"ln_bt_{n}_q40"]
            epsq = int(c.read_data_file(f"ln_epsq_{n.replace('l0_', 'l0.')}.txt")[0])
            self.assertEqual(epsq, 0)    # LN_EPS/s_in^2 rounds to 0: v<1 -> 1 is the guard
            want = [" ".join(map(str, c.layernorm_q40(r, gq, bq, epsq, RSQRT_LUT))) for r in rows]
            zero = c.layernorm_q40(rows[0], gq, bq, epsq, RSQRT_LUT)
            self.assertEqual(zero, [c.clamp_act(c.div_round_even(b, c.Q40_ONE)) for b in bq])
            for k in self.kinds():
                self.assertEqual(run_harness(k, [cmd + " " + " ".join(map(str, r)) for r in rows]),
                                 want, (n, k))


# ---------------------------------------------------------------------------
# D. 4:1 integer aggregation
# ---------------------------------------------------------------------------

class AggregationTests(unittest.TestCase):
    def test_round_half_away_from_zero_all_group_sums(self):
        # every reachable group sum s in [-2044, 2044], as 4 valid INT10 codes
        for s in range(-2044, 2045):
            g = [s // 4] * 4
            g[0] += s - sum(g)
            if abs(g[0]) > 511:
                g = [int(np.sign(s)) * 511] * 3 + [s - 3 * int(np.sign(s)) * 511]
            arr = np.array(g, dtype=np.int64).reshape(4, 1)
            want = int(math.copysign(math.floor(abs(s) / 4 + 0.5), s)) if s else 0
            self.assertEqual(int(aggregate_round_div(arr, 4)[0, 0]), want, s)
            self.assertEqual(c.aggregate4(g), want, s)

    def test_ties_differ_from_half_even(self):
        for s, away in ((2, 1), (-2, -1), (6, 2), (-6, -2), (10, 3), (-10, -3), (2042, 511)):
            self.assertEqual(c.aggregate4([s, 0, 0, 0]) if abs(s) <= 511
                             else c.aggregate4([511, 511, 511, s - 1533]), away)
        self.assertNotEqual(c.aggregate4([10, 0, 0, 0]), int(np.round(10 / 4)))  # 3 vs 2

    def test_boundaries(self):
        self.assertEqual(c.aggregate4([511] * 4), 511)
        self.assertEqual(c.aggregate4([-511] * 4), -511)
        self.assertEqual(c.aggregate4([511, 511, 511, 510]), 511)
        self.assertEqual(c.aggregate4([-511, -511, -511, -510]), -511)


@unittest.skipUnless(CHECKPOINT.exists(), "SEQ=256 checkpoint not present")
class EndToEndTopTests(HarnessMixin, unittest.TestCase):
    """transformer_top (incl. its inline 4:1 aggregation) vs the Python reference."""

    @classmethod
    def setUpClass(cls):
        import torch
        import src.quantize.integer_model as im
        from src.quantize.evaluate_integer_bits import set_bits
        cls.saved = (im.ACT_BITS, im.W_BITS, im.ATTN_BITS)
        ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        state = {k: v.cpu().numpy() for k, v in ck["model_state_dict"].items()}
        set_bits(10)
        cls.scales = _export_scales()
        cfg = {"input_dim": 8, "seq_len": 256, "num_classes": 4, "d_model": 32, "nhead": 4,
               "num_layers": 1, "dim_feedforward": 64, "dropout": 0.1}
        cls.model = im.IntegerTransformer(state, cfg, cls.scales)

    @classmethod
    def tearDownClass(cls):
        import src.quantize.integer_model as im
        im.ACT_BITS, im.W_BITS, im.ATTN_BITS = cls.saved

    def _ref(self, w1024, agg=aggregate_round_div):
        s = self.scales["input"]
        return self.model.infer_int(agg(w1024[None], 4).astype(np.float64) * s)[0]

    def test_tie_heavy_windows(self):
        rng = np.random.default_rng(8)
        wins = []
        for amp in (2, 40, 511):
            base = rng.integers(-amp, amp + 1, (256, 8))
            w = np.repeat(base, 4, axis=0)
            w[1::4] += np.where(rng.random((256, 8)) < 0.5, 1, -1)   # sum = 4b +-1 +-1
            w[2::4] += np.where(rng.random((256, 8)) < 0.5, 1, -1)
            wins.append(np.clip(w, -511, 511))
        wins.append(np.full((1024, 8), 511))
        wins.append(np.full((1024, 8), -511))
        ties = sum(int((np.abs(w.reshape(256, 4, 8).sum(1)) % 4 == 2).sum()) for w in wins)
        self.assertGreater(ties, 1000)
        lines = ["top " + " ".join(map(str, w.ravel())) for w in wins]
        want = [" ".join(map(str, self._ref(w))) for w in wins]
        for k in self.kinds():
            self.assertEqual(run_harness(k, lines), want, k)

    def test_ingress_golden_vectors(self):
        rows = (INGRESS_GOLDEN / "golden_in.txt").read_text().split("\n")[1:]
        exp = [l.split() for l in (INGRESS_GOLDEN / "golden_expected.txt").read_text().split("\n")
               if l and not l.startswith("#")]
        lines = ["top " + r for r in rows if r.strip()]
        want = [" ".join(e[3:]) for e in exp]
        for k in self.kinds():
            self.assertEqual(run_harness(k, lines), want, k)
        for r, e in zip(rows, exp):
            w = np.array(r.split(), dtype=np.int64).reshape(1024, 8)
            self.assertEqual(" ".join(map(str, self._ref(w))), " ".join(e[3:]))


if __name__ == "__main__":
    unittest.main()

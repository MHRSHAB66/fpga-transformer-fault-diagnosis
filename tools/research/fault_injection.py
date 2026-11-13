"""B01: deterministic logical fault-injection pilot on the INT10 reference.

Logical emulation of single-bit corruption in the canonical INT10 datapath; NOT
radiation testing, natural upset rates or configuration-memory coverage. The
reference is the research INT10 model (tools/research/precision_ablation.py),
which equals the canonical model logit for logit (A02 selfcheck); with no
injection this harness must reproduce the canonical logits exactly (checked on
every sampled window before the campaign).

Stage groups (tensor, hardware type -> width):
  G1 input      token codes entering the projection     act_t  10 bit
  G2 proj_acc   projection accumulators before requant   acc_t  32 bit
  G3 qkv        Q/K/V codes                              act_t  10 bit
  G4 scores     attention score codes                    act_t  10 bit
  G5 prob       attention probability codes              act_t  10 bit
  G6 ctx_acc    context accumulators before requant      acc_t  32 bit
  G7 ffn_acc    first FFN layer accumulators             acc_t  32 bit
  G8 logits     output logit codes                       act_t  10 bit
Bit bands: lowest, middle and highest third of the type's bits. One transient
flip per event (one element, one bit, one window). Persistent weight-read
corruption (W) and vote-state corruption (V) are separate campaigns.

Candidate checks (evaluated on the corrupted tensors, not acting on outputs):
  C1 code range     any 10-bit code outside the legal [-511, 511]
  C2 proven range   accumulator outside its proven legal bound (A03 bounds)
  C3 prob row sum   attention row sum outside [511 - 128, 511 + 128]

    python -m tools.research.fault_injection campaign --out artifacts/research/B01_20261113
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

GROUPS = {"G1_input": 10, "G2_proj_acc": 32, "G3_qkv": 10, "G4_scores": 10,
          "G5_prob": 10, "G6_ctx_acc": 32, "G7_ffn_acc": 32, "G8_logits": 10}
SEED = 20261113
N_WINDOWS, EVENTS = 256, 128


def bands(width):
    t = width // 3
    return {"low": (0, t), "mid": (t, 2 * t), "high": (2 * t, width)}


def flip(value, bit, width):
    """Flip one bit of a width-bit two's-complement value."""
    mask = (1 << width) - 1
    u = (int(value) & mask) ^ (1 << bit)
    return u - (1 << width) if u >> (width - 1) else u


class Injector:
    """Forward pass of one window with an optional single-bit flip and range taps."""

    def __init__(self, model):
        import src.quantize.integer_model as im
        self.m, self.im = model, im
        self.bounds = self._proven_bounds()

    def _proven_bounds(self):
        """Legal accumulator bounds: 511 * sum|W_row| + |bias| (inputs clamped to +-511)."""
        m, s = self.m, self.m.scales
        b = {}
        for name, s_in in (("proj", s["input"]), ("l0.ff0", s["l0.ln1"])):
            qb = np.round(m.bias[name] / (s_in * m.sW[name])).astype(np.int64)
            b[name] = int((511 * np.abs(m.qW[name]).sum(axis=1) + np.abs(qb)).max())
        b["ctx"] = (511 + 128) * 511   # probability row sum <= 511 + S/2, |v| <= 511
        return b

    def forward(self, x_real, inj=None):
        """inj = (group, flat_index, bit) or None. Returns logits, taps dict."""
        m, im = self.m, self.im
        from src.quantize.integer_ops import quantize_sym, requant_float, gelu_lookup
        from tools.research.precision_ablation import _exact_matmul
        s = m.scales
        taps = {"code_range": False, "acc_range": False, "prob_rowsum": False, "perturbed": False}

        def hit(group, arr, width):
            if inj is None or inj[0] != group:
                return arr
            a = arr.copy()
            flat = a.reshape(-1)
            idx = inj[1] % flat.size
            new = flip(flat[idx], inj[2], width)
            taps["perturbed"] = new != flat[idx]
            flat[idx] = new
            return a

        def code_check(a):
            if np.abs(a).max() > 511:
                taps["code_range"] = True

        def linear_acc(q_in, name, s_in):
            qb = np.round(m.bias[name] / (s_in * m.sW[name])).astype(np.int64)
            return _exact_matmul(q_in, m.qW[name].T) + qb

        def requant(acc, name, s_in, s_out):
            return requant_float(acc, (s_in * m.sW[name]) / s_out, im.ACT_BITS)

        q = quantize_sym(x_real, s["input"], im.ACT_BITS)
        q = hit("G1_input", q, 10); code_check(q)
        acc = hit("G2_proj_acc", linear_acc(q, "proj", s["input"]), 32)
        if np.abs(acc).max() > self.bounds["proj"]:
            taps["acc_range"] = True
        x = requant(acc, "proj", s["input"], s["embed"])
        q_pe = np.round(m.pe / s["embed"]).astype(np.int64)
        x = m._add(x, s["embed"], q_pe[None], s["embed"], s["embed"])

        pos = im.layer_positions(0)
        qkv = []
        for nm, key in (("l0.wq", "q"), ("l0.wk", "k"), ("l0.wv", "v")):
            qkv.append(requant(linear_acc(x, nm, s["embed"]), nm, s["embed"], s[pos[key]]))
        qkv = hit("G3_qkv", np.stack(qkv), 10); code_check(qkv)
        qh, kh, vh = (m._split(t) for t in qkv)
        M = (s[pos["q"]] * s[pos["k"]]) / (math.sqrt(m.head_dim) * s[pos["scores"]])
        sc = requant_float(_exact_matmul(qh, kh.transpose(0, 1, 3, 2)), M, im.ACT_BITS)
        sc = hit("G4_scores", sc, 10); code_check(sc)
        m._cur_layer = 0
        diag = m.diag
        m.diag = None
        pa = m._softmax(sc, None)
        m.diag = diag
        pa = hit("G5_prob", pa, 10); code_check(pa)
        rs = pa.sum(axis=-1)
        if (rs < 511 - 128).any() or (rs > 511 + 128).any():
            taps["prob_rowsum"] = True
        cacc = hit("G6_ctx_acc", _exact_matmul(pa, vh), 32)
        if np.abs(cacc).max() > self.bounds["ctx"]:
            taps["acc_range"] = True
        q_ctx = requant_float(cacc, (1.0 / m.p_qmax * s[pos["v"]]) / s[pos["ctx"]], im.ACT_BITS)
        attn = requant(linear_acc(m._merge(q_ctx), "l0.out", s[pos["ctx"]]), "l0.out", s[pos["ctx"]], s[pos["attn_out"]])
        res1 = m._add(x, s["embed"], attn, s[pos["attn_out"]], s[pos["res1"]])
        pfx = "encoder_layers.0."
        ln1 = m._layernorm(res1, s[pos["res1"]], m.f[pfx + "attention_norm.weight"], m.f[pfx + "attention_norm.bias"], s[pos["ln1"]])
        facc = hit("G7_ffn_acc", linear_acc(ln1, "l0.ff0", s[pos["ln1"]]), 32)
        if np.abs(facc).max() > self.bounds["l0.ff0"]:
            taps["acc_range"] = True
        ff1 = requant(facc, "l0.ff0", s[pos["ln1"]], s[pos["ff1"]])
        ff_act = gelu_lookup(ff1, m.gelu_lut[0], im.ACT_BITS)
        ff2 = requant(linear_acc(ff_act, "l0.ff3", s[pos["ff_act"]]), "l0.ff3", s[pos["ff_act"]], s[pos["ff2"]])
        res2 = m._add(ln1, s[pos["ln1"]], ff2, s[pos["ff2"]], s[pos["res2"]])
        x = m._layernorm(res2, s[pos["res2"]], m.f[pfx + "feedforward_norm.weight"], m.f[pfx + "feedforward_norm.bias"], s[pos["ln2"]])
        pooled = np.sum(x, axis=1)
        q_pool = requant_float(pooled, s[pos["ln2"]] / (x.shape[1] * s["pool"]), im.ACT_BITS)
        cls_ln = m._layernorm(q_pool, s["pool"], m.f["classifier.0.weight"], m.f["classifier.0.bias"], s["cls_ln"])
        c1 = requant(linear_acc(cls_ln, "cls1", s["cls_ln"]), "cls1", s["cls_ln"], s["cls1"])
        c_act = gelu_lookup(c1, m.cls_gelu_lut, im.ACT_BITS)
        logits = requant(linear_acc(c_act, "cls4", s["cls_act"]), "cls4", s["cls_act"], s["logits"])
        logits = hit("G8_logits", logits, 10); code_check(logits)
        return logits, taps


def argmax_first(l):
    return int(np.argmax(l))


def sample_windows(rng):
    """256 validation windows: 64 per class, spread over as many files as possible."""
    from tools.research.precision_ablation import split_rows
    _, val_rows, y, files = split_rows()
    pick = []
    for c in range(4):
        cand = np.flatnonzero(y == c)
        fs = rng.permutation(np.unique(files[cand]))
        per_file = {f: rng.permutation(cand[files[cand] == f]) for f in fs}
        k = 0
        while sum(1 for p in pick if y[p] == c) < N_WINDOWS // 4:
            f = fs[k % len(fs)]
            used = sum(1 for p in pick if files[p] == f)
            if used < len(per_file[f]):
                pick.append(int(per_file[f][used]))
            k += 1
    pick = np.array(sorted(pick))
    return val_rows[pick], y[pick], files[pick]


def cmd_campaign(args):
    from tools.research.precision_ablation import CACHE, load_state, make_model
    import src.quantize.integer_model as im
    rng = np.random.default_rng(SEED)
    canon = json.loads((Path(__file__).resolve().parents[2] / "artifacts/quantized/f_seq256_int10/integer_config.json").read_text())["scales"]
    state = load_state()
    model = make_model(state, 10, 10, 10, False, scales=canon, diag=True)
    inj = Injector(model)
    x = np.load(CACHE / "x.npy", mmap_mode="r")
    rows, y, files = sample_windows(rng)
    xw = np.ascontiguousarray(x[rows]).astype(np.float64)
    # 1) injection disabled == canonical model, logit for logit
    clean = np.stack([inj.forward(xw[i:i + 1])[0][0] for i in range(len(rows))])
    canonical = np.concatenate([model.infer_int(xw[i:i + 16]) for i in range(0, len(rows), 16)])
    assert (clean == canonical).all(), "injection-disabled path differs from the canonical model"
    clean_cls = clean.argmax(1)
    # tensor sizes per group (one forward with taps of shapes)
    sizes = {"G1_input": 256 * 8, "G2_proj_acc": 256 * 32, "G3_qkv": 3 * 256 * 32, "G4_scores": 4 * 256 * 256,
             "G5_prob": 4 * 256 * 256, "G6_ctx_acc": 4 * 256 * 8, "G7_ffn_acc": 256 * 64, "G8_logits": 4}
    records = []
    t0 = time.time()
    for g, width in GROUPS.items():
        for band, (lo, hi) in bands(width).items():
            for e in range(EVENTS):
                w = int(rng.integers(len(rows)))
                idx = int(rng.integers(sizes[g]))
                bit = int(rng.integers(lo, hi))
                logits, taps = inj.forward(xw[w:w + 1], (g, idx, bit))
                cls = argmax_first(logits[0])
                records.append({"group": g, "band": band, "event": e, "window_row": int(rows[w]), "label": int(y[w]),
                                "element": idx, "bit": bit, "perturbed": bool(taps["perturbed"]),
                                "logits_changed": bool((logits[0] != clean[w]).any()),
                                "class_changed": cls != int(clean_cls[w]), "wrong_vs_label": cls != int(y[w]),
                                "C1_code_range": taps["code_range"], "C2_acc_range": taps["acc_range"],
                                "C3_prob_rowsum": taps["prob_rowsum"]})
        print(g, "done %.0f s" % (time.time() - t0), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "injections.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    summary = summarize(records)
    summary.update({"id": "B01", "seed": SEED, "windows": int(len(rows)), "events_per_group_band": EVENTS,
                    "injection_disabled_equals_canonical": True, "clean_accuracy_on_sample": float((clean_cls == y).mean()),
                    "proven_bounds": inj.bounds, "seconds": round(time.time() - t0, 1),
                    "scope": "logical single-bit emulation on the INT10 reference; not radiation, upset rate or config-memory coverage"})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["by_group"], indent=1))


def summarize(records):
    out = {"by_group": {}, "by_group_band": {}}
    def agg(rs):
        n = len(rs)
        det = [r["C1_code_range"] or r["C2_acc_range"] or r["C3_prob_rowsum"] for r in rs]
        cc = [r["class_changed"] for r in rs]
        return {"requested": n, "perturbed": sum(r["perturbed"] for r in rs),
                "logits_changed": sum(r["logits_changed"] for r in rs), "class_changed": sum(cc),
                "detected_any": sum(det), "detected_C1": sum(r["C1_code_range"] for r in rs),
                "detected_C2": sum(r["C2_acc_range"] for r in rs), "detected_C3": sum(r["C3_prob_rowsum"] for r in rs),
                "silent_class_corruption": sum(c and not d for c, d in zip(cc, det)),
                "detected_without_class_change": sum(d and not c for c, d in zip(cc, det))}
    for g in GROUPS:
        out["by_group"][g] = agg([r for r in records if r["group"] == g])
        for b in ("low", "mid", "high"):
            out["by_group_band"][g + "/" + b] = agg([r for r in records if r["group"] == g and r["band"] == b])
    total_silent = sum(v["silent_class_corruption"] for v in out["by_group"].values())
    top2 = sorted(out["by_group"].items(), key=lambda kv: -kv[1]["silent_class_corruption"])[:2]
    out["concentration"] = {"total_silent_class_corruptions": total_silent,
                            "top2_groups": [k for k, _ in top2],
                            "top2_share": (sum(v["silent_class_corruption"] for _, v in top2) / total_silent) if total_silent else None}
    out["gate_B01_continue_to_B02"] = bool(out["concentration"]["top2_share"] is not None and out["concentration"]["top2_share"] >= 0.6)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("campaign")
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cmd_campaign(args)


if __name__ == "__main__":
    main()

"""Independent offline review of saved evidence; never accesses the board."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    result = {"reviewed_utc": datetime.now(timezone.utc).isoformat(), "board_accessed": False}
    frozen = load("artifacts/research/F00_20261113/frozen_baseline_manifest.json")
    identity = {}
    for category in ("sources", "model_and_export", "hls_sources", "golden", "reference_outputs"):
        checks = []
        for e in frozen[category]:
            p = ROOT / e["path"]
            checks.append({"path": e["path"], "exists": p.is_file(),
                           "matches_frozen": p.is_file() and sha(p) == e.get("sha256")})
        identity[category] = {"checked": len(checks), "mismatches": [x for x in checks if not x["matches_frozen"]]}
    result["frozen_identity"] = identity

    manifest = load("artifacts/manifests/artifact_manifest.json")
    selected = [a for a in manifest["artifacts"] if a["path"].startswith(("artifacts/research/", "artifacts/deploy/research_"))]
    checks, skipped = [], []
    for e in selected:
        p = ROOT / e["path"]
        if p.is_file() and p.stat().st_size > 64 * (1 << 20):
            skipped.append(e["path"])
            continue
        checks.append({"path": e["path"], "exists": p.is_file(),
                       "matches_manifest": p.is_file() and sha(p) == e.get("sha256")})
    result["research_artifact_hashes"] = {"checked": len(checks), "skipped_large": skipped,
                                           "mismatches": [x for x in checks if not x["matches_manifest"]]}
    bundles = {}
    for d in sorted((ROOT / "artifacts/deploy").glob("research_*_20261113")):
        entries = []
        for line in (d / "MANIFEST.sha256").read_text().splitlines():
            digest, rel = line.split(" *", 1)
            p = d / rel.removeprefix("./")
            entries.append({"path": rel, "matches": p.is_file() and sha(p) == digest})
        bundles[d.name] = {"files": len(entries), "mismatches": [x for x in entries if not x["matches"]]}
    result["deployment_bundle_hashes"] = bundles

    canonical = np.loadtxt(ROOT / "artifacts/results/onchip_preprocess_20261109/csim_raw_argmax/csim_logits.txt", dtype=np.int64)
    labels = np.loadtxt(ROOT / "artifacts/results/fpga_rawinput_testset_20261109/labels.txt", dtype=np.int64)
    fulltests = {}
    paths = {"A05": ("144220", canonical), "B02": ("184417", canonical),
             "A06": ("213019", np.loadtxt(ROOT / "artifacts/research/A06_20261113/csim_p8u_fulltest/csim_logits.txt", dtype=np.int64))}
    for name, (tag, reference) in paths.items():
        d = ROOT / ("artifacts/results/raw_core_board_fulltest_20261113_" + tag)
        log = d / "fulltest.out"
        rows, bad_ref, duplicates, seen, flags = 0, 0, 0, set(), []
        for line in log.read_text().splitlines():
            if not line.startswith("R "):
                continue
            kv = dict(x.split("=", 1) for x in line.split()[1:])
            idx = int(kv["idx"])
            duplicates += int(idx in seen)
            seen.add(idx)
            lg = [int(x) for x in kv["logits"].split(",")]
            bad_ref += int(lg != reference[idx, 1:5].tolist() or int(kv["cls"]) != reference[idx, 5])
            if "integrity" in kv:
                flags.append(int(kv["integrity"]))
            rows += 1
        fulltests[name] = {"rows": rows, "missing_indices": len(set(range(len(reference))) - seen),
                           "duplicates": duplicates, "reference_mismatches": bad_ref,
                           "integrity_rows": len(flags), "nonzero_integrity": sum(v != 0 for v in flags),
                           "accuracy_recomputed": float((reference[:, 5] == labels).mean())}
    result["fulltests_recomputed_from_board_logs"] = fulltests

    rt = {}
    rt_paths = {"endurance": "20261111_173322", "A05": "20261113_151033",
                "B02": "20261113_191441", "A06": "20261113_220122"}
    for name, tag in rt_paths.items():
        d = ROOT / ("artifacts/results/raw_core_board_rt_" + tag)
        s = json.loads((d / "out/summary.json").read_text())
        p = d / "out/rt_raw.npz"
        if not p.exists():
            rt[name] = {"raw_trace_missing": True}
            continue
        z = np.load(p)
        ref = paths["A06"][1] if name == "A06" else canonical
        w = z["window_index"]
        response = 1000 * (z["done"] - z["release"])
        accepted = response[s["warmup"]:]
        rt[name] = {"trace_rows": len(w), "accepted": len(accepted),
                    "recomputed_misses": int((accepted > s["period_ms"]).sum()),
                    "recomputed_max_ms": float(accepted.max()),
                    "logit_mismatches": int((z["logits"].astype(np.int64) != ref[w, 1:5]).any(1).sum()),
                    "class_mismatches": int((z["cls"] != ref[w, 5]).sum()),
                    "summary_count_matches": int(s["executed"]) == len(w), "npz_keys": list(z.files)}
    result["rt_traces_recomputed"] = rt

    stress = {}
    for d in sorted((ROOT / "artifacts/research/S01_20261113").iterdir()):
        p = d / "out/rt_raw.npz"
        if not p.exists():
            continue
        z = np.load(p)
        s = json.loads((d / "out/summary.json").read_text())
        w = z["window_index"]
        response = 1000 * (z["done"] - z["release"])
        score = response[s["warmup"]:]
        stress[d.name] = {"rows": len(w), "scored": len(score),
                           "recomputed_misses": int((score > s["period_ms"]).sum()),
                           "recomputed_max_ms": float(score.max()),
                           "logit_mismatches": int((z["logits"] != canonical[w, 1:5]).any(1).sum()),
                           "class_mismatches": int((z["cls"] != canonical[w, 5]).sum())}
    result["S01_traces_recomputed"] = stress

    from tools.research.precision_ablation import split_rows
    _, valrows, valy, valfiles = split_rows()
    a02 = load("artifacts/research/A02_20261113/summary.json")
    precision = {}
    for name, summary in a02["variants"].items():
        z = np.load(ROOT / ("artifacts/research/A02_20261113/" + name + "/val_outputs.npz"))
        pred = z["logits"].argmax(1)
        cm = np.bincount(4 * valy + pred, minlength=16).reshape(4, 4)
        f1 = np.divide(2 * cm.diagonal(), cm.sum(0) + cm.sum(1), out=np.zeros(4), where=(cm.sum(0) + cm.sum(1)) != 0)
        precision[name] = {"row_identity": bool(np.array_equal(z["rows"], valrows)), "windows": len(pred),
                            "files": len(np.unique(valfiles)), "accuracy": float((pred == valy).mean()),
                            "macro_f1": float(f1.mean()), "zero_attention_rows": int(z["zero_rows"].sum()),
                            "matches_saved_metrics": bool(abs(float(f1.mean()) - summary["macro_f1"]) < 1e-12)}
    p8 = np.load(ROOT / "artifacts/research/A02_20261113/p8u/val_outputs.npz")
    p9 = np.load(ROOT / "artifacts/research/A02_20261113/p9s/val_outputs.npz")
    result["A02_metrics_recomputed"] = precision
    result["A02_p8u_equals_p9s_logits"] = bool(np.array_equal(p8["logits"], p9["logits"]))

    from tools.research.q01_quality import evaluate
    cal = load("artifacts/calibration/conf_temperature_v1/temperature.json")
    example = np.array([[20, 10, 0, -10]], dtype=np.int64)
    actual = float(evaluate(example, np.array([0]))[2][0])
    v = example * cal["s_logits"] / cal["T"]
    prob = np.exp(v - v.max(axis=1, keepdims=True))
    expected = float((prob / prob.sum(axis=1, keepdims=True)).max())
    q = load("artifacts/research/Q01_20261113/summary.json")
    result["Q01_confidence_unit_check"] = {"integer_logits": example.tolist(), "q01_confidence": actual,
                                            "calibration_contract_confidence": expected,
                                            "missing_scale": cal["s_logits"], "summary": q["report"]}

    from tools.research.temporal_policies import policy_stream
    clean = ROOT / "artifacts/results/raw_core_board_fulltest_20261111_130927/fulltest.out"
    cls, logits, idx = [], [], []
    for line in clean.read_text().splitlines():
        if line.startswith("R "):
            kv = dict(x.split("=", 1) for x in line.split()[1:])
            idx.append(int(kv["idx"]))
            cls.append(int(kv["cls"]))
            logits.append([int(x) for x in kv["logits"].split(",")])
    order = np.argsort(idx)
    cls, logits = np.asarray(cls)[order], np.asarray(logits)[order]
    y = labels[np.asarray(idx)[order]]
    t = load("artifacts/research/T01_20261113/summary.json")
    chosen = sorted(t["short_events_test_clean_synthetic"])
    rng = np.random.default_rng(20261113)
    idx_by = {c: np.flatnonzero(y == c) for c in range(4)}
    evchecks = {}
    for policy in chosen:
        fam, _, parameter = policy.partition("_")
        parameter = None if not parameter else (float(parameter) if fam == "leaky" else int(parameter))
        per = {}
        for length in (2, 4, 8, 16):
            invalid, detected_during, detected_after, legacy_hits, fingerprints = 0, 0, 0, 0, []
            for _ in range(200):
                a, b = rng.choice(4, 2, replace=False)
                sa, sb = rng.choice(idx_by[a][:-60]), rng.choice(idx_by[b][:-length])
                seq = np.concatenate([np.arange(sa, sa + 40), np.arange(sb, sb + length), np.arange(sa + 40, sa + 60)])
                expected_y = np.array([a] * 40 + [b] * length + [a] * 20)
                invalid += int(not np.array_equal(y[seq], expected_y))
                dec = policy_stream(fam, parameter, cls[seq], logits[seq])
                hit = np.flatnonzero(dec[40:40 + length + 20] == b)
                if len(hit):
                    legacy_hits += 1
                    detected_during += int(hit[0] < length)
                    detected_after += int(hit[0] >= length)
                fingerprints.append([int(sa), int(sb), int(a), int(b)])
            per[str(length)] = {"malformed_label_sequences": invalid, "events": 200,
                                "detected_during": detected_during, "detected_only_after_event": detected_after,
                                "legacy_detected": legacy_hits,
                                "saved_fraction_reproduced": abs(legacy_hits / 200 - t["short_events_test_clean_synthetic"][policy][str(length)]["detected_fraction"]) < 1e-12,
                                "event_manifest_sha256": hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest()}
        evchecks[policy] = per
    tied_logits = np.zeros((1001, 4), dtype=np.int64)
    tied_logits[0, 1] = 100
    tie_classes = tied_logits.argmax(1)
    dec = policy_stream("evid", 64, tie_classes, tied_logits)
    result["T01_checks"] = {"short_events": evchecks,
                              "bounded_state_does_not_bound_horizon": {"windows_after_change": 1000,
                                                                         "raw_classes_consistent_with_logits": bool(np.array_equal(tie_classes, tied_logits.argmax(1))),
                                                                         "ever_reaches_new_class0": bool((dec[1:] == 0).any())}}

    bdir = ROOT / "artifacts/research/B03_20261113"
    man = np.loadtxt(bdir / "manifest.txt", dtype=np.int64)
    cm = np.loadtxt(bdir / "cmodel_rows_lp64.txt", dtype=np.int64)
    br = []
    for line in (bdir / "board_rows.txt").read_text().splitlines():
        kv = dict(x.split("=", 1) for x in line.split()[1:])
        br.append({k: int(kv[k]) for k in ("row", "idx", "site", "word", "bit", "cls", "integrity")} |
                  {"logits": [int(x) for x in kv["logits"].split(",")]})
    map_bad = sum(r["row"] != i or [r[k] for k in ("idx", "site", "word", "bit")] != man[i].tolist() for i, r in enumerate(br))
    eq_bad = sum(r["logits"] != cm[i, 1:5].tolist() or r["cls"] != cm[i, 5] or r["integrity"] != cm[i, 6] for i, r in enumerate(br))
    changed = [(r["cls"] != canonical[r["idx"], 5], r["integrity"] != 0) for r in br if r["site"]]
    result["B03_independent_recheck"] = {"manifest_rows": len(man), "board_rows": len(br), "cmodel_rows": len(cm),
                                         "row_mapping_errors": map_bad, "lp64_reference_errors": eq_bad,
                                         "class_changing": sum(a for a, _ in changed),
                                         "flagged_class_changing": sum(a and b for a, b in changed),
                                         "silent_class_changes": sum(a and not b for a, b in changed)}

    def encode_scalar(v):
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    (OUT / "audit_results.json").write_text(json.dumps(result, indent=2, default=encode_scalar), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("Q01_confidence_unit_check", "T01_checks")}, indent=2, default=encode_scalar))
    print("Q01 confidence:", actual, "expected:", expected)
    print("T01:", json.dumps(result["T01_checks"], indent=1))


if __name__ == "__main__":
    main()

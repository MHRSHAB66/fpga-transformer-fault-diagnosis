"""Run the desktop g++ C-model of a SEQ=256 INT10 HLS top over the EXACT input
file the physical board consumed (fpga_testset_in.i32.gz), in parallel shards.

Purpose: a same-input, same-model per-window comparison between
  (a) the physical board predictions (fpga_pred.txt), and
  (b) the synthesizable HLS C++ source compiled with g++ (functional C-model),
without touching the board or any recorded result.

Writes (new versioned dir, nothing overwritten):
    <out>/csim_logits.txt   "<idx> l0 l1 l2 l3" per window, ascending idx
    <out>/csim_run.json     binary, source, shards, wall time

Usage:
    python -m tools.run_csim_fulltest --exe tmp/csim/batch_dsp_margin.exe \
        --input artifacts/results/fpga_fulltest_20261008_141335/fpga_testset_in.i32.gz \
        --n 95404 --shards 12 --out artifacts/results/board_vs_reference_<ts>/csim_dsp_margin
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import threading
import time
from pathlib import Path

REC_BYTES = 1024 * 8 * 4


def _feed(gz_path: Path, start: int, count: int, proc: subprocess.Popen) -> None:
    with gzip.open(gz_path, "rb") as fin:
        skip = start * REC_BYTES
        while skip > 0:
            chunk = fin.read(min(skip, 64 * REC_BYTES))
            if not chunk:
                break
            skip -= len(chunk)
        left = count * REC_BYTES
        while left > 0:
            chunk = fin.read(min(left, 16 * REC_BYTES))
            if not chunk:
                break
            proc.stdin.write(chunk)
            left -= len(chunk)
    proc.stdin.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per = (args.n + args.shards - 1) // args.shards
    jobs = []
    t0 = time.time()
    for s in range(args.shards):
        start, count = s * per, min(per, args.n - s * per)
        if count <= 0:
            break
        part = args.out / f"part{s:02d}.txt"
        fout = open(part, "w")
        proc = subprocess.Popen([str(args.exe), str(start), str(count)],
                                stdin=subprocess.PIPE, stdout=fout)
        th = threading.Thread(target=_feed, args=(args.input, start, count, proc), daemon=True)
        th.start()
        jobs.append((proc, th, fout, part, start, count))
    rcs = []
    for proc, th, fout, *_ in jobs:
        th.join()
        rcs.append(proc.wait())
        fout.close()
    if any(rcs):
        raise SystemExit(f"shard failures: {rcs}")

    lines = []
    for *_, part, start, count in jobs:
        got = part.read_text().splitlines()
        if len(got) != count:
            raise SystemExit(f"{part}: expected {count} lines, got {len(got)}")
        lines.extend(got)
        part.unlink()
    (args.out / "csim_logits.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    (args.out / "csim_run.json").write_text(json.dumps({
        "kind": "desktop g++ functional C-model (NOT synthesis, NOT hardware)",
        "exe": str(args.exe), "input": str(args.input), "n": args.n,
        "shards": len(jobs), "wall_seconds": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out / 'csim_logits.txt'} ({len(lines)} windows)")


if __name__ == "__main__":
    main()

"""Collect every 2026-10-23 timing/diagnostic run into one table (no new measurement).

For each run dir under artifacts/results/fpga_timing_20261023/: wrapper result and
exit codes, preflight status, windows, match vs recorded predictions, rate,
output-interval / latency statistics, plus the number of SSH logins on the board
during the run (from a copy of the board's auth.log 'Accepted publickey' lines,
--logins). Every SSH login runs Ubuntu's login (MOTD) scripts on the board,
which costs CPU; the rate is therefore reported as a measured confound.

Usage: python -m tools.summarize_timing_20261023 --logins <logins.txt> --out <summary.json>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "results" / "fpga_timing_20261023"


def _times(run):
    ro = run.get("run_out") or []
    s = [x.split("=", 1)[1] for x in ro if x.startswith("START_UTC=")]
    e = [x.split("=", 1)[1] for x in ro if x.startswith("END_UTC=")]
    f = "%Y-%m-%dT%H:%M:%SZ"
    return (dt.datetime.strptime(s[0], f) if s else None, dt.datetime.strptime(e[0], f) if e else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logins", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    logins = []
    for line in a.logins.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            logins.append(dt.datetime.strptime("2019 " + line.strip(), "%Y %b %d %H:%M:%S"))
        except ValueError:
            pass
    rows = []
    for d in sorted(p for p in BASE.iterdir() if p.is_dir()):
        rj = d / "run.json"
        if not rj.exists():
            continue
        run = json.loads(rj.read_text(encoding="utf-8"))
        row = {"run": d.name, "result": run.get("result"), "workload_rc": run.get("workload_rc"),
               "guard_rc": run.get("guard_rc"), "post_run_process_check": run.get("post_run_process_check")}
        pf = d / "preflight.json"
        if pf.exists():
            row["preflight_all_ok"] = all(c["ok"] for c in json.loads(pf.read_text()))
        s, e = _times(run)
        if s and e:
            row["board_start_utc"], row["board_end_utc"] = s.isoformat(), e.isoformat()
            cov_end = min(e, logins[-1]) if logins else s
            n = sum(s <= x <= cov_end for x in logins)
            mins = (cov_end - s).total_seconds() / 60.0
            row["ssh_logins_during_run"] = n
            row["ssh_logins_per_min"] = round(n / mins, 1) if mins > 0 else None
            row["login_log_covers_run"] = bool(logins and logins[-1] >= e)
            row["login_rate_over_minutes"] = round(mins, 1)
        ts = d / "timing_summary.json"
        if ts.exists():
            t = json.loads(ts.read_text(encoding="utf-8"))
            row.update({k: t.get(k) for k in ("windows", "match_recorded_fpga_pred", "rate_inf_s_after_warmup",
                                              "pc_rate_inf_s_after_warmup", "summary_line")})
            for k in ("output_interval_ms", "lat_ms_field", "resp_ms_paced", "pc_receive_interval_ms",
                      "pc_extra_delay_ms_vs_min"):
                if t.get(k):
                    row[k] = {x: t[k][x] for x in ("p50", "p99", "p99.9", "max", "over_20.48ms",
                                                   "over_50ms", "over_100ms")}
        ds = sorted(d.glob("remote/out/summary.json"))
        if ds:
            dj = json.loads(ds[0].read_text(encoding="utf-8"))
            row["diag"] = {k: dj.get(k) for k in ("mode", "n", "aborted", "wall_ms", "cpu_ms", "slow_threshold_ms",
                                                  "slow_count", "slow_with_involuntary_preemption",
                                                  "slow_with_voluntary_wait", "readahead_mb", "temp_max_c")}
        rows.append(row)
    a.out.write_text(json.dumps({"login_log_last_entry": logins[-1].isoformat() if logins else None,
                                 "runs": rows}, indent=2), encoding="utf-8")
    for r in rows:
        print("%-36s %-8s logins/min=%-5s %s" % (r["run"], (r["result"] or "")[:8], r.get("ssh_logins_per_min"),
              (r.get("output_interval_ms") or {}).get("over_20.48ms", (r.get("diag") or {}).get("slow_count"))))


if __name__ == "__main__":
    main()

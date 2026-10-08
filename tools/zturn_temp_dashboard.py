#!/usr/bin/env python3
"""Real-time board-temperature dashboard for the MYIR Z-Turn (XC7Z020 / PYNQ).

A dependency-free (stdlib-only) web dashboard. It reads the Zynq XADC die
temperature over SSH and shows it live in the browser: a large colour-coded
reading, a rolling chart, and min/mean/max.

    temperature_c = (raw + offset) * scale / 1000      # XADC iio:device0

Endpoints:
    /                -> the dashboard HTML/JS page
    /api/temp        -> JSON: one fresh (SSH) temperature reading
    /api/config      -> JSON: poll interval + threshold bands

Usage (on the Windows host, same place the deploy scripts run):
    python tools/zturn_temp_dashboard.py            # serve on http://0.0.0.0:8099
    python tools/zturn_temp_dashboard.py --once      # one SSH read, print, exit
    python tools/zturn_temp_dashboard.py --http-port 9000 --interval 1.5

SSH auth reuses the dedicated key created for board automation
(~/.ssh/id_zturn); no password is stored by this tool. This is a read-only
monitor: it only `cat`s three sysfs files and never programs the FPGA.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

XADC = "/sys/bus/iio/devices/iio:device0"
RAW = XADC + "/in_temp0_raw"
OFFSET = XADC + "/in_temp0_offset"
SCALE = XADC + "/in_temp0_scale"

# Conservative safety bands (deg C), shared with the board bring-up procedure.
WARN_C = 70.0
STOP_C = 75.0
CRIT_C = 80.0

CFG = {
    "host": "192.168.2.99",
    "user": "root",
    "port": 22,
    "key": None,            # resolved to ~/.ssh/id_zturn by default
    "interval": 2.0,        # client poll seconds
    "history": 300,         # points kept in the chart
    "ssh_timeout": 8,       # per-read SSH timeout (s)
    "cache_s": 1.0,         # coalesce reads faster than this
}

_LAST = {"t": 0.0, "payload": None}     # tiny server-side cache


def _ssh_argv():
    key = CFG["key"] or os.path.join(os.path.expanduser("~"), ".ssh", "id_zturn")
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "ConnectTimeout=%d" % int(CFG["ssh_timeout"]),
        "-o", "StrictHostKeyChecking=accept-new",
        "-p", str(CFG["port"]), "-i", key,
        "%s@%s" % (CFG["user"], CFG["host"]),
        "cat %s %s %s" % (RAW, OFFSET, SCALE),
    ]


def read_temp():
    """One live reading. Returns a JSON-able dict (ok True/False)."""
    now = time.time()
    if _LAST["payload"] is not None and (now - _LAST["t"]) < CFG["cache_s"]:
        return _LAST["payload"]

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        proc = subprocess.run(
            _ssh_argv(), capture_output=True, text=True,
            timeout=CFG["ssh_timeout"] + 4, creationflags=creationflags)
    except Exception as exc:
        payload = {"ok": False, "ts": ts, "error": "ssh launch failed: %s" % exc}
        _LAST.update(t=now, payload=payload)
        return payload

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "ssh returned %d" % proc.returncode).strip()
        payload = {"ok": False, "ts": ts, "error": err[:300]}
        _LAST.update(t=now, payload=payload)
        return payload

    try:
        nums = proc.stdout.split()
        raw, offset, scale = int(nums[0]), float(nums[1]), float(nums[2])
        temp = (raw + offset) * scale / 1000.0
        payload = {"ok": True, "ts": ts, "temp_c": round(temp, 3),
                   "raw": raw, "offset": offset, "scale": scale}
    except Exception as exc:
        payload = {"ok": False, "ts": ts,
                   "error": "parse failed from %r: %s" % (proc.stdout[:120], exc)}
    _LAST.update(t=now, payload=payload)
    return payload


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Z-Turn board temperature</title>
<style>
  :root { --bg:#0f1420; --card:#1a2233; --fg:#e8eef7; --muted:#8aa0bd; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: Segoe UI, Roboto, system-ui, sans-serif;
         background:var(--bg); color:var(--fg); }
  header { padding:14px 20px; border-bottom:1px solid #243049;
           display:flex; align-items:center; gap:12px; }
  header h1 { font-size:16px; font-weight:600; margin:0; }
  #dot { width:12px; height:12px; border-radius:50%; background:#666; }
  .wrap { max-width:960px; margin:0 auto; padding:20px; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
  .card { background:var(--card); border:1px solid #243049; border-radius:12px; padding:18px; }
  .big { font-size:88px; font-weight:700; line-height:1; letter-spacing:-2px; }
  .unit { font-size:28px; color:var(--muted); }
  .status { font-size:20px; font-weight:600; margin-top:8px; }
  .muted { color:var(--muted); font-size:13px; }
  .stat { font-size:13px; color:var(--muted); }
  .stat b { color:var(--fg); font-size:18px; display:block; }
  .row { display:flex; gap:22px; margin-top:10px; }
  canvas { width:100%; height:260px; display:block; }
  .err { color:#ff6b6b; font-size:13px; margin-top:8px; min-height:16px; word-break:break-all; }
  @keyframes blink { 50% { opacity:.35; } }
  .crit { animation: blink 0.8s infinite; }
</style>
</head>
<body>
<header>
  <span id="dot"></span>
  <h1>MYIR Z-Turn &mdash; Zynq XADC die temperature (live)</h1>
  <span class="muted" id="target"></span>
</header>
<div class="wrap">
  <div class="grid">
    <div class="card">
      <div><span class="big" id="temp">--.-</span><span class="unit">&deg;C</span></div>
      <div class="status" id="status">connecting…</div>
      <div class="row">
        <div class="stat">min<b id="min">--</b></div>
        <div class="stat">mean<b id="mean">--</b></div>
        <div class="stat">max<b id="max">--</b></div>
        <div class="stat">updated<b id="ts">--</b></div>
      </div>
      <div class="err" id="err"></div>
      <div class="muted" id="bands"></div>
    </div>
    <div class="card">
      <canvas id="chart" width="900" height="260"></canvas>
    </div>
  </div>
</div>
<script>
let CFG = {interval:2000, history:300, warn:70, stop:75, crit:80};
const hist = [];          // {t, v}
let minV=null, maxV=null, sum=0, n=0;

function bandColor(v){
  if (v>=CFG.crit) return "#b00020";
  if (v>=CFG.stop) return "#ff4d4f";
  if (v>=CFG.warn) return "#ffa94d";
  return "#51cf66";
}
function bandText(v){
  if (v>=CFG.crit) return "CRITICAL (>= "+CFG.crit+"°C)";
  if (v>=CFG.stop) return "STOP (>= "+CFG.stop+"°C)";
  if (v>=CFG.warn) return "WARNING ("+CFG.warn+"–"+CFG.stop+"°C)";
  return "normal (< "+CFG.warn+"°C)";
}

function drawChart(){
  const c = document.getElementById("chart");
  const ctx = c.getContext("2d");
  const W=c.width, H=c.height, padL=42, padB=22, padT=10, padR=10;
  ctx.clearRect(0,0,W,H);
  // y-range: fixed sensible window, expand if needed
  let lo=40, hi=85;
  for(const p of hist){ if(p.v<lo) lo=Math.floor(p.v)-2; if(p.v>hi) hi=Math.ceil(p.v)+2; }
  const x0=padL, x1=W-padR, y0=H-padB, y1=padT;
  const vx = i => x0 + (x1-x0) * (hist.length<=1?0:i/(hist.length-1));
  const vy = v => y0 + (y1-y0) * ((v-lo)/(hi-lo));
  // grid + y labels
  ctx.strokeStyle="#2a3650"; ctx.fillStyle="#8aa0bd"; ctx.font="11px sans-serif"; ctx.lineWidth=1;
  for(let t=Math.ceil(lo/5)*5; t<=hi; t+=5){
    const y=vy(t); ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke();
    ctx.fillText(t+"°", 6, y+3);
  }
  // threshold lines
  const thr=[[CFG.warn,"#ffa94d"],[CFG.stop,"#ff4d4f"],[CFG.crit,"#b00020"]];
  ctx.setLineDash([5,4]);
  for(const [t,col] of thr){ if(t<lo||t>hi) continue; ctx.strokeStyle=col; const y=vy(t);
    ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke(); }
  ctx.setLineDash([]);
  // series
  if(hist.length){
    ctx.strokeStyle="#4dabf7"; ctx.lineWidth=2; ctx.beginPath();
    hist.forEach((p,i)=>{ const x=vx(i), y=vy(p.v); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
    ctx.stroke();
    const last=hist[hist.length-1];
    ctx.fillStyle=bandColor(last.v);
    ctx.beginPath(); ctx.arc(vx(hist.length-1), vy(last.v), 4, 0, 7); ctx.fill();
  }
}

function fmt(x){ return (x==null)?"--":x.toFixed(1); }

async function tick(){
  const dot=document.getElementById("dot");
  try{
    const r = await fetch("/api/temp", {cache:"no-store"});
    const d = await r.json();
    if(d.ok){
      const v=d.temp_c;
      hist.push({t:d.ts, v}); while(hist.length>CFG.history) hist.shift();
      if(minV==null||v<minV) minV=v; if(maxV==null||v>maxV) maxV=v; sum+=v; n++;
      const tEl=document.getElementById("temp");
      tEl.textContent=v.toFixed(1); tEl.style.color=bandColor(v);
      const st=document.getElementById("status");
      st.textContent=bandText(v); st.style.color=bandColor(v);
      st.className="status"+(v>=CFG.crit?" crit":"");
      document.getElementById("min").textContent=fmt(minV);
      document.getElementById("max").textContent=fmt(maxV);
      document.getElementById("mean").textContent=fmt(n?sum/n:null);
      document.getElementById("ts").textContent=(d.ts||"").replace("T"," ").slice(11);
      document.getElementById("err").textContent="";
      dot.style.background=bandColor(v);
    }else{
      document.getElementById("status").textContent="disconnected";
      document.getElementById("err").textContent=d.error||"read failed";
      dot.style.background="#666";
    }
  }catch(e){
    document.getElementById("status").textContent="dashboard error";
    document.getElementById("err").textContent=String(e);
    dot.style.background="#666";
  }
  drawChart();
}

async function boot(){
  try{ CFG = Object.assign(CFG, await (await fetch("/api/config")).json()); }catch(e){}
  document.getElementById("target").textContent = CFG.user+"@"+CFG.host;
  document.getElementById("bands").textContent =
    "bands: <"+CFG.warn+" normal · "+CFG.warn+"–"+CFG.stop+" warn · ≥"+CFG.stop+" stop · ≥"+CFG.crit+" critical";
  tick(); setInterval(tick, CFG.interval);
}
boot();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/temp":
            self._send(200, json.dumps(read_temp()), "application/json")
        elif path == "/api/config":
            cfg = {"host": CFG["host"], "user": CFG["user"],
                   "interval": int(CFG["interval"] * 1000), "history": CFG["history"],
                   "warn": WARN_C, "stop": STOP_C, "crit": CRIT_C}
            self._send(200, json.dumps(cfg), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def log_message(self, fmt, *args):  # quiet: only log errors
        if not str(args[1] if len(args) > 1 else "").startswith("2"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=CFG["host"])
    ap.add_argument("--user", default=CFG["user"])
    ap.add_argument("--port", type=int, default=CFG["port"], help="SSH port")
    ap.add_argument("--key", default=CFG["key"], help="SSH private key (default ~/.ssh/id_zturn)")
    ap.add_argument("--interval", type=float, default=CFG["interval"], help="client poll seconds")
    ap.add_argument("--history", type=int, default=CFG["history"])
    ap.add_argument("--bind", default="0.0.0.0", help="HTTP bind address")
    ap.add_argument("--http-port", type=int, default=8099)
    ap.add_argument("--once", action="store_true", help="one SSH read, print JSON, exit")
    args = ap.parse_args()

    CFG.update(host=args.host, user=args.user, port=args.port, key=args.key,
               interval=args.interval, history=args.history)

    if args.once:
        print(json.dumps(read_temp(), indent=2))
        return 0

    srv = ThreadingHTTPServer((args.bind, args.http_port), Handler)
    shown = args.bind if args.bind != "0.0.0.0" else "localhost"
    print("Z-Turn temperature dashboard: http://%s:%d/  (%s@%s, every %.1fs)"
          % (shown, args.http_port, CFG["user"], CFG["host"], CFG["interval"]))
    print("Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

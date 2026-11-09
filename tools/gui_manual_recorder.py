"""Observe-only recorder for a MANUAL GUI pass (it never clicks anything).

Opens exactly the window of `python -m host.fpga_result_viewer --mode live`
(host.fpga_result_viewer.build_app) and records, to <out>/:

  pointer_events.jsonl  every real left-button press/release on the Live-tab
                        buttons (Tk <ButtonPress-1>/<ButtonRelease-1> carry screen
                        coordinates; Button.invoke() never generates them, so these
                        lines prove physical pointer input)
  gui_events.jsonl      every event the Live tab rendered (status/meta/result/...)
  close.json            the window-manager close request (X button / Alt+F4 /
                        taskbar close are indistinguishable here)

Usage: python -m tools.gui_manual_recorder --out artifacts/results/gui_manual_validation_<ts>
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    from host.fpga_result_viewer import build_app
    root, _, live = build_app(argparse.Namespace(mode="live", result_dir=None, auto=False, screenshot=None))
    t0 = time.perf_counter()
    pf = open(out / "pointer_events.jsonl", "a", encoding="utf-8")
    gf = open(out / "gui_events.jsonl", "a", encoding="utf-8")

    def stamp():
        return {"t_s": round(time.perf_counter() - t0, 3), "pc_time": datetime.now().isoformat(timespec="milliseconds")}

    def bind_buttons(w):
        for c in w.winfo_children():
            if c.winfo_class() == "Button":
                text = c.cget("text")
                for seq, kind in (("<ButtonPress-1>", "press"), ("<ButtonRelease-1>", "release")):
                    c.bind(seq, lambda e, t=text, k=kind: (pf.write(json.dumps(dict(
                        stamp(), button=t, event=k, x_root=e.x_root, y_root=e.y_root,
                        send_event=bool(getattr(e, "send_event", False)))) + "\n"), pf.flush()), add="+")
            bind_buttons(c)
    bind_buttons(live.ui)

    orig_apply = live._apply

    def rec_apply(ev):
        orig_apply(ev)
        keep = {k: ev.get(k) for k in ("type", "msg", "idx", "cls", "label", "frame", "lat_ms",
                                        "temp_c", "conf", "class_source", "n", "executed",
                                        "aborted", "temp_start", "temp_max", "temp_end") if k in ev}
        keep["conn_line"] = live.conn_var.get()
        gf.write(json.dumps(dict(stamp(), **keep)) + "\n")
        gf.flush()
    live._apply = rec_apply

    orig_close = root.on_close

    def close_request():
        rec = dict(stamp(), event="WM_DELETE_WINDOW (window-manager close: X / Alt+F4 / taskbar)",
                   run_active=bool(live.thread and live.thread.is_alive()))
        a_t = time.perf_counter()
        orig_close()
        rec["close_handler_s"] = round(time.perf_counter() - a_t, 3)
        (out / "close.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        pf.close()
        gf.close()
    root.protocol("WM_DELETE_WINDOW", close_request)
    root.title(root.title() + "  [manual validation recorder: observing only]")
    (out / "recorder_started.json").write_text(json.dumps(stamp(), indent=2), encoding="utf-8")
    print("GUI open; recording to", out, flush=True)
    root.mainloop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only live console for the evaluation pack.

Serves the same detector the submission runs, driven frame by frame at wall-clock
speed, so what appears on screen is what a deployment would see rather than a
replay of a finished run. There is nothing to adjust here on purpose: no
thresholds, no sliders, no knobs. Pick a video, watch the events arrive.

The scoring objects are the shipped ones - StreamScorer and HysteresisTracker,
via ahc.live.LiveWorker - and the encoder is loaded once and shared, so
switching video costs nothing.

    python scripts/live_dashboard.py --videos data/ahc/eval --port 8080
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ahc import features, prompts
from ahc.live import LiveConfig, LiveWorker
from ahc.submission import CLASSES

# The console must fire where the submission fires. LiveConfig's own defaults
# (0.4 / 0.16) are four times less sensitive than the shipped onset thresholds,
# so a clip scoring 0.126 - which the pipeline reports as an event - showed
# nothing here at all.
D3_HI, D3_LO = 0.10, 0.04

VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Live anomaly detection - evaluation pack</title>
<style>
:root{--ink:#0f1216;--panel:#171b23;--line:#262c38;--fg:#e9edf2;--dim:#98a2b3;
      --key:#6ea8fe;--hot:#f2778a;--ok:#5ed39a;
      --mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--ink);color:var(--fg);
     font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{display:flex;align-items:baseline;gap:14px;padding:14px 20px;
       border-bottom:1px solid var(--line)}
h1{font-size:15px;margin:0;font-weight:600}
.tag{font:600 10.5px var(--mono);letter-spacing:.12em;text-transform:uppercase;
     color:var(--key)}
.live{margin-left:auto;font:600 11px var(--mono);color:var(--dim)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;
     background:var(--hot);margin-right:6px;animation:p 1.4s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.25}}
main{display:grid;grid-template-columns:260px 1fr 340px;gap:0;
     height:calc(100vh - 51px)}
aside{border-right:1px solid var(--line);overflow-y:auto;padding:10px}
.grp{font:600 10px var(--mono);letter-spacing:.1em;color:var(--dim);
     text-transform:uppercase;margin:12px 0 6px 6px}
.v{display:flex;gap:8px;align-items:center;padding:6px 8px;border-radius:6px;
   cursor:pointer;font-size:13px}
.v:hover{background:#1c2129}
.v.on{background:#1d283a;color:#fff}
.v .id{font-family:var(--mono);font-size:12px}
.v .d{margin-left:auto;font-size:10.5px;color:var(--dim)}
.v .hit{width:6px;height:6px;border-radius:50%;background:var(--hot);flex-shrink:0}
.v .miss{width:6px;height:6px;border-radius:50%;background:#2b3240;flex-shrink:0}
.key{font-size:10.5px;color:var(--dim);padding:4px 8px 8px;display:flex;
     gap:6px;align-items:center}
section{display:flex;flex-direction:column;align-items:center;
        justify-content:center;padding:16px;overflow:hidden}
#shot{max-width:100%;max-height:100%;border-radius:8px;
      border:1px solid var(--line);background:#000}
.hint{color:var(--dim);font-size:13px;text-align:center;max-width:340px}
.panel{border-left:1px solid var(--line);overflow-y:auto;padding:14px}
h2{font:600 10.5px var(--mono);letter-spacing:.1em;text-transform:uppercase;
   color:var(--dim);margin:0 0 8px}
.now{background:var(--panel);border:1px solid var(--line);border-radius:7px;
     padding:10px 12px;margin-bottom:14px}
.big{font:600 22px/1.1 var(--mono)}
.sub{color:var(--dim);font-size:11.5px;margin-top:3px}
.bar{height:6px;border-radius:3px;background:#232936;margin-top:9px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:var(--ok);transition:width .2s}
.ev{border:1px solid var(--line);border-left:3px solid var(--key);
    border-radius:6px;padding:8px 10px;margin-bottom:7px;background:var(--panel)}
.ev.open{border-left-color:var(--hot)}
.ev.arm{border-left-color:#f3c363;opacity:.85}
.armbar{height:4px;border-radius:2px;background:#232936;margin-top:6px;overflow:hidden}
.armbar i{display:block;height:100%;background:#f3c363;width:0}
.ev .c{font-weight:600;font-size:13px}
.ev .t{font-family:var(--mono);font-size:11.5px;color:var(--dim);margin-top:2px}
.none{color:var(--dim);font-size:12.5px}
canvas{width:100%;height:52px;display:block;margin-top:4px}
.raw{font-family:var(--mono);font-size:11.5px;width:100%;border-collapse:collapse}
.raw td{padding:2px 0;border:0}
.raw td:last-child{text-align:right;color:#fff}
.raw td:first-child{color:var(--dim)}
.cls{font-family:var(--mono);font-size:11.5px;display:flex;justify-content:space-between;
     padding:2px 0}
.cls b{color:#fff;font-weight:600}
.cls span:first-child{color:var(--dim)}
</style></head><body>
<header>
  <span class="tag">Live</span>
  <h1>Real-time video anomaly detection</h1>
  <span class="live" id="stat">idle</span>
</header>
<main>
  <aside id="list"></aside>
  <section id="stage"><p class="hint">Pick a video on the left.<br>
    Events appear here as the detector finds them.</p></section>
  <div class="panel">
    <h2>Now</h2>
    <div class="now">
      <div class="big" id="score">--</div>
      <div class="sub" id="meta">waiting</div>
      <div class="bar"><i id="fill"></i></div>
      <canvas id="curve" width="600" height="52"></canvas>
    </div>
    <h2>Detector output</h2>
    <table class="raw" id="raw"></table>
    <h2 style="margin-top:14px">Class scores</h2>
    <div id="cls"></div>
    <h2 style="margin-top:14px">Events detected</h2>
    <div id="events"><p class="none">None yet.</p></div>
  </div>
</main>
<script>
let cur = null;
const $ = s => document.querySelector(s);

fetch('/api/videos').then(r => r.json()).then(vs => {
  const box = $('#list');
  const key = document.createElement('div');
  key.className = 'key';
  key.innerHTML = '<span class="hit"></span> has detections &nbsp;<span class="miss"></span> none';
  box.appendChild(key);
  let last = null;
  vs.forEach(v => {
    if (v.level !== last) {
      last = v.level;
      const h = document.createElement('div');
      h.className = 'grp';
      h.textContent = 'Difficulty ' + v.level;
      box.appendChild(h);
    }
    const d = document.createElement('div');
    d.className = 'v';
    const hit = v.found && v.found.length;
    d.innerHTML = `<span class="${hit ? 'hit' : 'miss'}"></span>` +
      `<span class="id">${v.id}</span><span class="d">${hit ? v.found[0].replace(/_/g,' ').slice(0,18) : v.dur}</span>`;
    if (hit) d.title = v.found.join(', ');
    d.onclick = () => pick(v.id, d);
    box.appendChild(d);
  });
});

function pick(id, el) {
  document.querySelectorAll('.v').forEach(e => e.classList.remove('on'));
  el.classList.add('on');
  cur = id;
  $('#events').innerHTML = '<p class="none">None yet.</p>';
  $('#stage').innerHTML = '<img id="shot" alt="live view">';
  fetch('/api/select?vid=' + id).then(() => {
    $('#shot').src = '/stream.mjpg?t=' + Date.now();
    $('#stat').innerHTML = '<span class="dot"></span>running ' + id;
  });
}

function draw(curve) {
  const c = $('#curve'), x = c.getContext('2d');
  x.clearRect(0, 0, c.width, c.height);
  if (!curve || curve.length < 2) return;
  const ys = curve.map(p => p[1]);
  const lo = Math.min(...ys, 0), hi = Math.max(...ys, 0.1), sp = (hi - lo) || 1;
  x.beginPath();
  curve.forEach((p, i) => {
    const px = i / (curve.length - 1) * c.width;
    const py = c.height - (p[1] - lo) / sp * (c.height - 6) - 3;
    i ? x.lineTo(px, py) : x.moveTo(px, py);
  });
  x.strokeStyle = '#6ea8fe'; x.lineWidth = 1.5; x.stroke();
}

setInterval(() => {
  if (!cur) return;
  fetch('/api/state').then(r => r.json()).then(s => {
    if (!s.ready) return;
    $('#score').textContent = (s.score >= 0 ? '+' : '') + s.score.toFixed(2);
    $('#meta').textContent = `${s.t.toFixed(1)}s  ${s.video}` +
      (s.warm ? '' : '   (baseline warming up)');
    const n = Math.max(0, Math.min(1, (s.score - s.lo) / ((s.hi * 2 - s.lo) || 1)));
    const f = $('#fill');
    f.style.width = (n * 100) + '%';
    f.style.background = s.score >= s.hi ? '#f2778a'
                       : s.score >= s.lo ? '#6ea8fe' : '#5ed39a';
    draw(s.curve);

    // Exactly what the detector reports, unrounded beyond its own precision.
    const rows = [
      ['score (onset)',   s.score],
      ['semantic',        s.semantic],
      ['deviation',       s.deviation],
      ['level1',          s.level1],
      ['level1 max',      s.level1_max],
      ['threshold hi',    s.hi],
      ['threshold lo',    s.lo],
      ['baseline warm',   s.warm],
      ['tracker',         s.active ? 'OPEN' : (s.arming ? 'arming' : 'idle')],
      ['frames at 4 Hz',  s.curve ? s.curve.length : 0],
    ];
    $('#raw').innerHTML = rows.map(r =>
      `<tr><td>${r[0]}</td><td>${typeof r[1] === 'number' ? r[1].toFixed(3) : r[1]}</td></tr>`
    ).join('');

    $('#cls').innerHTML = (s.top_classes || []).map(c =>
      `<div class="cls"><span>${c.name.replace(/_/g,' ')}</span><b>${c.p.toFixed(4)}</b></div>`
    ).join('') || '<p class="none">--</p>';

    let arm = '';
    if (!s.active && s.arming) {
      const pct = Math.min(100, s.arming.held / s.arming.need * 100);
      arm = `<div class="ev arm"><div class="c">confirming\u2026</div>
        <div class="t">evidence held ${s.arming.held.toFixed(1)}s of
          ${s.arming.need.toFixed(1)}s needed</div>
        <div class="armbar"><i style="width:${pct}%"></i></div></div>`;
    }
    const all = (s.active ? [Object.assign({open: 1}, s.active)] : [])
                .concat((s.events || []).slice().reverse());
    $('#events').innerHTML = arm + (all.length ? all.map(e => `
      <div class="ev ${e.open ? 'open' : ''}">
        <div class="c">${e.class_name.replace(/_/g, ' ')}</div>
        <div class="t">${e.start.toFixed(1)}s –
          ${e.end === null ? 'ongoing' : e.end.toFixed(1) + 's'}
          · ${e.duration.toFixed(1)}s · peak ${e.peak.toFixed(2)}</div>
      </div>`).join('') : (arm ? '' : '<p class="none">None yet.</p>'));
  });
}, 400);
</script></body></html>"""


class State:
    """One worker at a time; the encoder and prompt bank are shared."""

    def __init__(self, root: Path, encoder, bank):
        self.root, self.encoder, self.bank = root, encoder, bank
        self.found: dict[str, list[str]] = {}
        self.worker: LiveWorker | None = None
        self.lock = threading.Lock()
        self.videos = self._scan()

    def _scan(self) -> list[dict]:
        import cv2
        out = []
        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            secs = n / fps if fps > 0 else 0.0
            # Difficulty from the pack's own L1/L2/L3 layout, else from duration.
            lvl = next((int(x[1]) for x in p.parts if len(x) == 2
                        and x[0] in "Ll" and x[1] in "123"), None)
            if lvl is None:
                lvl = 1 if secs < 60 else 3 if secs > 300 else 2
            out.append({"id": p.stem, "path": str(p), "level": lvl,
                        "dur": f"{secs:.0f}s"})
        return sorted(out, key=lambda v: (v["level"], v["id"]))

    def select(self, vid: str) -> bool:
        match = next((v for v in self.videos if v["id"] == vid), None)
        if not match:
            return False
        with self.lock:
            if self.worker:
                self.worker.stop()
            self.worker = LiveWorker(match["path"], self.encoder, self.bank,
                                     LiveConfig(loop=False, hi=D3_HI, lo=D3_LO))
            self.worker.start()
        return True


class Handler(BaseHTTPRequestHandler):
    state: State = None                      # set on the class before serving

    def log_message(self, *a):               # keep the console readable
        pass

    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path == "/":
            return self._send(200, "text/html; charset=utf-8", PAGE.encode())

        if u.path == "/api/videos":
            vs = [{**{k: v[k] for k in ("id", "level", "dur")},
                   "found": self.state.found.get(v["id"], [])}
                  for v in self.state.videos]
            return self._send(200, "application/json", json.dumps(vs).encode())

        if u.path == "/api/select":
            ok = self.state.select((q.get("vid") or [""])[0])
            return self._send(200 if ok else 404, "application/json",
                              json.dumps({"ok": ok}).encode())

        if u.path == "/api/state":
            w = self.state.worker
            body = json.dumps(w.state if w else {"ready": False})
            return self._send(200, "application/json", body.encode())

        if u.path == "/stream.mjpg":
            return self._mjpeg()

        self._send(404, "text/plain", b"not found")

    def _mjpeg(self):
        import time
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                w = self.state.worker
                jpg = w.jpeg if w else None
                if jpg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpg)).encode()
                                     + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.04)
        except (BrokenPipeError, ConnectionResetError):
            pass                              # the tab was closed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/ahc/eval")
    ap.add_argument("--encoder", default="google/siglip2-base-patch16-224")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--submission", default="",
                    help="a submission JSON; videos it found events in are "
                         "marked in the list so a demo need not guess")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    root = Path(a.videos).expanduser()
    print(f"loading {a.encoder} ...", flush=True)
    enc = features.FrameEncoder(a.encoder, device=a.device, batch_size=8)
    cache = features.Cache(root / ".ahc_cache", enc.model_id)
    bank = features.encode_prompt_bank(enc, prompts.build(list(CLASSES)), cache)

    found = {}
    if a.submission and Path(a.submission).exists():
        sub = json.loads(Path(a.submission).read_text())
        for pr in sub.get("predictions", []):
            names = sorted({e["class_name"] for e in pr.get("events", [])})
            if names:
                found[pr["video_id"]] = names
        print(f"{len(found)} videos carry detections in {a.submission}")

    st = State(root, enc, bank)
    st.found = found
    print(f"{len(st.videos)} videos from {root}")
    Handler.state = st
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"console on http://0.0.0.0:{a.port}  (ctrl-c to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        if st.worker:
            st.worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

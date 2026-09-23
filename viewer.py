"""Live web viewer for miniagent session logs.

    python viewer.py                      # serve ./runs at http://localhost:8765
    python viewer.py --dir runs --port 8765

Open the URL, pick a session in the sidebar. The page auto-refreshes as the
session's .jsonl is appended while miniagent runs. Each record shows a readable
rendered view; each tool result can be flipped between rendered text and the
raw JSON string exactly as it appears in the .jsonl, and copied.

Stdlib only, no dependencies.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

LOG_DIR = Path("runs")  # overridden in main()


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>miniagent log viewer</title>
<style>
:root {
  --bg: #ffffff; --panel: #f6f7f9; --border: #e3e6ea; --fg: #1a1d21;
  --muted: #6b7280; --accent: #2563eb; --user: #eef4ff; --model: #f3f4f6;
  --tool: #fff7ed; --toolbar: #eef2f7; --result: #f0fdf4; --danger: #b91c1c;
  --code: #0f172a; --codebg: #f1f5f9;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1115; --panel: #161a20; --border: #2a2f37; --fg: #e5e7eb;
    --muted: #9aa4b2; --accent: #60a5fa; --user: #16233b; --model: #1b1f26;
    --tool: #2a2013; --toolbar: #1b2029; --result: #12241a; --danger: #f87171;
    --code: #e5e7eb; --codebg: #0b1220;
  }
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
  color: var(--fg); background: var(--bg); }
.layout { display: flex; height: 100vh; }
aside { width: 260px; flex: 0 0 260px; border-right: 1px solid var(--border); background: var(--panel);
  overflow-y: auto; padding: 12px; }
aside h1 { font-size: 13px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin: 4px 6px 12px; }
.session { display: block; width: 100%; text-align: left; border: 1px solid transparent; background: none; color: var(--fg);
  padding: 8px 10px; border-radius: 8px; cursor: pointer; font: inherit; }
.session:hover { background: var(--bg); }
.session.active { background: var(--bg); border-color: var(--border); }
.session .name { font-weight: 600; }
.session .meta { font-size: 12px; color: var(--muted); }
main { flex: 1; overflow-y: auto; }
.toolbar { position: sticky; top: 0; z-index: 5; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
  padding: 10px 16px; background: var(--toolbar); border-bottom: 1px solid var(--border); }
.toolbar .title { font-weight: 700; margin-right: auto; }
button.tb { border: 1px solid var(--border); background: var(--bg); color: var(--fg); padding: 5px 10px;
  border-radius: 7px; cursor: pointer; font: inherit; }
button.tb.on { background: var(--accent); color: #fff; border-color: var(--accent); }
.live { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; }
.dot.off { background: #9ca3af; }
.content { padding: 16px; max-width: 900px; }
.turn { margin: 22px 0 10px; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); border-top: 1px solid var(--border); padding-top: 14px; }
.turn:first-child { border-top: none; margin-top: 4px; }
.rec { border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin: 8px 0; background: var(--model); position: relative; }
.rec.user { background: var(--user); }
.rec.tool { background: var(--tool); }
.rec.result { background: var(--result); }
.role { font-size: 12px; font-weight: 700; color: var(--muted); margin-bottom: 4px; letter-spacing: .02em; }
.role.danger { color: var(--danger); }
.body { white-space: pre-wrap; word-break: break-word; }
.call { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
.call .fn { color: var(--accent); font-weight: 600; }
.resultbox { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; }
.resultbox pre { margin: 4px 0 0; white-space: pre-wrap; word-break: break-word; max-height: 24em; overflow-y: auto; }
.resultbox pre.raw { background: var(--codebg); color: var(--code); padding: 10px; border-radius: 6px; }
.rawbar { margin-top: 6px; display: flex; gap: 8px; align-items: center; font-size: 12px; color: var(--muted); }
.rawbar button { border: 1px solid var(--border); background: var(--bg); color: var(--accent); cursor: pointer;
  font: inherit; padding: 2px 8px; border-radius: 6px; }
.empty { color: var(--muted); padding: 40px 16px; text-align: center; }
</style>
</head>
<body>
<div class="layout">
  <aside>
    <h1>Sessions</h1>
    <div id="sessions"></div>
  </aside>
  <main>
    <div class="toolbar">
      <span class="title" id="title">Pick a session →</span>
      <span class="live"><span class="dot" id="liveDot"></span><span id="liveText">live</span></span>
    </div>
    <div id="view" class="content"><div class="empty">No session selected.</div></div>
  </main>
</div>
<script>
let current = null;       // session file name
let lastMtime = null;
let rawShown = new Set(); // record indices whose result is shown as raw JSON string

const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const esc = s => (s == null ? "" : String(s));

function compactArgs(args, maxVal = 80) {
  if (!args) return "";
  return Object.entries(args).map(([k, v]) => {
    let s = typeof v === "string" ? v : JSON.stringify(v);
    s = s.replace(/\s+/g, " ");
    if (s.length > maxVal) s = s.slice(0, maxVal) + "…";
    return k + "=" + s;
  }).join(", ");
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return; } catch (e) {}
  const ta = el("textarea"); ta.value = text; document.body.appendChild(ta);
  ta.select(); document.execCommand("copy"); ta.remove();
}

async function loadSessions() {
  const r = await fetch("/api/sessions");
  const list = await r.json();
  const box = document.getElementById("sessions");
  box.innerHTML = "";
  for (const s of list) {
    const b = el("button", "session" + (s.name === current ? " active" : ""));
    b.appendChild(el("div", "name", s.name.replace(/\.jsonl$/, "")));
    b.appendChild(el("div", "meta", s.records + " records · " + s.mtime));
    b.onclick = () => selectSession(s.name);
    box.appendChild(b);
  }
}

function selectSession(name) {
  if (current !== name) { current = name; lastMtime = null; rawShown = new Set(); }
  loadSessions();
  refresh(true);
}

async function refresh(force) {
  if (!current) return;
  const r = await fetch("/api/session?name=" + encodeURIComponent(current));
  if (!r.ok) return;
  const data = await r.json();
  document.getElementById("title").textContent = current.replace(/\.jsonl$/, "");
  if (!force && data.mtime === lastMtime) return;   // nothing changed
  lastMtime = data.mtime;
  render(data);
  loadSessions();
}

function render(data) {
  const view = document.getElementById("view");
  view.innerHTML = "";
  let lastTurn = null;
  data.records.forEach((rec, i) => {
    if (rec.type === "tool_call") return;                 // redundant with model → tool
    if (rec.turn !== lastTurn) {
      const ts = (rec.ts || "").split("T")[1]?.slice(0, 8) || "";
      view.appendChild(el("div", "turn", "Turn " + rec.turn + " · " + ts));
      lastTurn = rec.turn;
    }
    const card = renderRecord(rec, i);
    if (card) view.appendChild(card);
  });
}

function renderRecord(rec, i) {
  const t = rec.type;
  if (t === "user") {
    const c = el("div", "rec user");
    c.appendChild(el("div", "role", "user"));
    c.appendChild(el("div", "body", esc(rec.text)));
    return c;
  }
  if (t === "model") {
    const c = el("div", "rec");
    if (rec.function_calls && rec.function_calls.length) {
      if (rec.text && rec.text.trim()) {
        c.appendChild(el("div", "role", "model"));
        c.appendChild(el("div", "body", esc(rec.text.trim())));
      }
      for (const call of rec.function_calls) {
        c.appendChild(el("div", "role", "model → tool"));
        const line = el("div", "call");
        const fn = el("span", "fn", call.name);
        line.appendChild(fn);
        line.appendChild(document.createTextNode("(" + compactArgs(call.args || {}) + ")"));
        c.appendChild(line);
      }
    } else if (rec.text == null) {
      const reason = rec.blocked || rec.error || "no response";
      c.appendChild(el("div", "role danger", "⚠️ model stopped"));
      c.appendChild(el("div", "body", esc(reason)));
    } else {
      c.appendChild(el("div", "role", "model"));
      c.appendChild(el("div", "body", esc(rec.text)));
    }
    return c;
  }
  if (t === "tool_result") {
    const c = el("div", "rec result");
    const name = rec.name || "";
    const result = esc(rec.result);
    const isErr = /^\s*(ERROR|\{"error")/.test(result);
    if (rec.declined) c.appendChild(el("div", "role danger", "🚫 result · " + name + " — user declined"));
    else c.appendChild(el("div", "role" + (isErr ? " danger" : ""), (isErr ? "❌ " : "") + "result · " + name));
    const box = el("div", "resultbox");
    const isRaw = rawShown.has(i);
    const shown = isRaw ? JSON.stringify(result) : result;   // raw = the string literal as logged
    box.appendChild(el("pre", isRaw ? "raw" : null, shown));
    const bar = el("div", "rawbar");
    const toggle = el("button", null, isRaw ? "Rendered" : "Raw");
    toggle.onclick = () => { if (isRaw) rawShown.delete(i); else rawShown.add(i); refresh(true); };
    const copy = el("button", null, "Copy");
    copy.title = "copy what is shown";
    copy.onclick = async () => {
      await copyText(shown);
      copy.textContent = "Copied ✓";
      setTimeout(() => { copy.textContent = "Copy"; }, 1200);
    };
    bar.appendChild(toggle);
    bar.appendChild(copy);
    bar.appendChild(el("span", null, result.length + " chars"));
    box.appendChild(bar);
    c.appendChild(box);
    return c;
  }
  return null;
}

loadSessions();
setInterval(() => { if (current) refresh(false); else loadSessions(); }, 1500);
</script>
</body>
</html>
"""


def _sessions_payload(log_dir: Path):
    items = []
    for p in sorted(log_dir.glob("*.jsonl"), reverse=True):
        try:
            with p.open(encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
            mtime = __import__("datetime").datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M:%S")
        except OSError:
            continue
        items.append({"name": p.name, "records": n, "mtime": mtime})
    return items


def _session_payload(log_dir: Path, name: str):
    if not name or "/" in name or "\\" in name or not name.endswith(".jsonl"):
        return None
    p = (log_dir / name).resolve()
    if p.parent != log_dir.resolve() or not p.is_file():
        return None
    raw = p.read_text(encoding="utf-8")
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return {"records": records, "mtime": p.stat().st_mtime}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str), "application/json; charset=utf-8")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif parsed.path == "/api/sessions":
            self._json(_sessions_payload(LOG_DIR))
        elif parsed.path == "/api/session":
            name = parse_qs(parsed.query).get("name", [""])[0]
            payload = _session_payload(LOG_DIR, name)
            if payload is None:
                self._json({"error": "not found"}, code=404)
            else:
                self._json(payload)
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")


def main() -> None:
    global LOG_DIR
    ap = argparse.ArgumentParser(description="Live web viewer for miniagent session logs.")
    ap.add_argument("--dir", default="runs", help="log directory to watch (default: runs)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    LOG_DIR = Path(args.dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {LOG_DIR}/ at http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        server.shutdown()


if __name__ == "__main__":
    main()

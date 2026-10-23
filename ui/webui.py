"""Shared bits of the stdlib web viewers (viewer.py, caseviewer.py, injviewer.py).

  - BASE_CSS / BASE_JS : theme, sidebar + toolbar layout, message cards, helpers
  - page()             : assemble a full HTML page from those plus page-specific parts
  - JSONHandler        : BaseHTTPRequestHandler with _send / _json / _body helpers
  - safe_file()        : resolve a bare file name inside a directory (no traversal)
  - serve()            : run a ThreadingHTTPServer until Ctrl-C
"""

import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


BASE_CSS = r"""
:root {
  --bg: #ffffff; --panel: #f6f7f9; --border: #e3e6ea; --fg: #1a1d21;
  --muted: #6b7280; --accent: #2563eb; --user: #eef4ff; --model: #f3f4f6;
  --tool: #fff7ed; --toolbar: #eef2f7; --result: #f0fdf4; --danger: #b91c1c;
  --dangerbg: #fef2f2; --code: #0f172a; --codebg: #f1f5f9;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1115; --panel: #161a20; --border: #2a2f37; --fg: #e5e7eb;
    --muted: #9aa4b2; --accent: #60a5fa; --user: #16233b; --model: #1b1f26;
    --tool: #2a2013; --toolbar: #1b2029; --result: #12241a; --danger: #f87171;
    --dangerbg: #2a1515; --code: #e5e7eb; --codebg: #0b1220;
  }
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
  color: var(--fg); background: var(--bg); }
.layout { display: flex; height: 100vh; }
aside { width: 260px; flex: 0 0 260px; border-right: 1px solid var(--border); background: var(--panel);
  overflow-y: auto; padding: 12px; }
aside h1 { font-size: 13px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin: 4px 6px 12px; }
.item { display: block; width: 100%; text-align: left; border: 1px solid transparent; background: none; color: var(--fg);
  padding: 8px 10px; border-radius: 8px; cursor: pointer; font: inherit; }
.item:hover { background: var(--bg); }
.item.active { background: var(--bg); border-color: var(--border); }
.item .name { font-weight: 600; word-break: break-all; }
.item .meta { font-size: 12px; color: var(--muted); }
main { flex: 1; overflow-y: auto; }
.toolbar { position: sticky; top: 0; z-index: 5; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
  padding: 10px 16px; background: var(--toolbar); border-bottom: 1px solid var(--border); }
.toolbar .title { font-weight: 700; margin-right: auto; }
button.tb { border: 1px solid var(--border); background: var(--bg); color: var(--fg); padding: 5px 10px;
  border-radius: 7px; cursor: pointer; font: inherit; }
button.tb.on { background: var(--accent); color: #fff; border-color: var(--accent); }
button.tb:disabled { opacity: .6; cursor: default; }
.live { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; }
.dot.off { background: #9ca3af; }
.content { padding: 16px; max-width: 900px; }
.rec { border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin: 8px 0; background: var(--model); position: relative; }
.rec.user { background: var(--user); }
.rec.tool { background: var(--tool); }
.rec.result { background: var(--result); }
.role { font-size: 12px; font-weight: 700; color: var(--muted); margin-bottom: 4px; letter-spacing: .02em; }
.role.danger { color: var(--danger); }
.body { white-space: pre-wrap; word-break: break-word; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.call { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; word-break: break-word; }
.call .fn { color: var(--accent); font-weight: 600; }
pre.raw { background: var(--codebg); color: var(--code); padding: 10px; border-radius: 6px; margin: 4px 0 0;
  font: 12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap; word-break: break-word; max-height: 32em; overflow-y: auto; }
.rawbar { margin-top: 6px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 12px; color: var(--muted); }
.rawbar button { border: 1px solid var(--border); background: var(--bg); color: var(--accent); cursor: pointer;
  font: inherit; padding: 2px 8px; border-radius: 6px; }
.rawbar button.on { background: var(--accent); color: #fff; border-color: var(--accent); }
.rawbar .ok { color: #16a34a; }
.rawbar .bad { color: var(--danger); }
.empty { color: var(--muted); padding: 40px 16px; text-align: center; }
@media (max-width: 700px) {
  .layout { flex-direction: column; height: auto; }
  aside { width: auto; flex: none; max-height: 40vh; border-right: none; border-bottom: 1px solid var(--border); }
}
"""

BASE_JS = r"""
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

// a small button that copies getText() and flashes "Copied ✓"
function copyButton(getText, label = "Copy") {
  const b = el("button", null, label);
  b.onclick = async () => { await copyText(getText()); b.textContent = "Copied ✓"; setTimeout(() => { b.textContent = label; }, 1200); };
  return b;
}

// the sidebar list: items = [{key, name, meta}]
function renderSidebar(box, items, currentKey, onPick) {
  box.innerHTML = "";
  for (const it of items) {
    const b = el("button", "item" + (it.key === currentKey ? " active" : ""));
    b.appendChild(el("div", "name", it.name));
    if (it.meta) b.appendChild(el("div", "meta", it.meta));
    b.onclick = () => onPick(it.key);
    box.appendChild(b);
  }
}
"""


def page(title: str, css: str, body: str, js: str) -> str:
    """A full HTML page: shared CSS/JS plus the page's own."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{BASE_CSS}{css}</style>
</head>
<body>
{body}
<script>{BASE_JS}{js}</script>
</body>
</html>
"""


def mtime_str(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M:%S")


def safe_file(directory: Path, name: str, suffix: str):
    """directory/name if name is a bare file name ending in suffix that exists there, else None."""
    if not name or "/" in name or "\\" in name or not name.endswith(suffix):
        return None
    p = (directory / name).resolve()
    if p.parent != directory.resolve() or not p.is_file():
        return None
    return p


class JSONHandler(BaseHTTPRequestHandler):
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

    def _html(self, html):
        self._send(200, html, "text/html; charset=utf-8")

    def _not_found(self):
        self._send(404, "not found", "text/plain; charset=utf-8")

    def _body(self):
        """The request body parsed as JSON (raises ValueError on bad JSON)."""
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))


def serve(handler, host: str, port: int, what: str) -> None:
    server = ThreadingHTTPServer((host, port), handler)
    print(f"serving {what} at http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        server.shutdown()

"""Live web viewer for miniagent session logs.

    python -m ui.viewer                      # serve ./runs at http://localhost:8765
    python -m ui.viewer --dir runs --port 8765

Open the URL, pick a session in the sidebar. The page auto-refreshes as the
session's .jsonl is appended while miniagent runs. Each record shows a readable
rendered view; each tool result can be flipped between rendered text and the
raw JSON string exactly as it appears in the .jsonl, copied, and edited in
either form. Saving an edit rewrites that record's "result" in the .jsonl (in
place, so a running miniagent keeps appending) and regenerates the .md.

Two modes (toolbar):
  Log      : the session as a transcript, one card per record
  Requests : the session's final request body as sent to the API (it holds the whole
             conversation), broken into its contents and parts
Every card flips between a readable view and its JSON.

Stdlib only, no dependencies.
"""

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from harness import runlog
from . import webui

LOG_DIR = Path("runs")  # overridden in main()


CSS = r"""
.turn { margin: 22px 0 10px; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); border-top: 1px solid var(--border); padding-top: 14px; }
.turn:first-child { border-top: none; margin-top: 4px; }
.views { position: absolute; top: 7px; right: 8px; display: flex; gap: 2px; }
.views button { border: 1px solid var(--border); background: var(--bg); color: var(--muted); cursor: pointer;
  font-family: inherit; font-size: 11px; padding: 1px 7px; border-radius: 5px; }
.views button.on { background: var(--accent); color: #fff; border-color: var(--accent); }
.rec > .role:first-child { padding-right: 130px; }
.rec.req { background: var(--panel); border-style: dashed; }
.kv { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 3px 14px; font-size: 13px; }
.kv .k { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; padding-top: 1px; }
.kv .v { white-space: pre-wrap; word-break: break-word; min-width: 0; max-height: 16em; overflow-y: auto; }
.pl { font-size: 12px; font-weight: 700; color: var(--muted); margin: 8px 0 0; }
.resultbox { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; }
.resultbox pre { margin: 4px 0 0; white-space: pre-wrap; word-break: break-word; max-height: 24em; overflow-y: auto; }
.resultbox textarea { display: block; width: 100%; margin: 4px 0 0; padding: 10px; border-radius: 6px;
  border: 1px solid var(--accent); background: var(--codebg); color: var(--code); resize: vertical;
  font: 12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; tab-size: 2; outline: none; }
"""

BODY = r"""
<div class="layout">
  <aside>
    <h1>Sessions</h1>
    <div id="sessions"></div>
  </aside>
  <main>
    <div class="toolbar">
      <span class="title" id="title">Pick a session →</span>
      <button class="tb on" id="mLog">Log</button>
      <button class="tb" id="mReq">Requests</button>
      <span class="live"><span class="dot" id="liveDot"></span><span id="liveText">live</span></span>
    </div>
    <div id="view" class="content"><div class="empty">No session selected.</div></div>
  </main>
</div>
"""

JS = r"""
let current = null;       // session file name
let lastMtime = null;
let lastData = null;      // last loaded session payload
let mode = "log";         // log | requests
let jsonShown = new Set();  // cards flipped to JSON: "r<i>" record i, "q<i>" request i, "p<i>.<c>.<p>" a part of it
let asSent = true;        // request JSON: the exact text sent, or pretty-printed
let parsed = new Map();   // record index -> parsed request body (null if not JSON)
let rawShown = new Set(); // record indices whose result is shown as raw JSON string
let drafts = new Map();   // record index -> {raw: bool, text} for results being edited

async function loadSessions() {
  const r = await fetch("/api/sessions");
  const list = await r.json();
  renderSidebar(document.getElementById("sessions"),
    list.map(s => ({ key: s.name, name: s.name.replace(/\.jsonl$/, ""), meta: s.records + " records · " + s.mtime })),
    current, selectSession);
}

function selectSession(name) {
  if (current !== name && drafts.size && !confirm("Discard unsaved edits?")) return;
  if (current !== name) {
    current = name; lastMtime = null; lastData = null; rawShown = new Set(); drafts = new Map();
    jsonShown = new Set(); parsed = new Map();
  }
  loadSessions();
  refresh(true);
}

async function refresh(force) {
  if (!current) return;
  if (!force && drafts.size) return;   // don't re-render (and steal focus) under an open editor
  const r = await fetch("/api/session?name=" + encodeURIComponent(current));
  if (!r.ok) return;
  const data = await r.json();
  document.getElementById("title").textContent = current.replace(/\.jsonl$/, "");
  if (!force && data.mtime === lastMtime) return;   // nothing changed
  lastMtime = data.mtime;
  render(data);
  loadSessions();
}

function rerender() { if (lastData) render(lastData); }

function render(data) {
  lastData = data;
  document.getElementById("mLog").className = "tb" + (mode === "log" ? " on" : "");
  document.getElementById("mReq").className = "tb" + (mode === "requests" ? " on" : "");
  const view = document.getElementById("view");
  view.innerHTML = "";
  if (mode === "requests") renderRequests(view, data);
  else renderLog(view, data);
}

const CARD_CLASS = { user: "user", tool_result: "result", request: "req" };

function renderLog(view, data) {
  let lastTurn = null;
  data.records.forEach((rec, i) => {
    if (rec.type === "tool_call") return;                 // redundant with model → tool
    if (rec.turn !== lastTurn) {
      const ts = (rec.ts || "").split("T")[1]?.slice(0, 8) || "";
      view.appendChild(el("div", "turn", "Turn " + rec.turn + " · " + ts));
      lastTurn = rec.turn;
    }
    const key = "r" + i;
    const card = jsonShown.has(key)
      ? jsonCard(CARD_CLASS[rec.type] || "", rec.type + " · log record", JSON.stringify(rec, null, 2))
      : renderRecord(rec, i);
    if (!card) return;
    card.appendChild(viewToggle(key));
    view.appendChild(card);
  });
}

// the Readable / JSON switch in a card's corner
function viewToggle(key) {
  const box = el("div", "views");
  for (const [label, json] of [["Readable", false], ["JSON", true]]) {
    const b = el("button", jsonShown.has(key) === json ? "on" : null, label);
    b.onclick = () => { if (json) jsonShown.add(key); else jsonShown.delete(key); rerender(); };
    box.appendChild(b);
  }
  return box;
}

function jsonCard(cls, label, text, extraButtons) {
  const c = el("div", "rec " + cls);
  c.appendChild(el("div", "role", label));
  c.appendChild(el("pre", "raw", text));
  const bar = el("div", "rawbar");
  if (extraButtons) extraButtons(bar);
  bar.appendChild(copyButton(() => text));
  bar.appendChild(el("span", null, text.length + " chars"));
  c.appendChild(bar);
  return c;
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
    const bar = el("div", "rawbar");
    const btn = (label, fn) => { const b = el("button", null, label); b.onclick = fn; bar.appendChild(b); return b; };
    const draft = drafts.get(i);
    if (draft) {
      const ta = el("textarea");
      ta.value = draft.text;
      ta.spellcheck = false;
      const status = el("span");
      const check = () => {
        draft.text = ta.value;
        ta.rows = Math.min(Math.max(ta.value.split("\n").length + 1, 4), 30);
        let ok, msg;
        if (draft.raw) {
          try { ok = typeof JSON.parse(ta.value) === "string"; } catch (e) { ok = false; }
          msg = ok ? "valid string literal" : "not a valid JSON string literal";
        } else {
          try { JSON.parse(ta.value); ok = true; } catch (e) { ok = false; }
          msg = ok ? "result is valid JSON" : "result is not valid JSON (will be saved as-is)";
        }
        status.className = ok ? "ok" : "bad";
        status.textContent = msg;
        return ok;
      };
      const save = async () => {
        if (draft.raw && !check()) return;
        const value = draft.raw ? JSON.parse(ta.value) : ta.value;
        const r = await fetch("/api/edit", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: current, i, old: rec.result, result: value }) });
        const res = await r.json();
        if (!r.ok) { status.className = "bad"; status.textContent = "save failed: " + res.error; return; }
        drafts.delete(i);
        refresh(true);
      };
      const cancel = () => { drafts.delete(i); refresh(true); };
      ta.oninput = check;
      ta.onkeydown = e => {
        if (e.key === "Tab") { e.preventDefault(); ta.setRangeText("  ", ta.selectionStart, ta.selectionEnd, "end"); check(); }
        else if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); save(); }
        else if (e.key === "Escape") { e.preventDefault(); cancel(); }
      };
      box.appendChild(ta);
      const toggle = btn(draft.raw ? "Rendered" : "Raw", () => {
        // convert the draft between its two forms
        if (draft.raw) {
          if (!check()) return;
          draft.text = JSON.parse(ta.value);
        } else {
          draft.text = JSON.stringify(ta.value);
        }
        draft.raw = !draft.raw;
        if (draft.raw) rawShown.add(i); else rawShown.delete(i);
        refresh(true);
      });
      btn("Save", save).title = "Ctrl+S";
      btn("Cancel", cancel).title = "Esc";
      bar.appendChild(status);
      check();
      setTimeout(() => ta.focus(), 0);
    } else {
      const isRaw = rawShown.has(i);
      const shown = isRaw ? JSON.stringify(result) : result;   // raw = the string literal as logged
      box.appendChild(el("pre", isRaw ? "raw" : null, shown));
      btn(isRaw ? "Rendered" : "Raw", () => { if (isRaw) rawShown.delete(i); else rawShown.add(i); refresh(true); });
      const copy = bar.appendChild(copyButton(() => shown));
      copy.title = "copy what is shown";
      btn("Edit", () => { drafts.set(i, { raw: isRaw, text: shown }); refresh(true); });
      bar.appendChild(el("span", null, result.length + " chars"));
    }
    box.appendChild(bar);
    c.appendChild(box);
    return c;
  }
  if (t === "request") {
    const q = parseRequest(rec, i);
    const c = el("div", "rec req");
    c.appendChild(el("div", "role", "→ final request · sent to the model"));
    if (!q) { c.appendChild(el("div", "body", "body is not JSON · " + kb(rec))); return c; }
    const s = partStats(q);
    c.appendChild(el("div", "body", s.contents + " contents · " + s.parts + " parts (" + s.kinds + ") · " + kb(rec)));
    const bar = el("div", "rawbar");
    const open = el("button", null, "Open in Requests");
    open.onclick = () => { mode = "requests"; rerender(); };
    bar.appendChild(open);
    c.appendChild(bar);
    return c;
  }
  return null;
}

// --- requests: the bodies sent to the API, in wire format ------------------------------

const PART_META = new Set(["thought", "thoughtSignature", "videoMetadata", "partMetadata"]);
const partKind = p => p.thought ? "thought" : (Object.keys(p).find(k => !PART_META.has(k)) || "?");
const kb = rec => ((rec.body || "").length / 1024).toFixed(1) + " KB";
// record i's request body, parsed (null if it isn't JSON)
function parseRequest(rec, i) {
  if (!parsed.has(i)) { let q = null; try { q = JSON.parse(rec.body); } catch (e) {} parsed.set(i, q); }
  return parsed.get(i);
}

function partStats(q) {
  const kinds = {};
  let parts = 0;
  for (const c of q.contents || []) for (const p of c.parts || []) { const k = partKind(p); kinds[k] = (kinds[k] || 0) + 1; parts++; }
  return { contents: (q.contents || []).length, parts, kinds: Object.entries(kinds).map(([k, n]) => k + " " + n).join(", ") };
}

function renderRequests(view, data) {
  const i = data.records.map(r => r.type).lastIndexOf("request");
  if (i < 0) {
    view.appendChild(el("div", "empty", "No request in this session yet: miniagent writes the final request when the session ends."));
    return;
  }
  const rec = data.records[i], q = parseRequest(rec, i);
  view.appendChild(requestCard(rec, i, q));
  if (!q) return;
  (q.contents || []).forEach((ct, ci) => {
    view.appendChild(el("div", "turn", "contents[" + ci + "] · " + ct.role + " · " + (ct.parts || []).length + " parts"));
    (ct.parts || []).forEach((p, pi) => view.appendChild(partCard(p, i, ci, pi, ct.role)));
  });
}

function requestCard(rec, i, q) {
  const key = "q" + i;
  const label = "final request · turn " + rec.turn;
  if (!q || jsonShown.has(key)) {
    const text = asSent || !q ? rec.body : JSON.stringify(q, null, 2);
    const c = jsonCard("req", label, text, bar => {
      for (const [lbl, v] of [["As sent", true], ["Pretty", false]]) {
        const b = el("button", asSent === v ? "on" : null, lbl);
        b.onclick = () => { asSent = v; rerender(); };
        bar.appendChild(b);
      }
    });
    if (q) c.appendChild(viewToggle(key));
    return c;
  }
  const c = el("div", "rec req");
  c.appendChild(el("div", "role", label));
  const kv = el("div", "kv");
  const row = (k, v) => { kv.appendChild(el("div", "k", k)); kv.appendChild(el("div", "v", v)); };
  const s = partStats(q);
  row("size", kb(rec) + " (" + (rec.body || "").length + " chars)");
  row("contents", s.contents + " contents · " + s.parts + " parts (" + s.kinds + ")");
  const sys = q.systemInstruction;
  if (sys) row("systemInstruction", (sys.role ? "role " + sys.role + " · " : "") + (sys.parts || []).map(p => p.text != null ? p.text : JSON.stringify(p)).join("\n"));
  if (q.tools) {
    const fns = q.tools.flatMap(t => (t.functionDeclarations || []).map(f => f.name));
    row("tools", fns.length + " functions: " + fns.join(", "));
  }
  for (const [k, v] of Object.entries(q)) if (!["contents", "systemInstruction", "tools"].includes(k)) row(k, JSON.stringify(v));
  c.appendChild(kv);
  c.appendChild(viewToggle(key));
  return c;
}

function partCard(p, i, ci, pi, role) {
  const key = "p" + i + "." + ci + "." + pi, kind = partKind(p);
  const cls = kind === "functionResponse" ? "result" : kind === "functionCall" ? "tool" : role === "user" ? "user" : "";
  const label = "parts[" + pi + "] · " + kind + (p.thoughtSignature ? " · thoughtSignature " + p.thoughtSignature.length + " chars" : "");
  let c;
  if (jsonShown.has(key)) c = jsonCard(cls, label, JSON.stringify(p, null, 2));
  else {
    c = el("div", "rec " + cls);
    c.appendChild(el("div", "role", label));
    if (kind === "text" || kind === "thought") c.appendChild(el("div", "body", p.text));
    else if (kind === "functionCall") {
      const line = el("div", "call");
      line.appendChild(el("span", "fn", p.functionCall.name));
      line.appendChild(document.createTextNode("(" + compactArgs(p.functionCall.args || {}) + ")"));
      c.appendChild(line);
    } else if (kind === "functionResponse") {
      c.appendChild(el("div", "call", "name: " + p.functionResponse.name));
      for (const [k, v] of Object.entries(p.functionResponse.response || {})) {
        const text = typeof v === "string" ? v : JSON.stringify(v, null, 2);
        c.appendChild(el("div", "pl", "response." + k + " · " + text.length + " chars"));
        c.appendChild(el("pre", "raw", text));
      }
    } else if (kind === "inlineData") {
      c.appendChild(el("div", "body", (p.inlineData.mimeType || "") + " · " + (p.inlineData.data || "").length + " chars of base64"));
    } else c.appendChild(el("pre", "raw", JSON.stringify(p, null, 2)));
  }
  c.appendChild(viewToggle(key));
  return c;
}

document.getElementById("mLog").onclick = () => { mode = "log"; rerender(); };
document.getElementById("mReq").onclick = () => { mode = "requests"; rerender(); };

loadSessions();
setInterval(() => { if (current) refresh(false); else loadSessions(); }, 1500);
"""

PAGE = webui.page("miniagent log viewer", CSS, BODY, JS)


def _sessions_payload(log_dir: Path):
    items = []
    for p in sorted(log_dir.glob("*.jsonl"), reverse=True):
        try:
            with p.open(encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
            mtime = webui.mtime_str(p)
        except OSError:
            continue
        items.append({"name": p.name, "records": n, "mtime": mtime})
    return items


def _session_path(log_dir: Path, name: str):
    return webui.safe_file(log_dir, name, ".jsonl")


def _parse_lines(lines):
    """Yield (line_index, record) for every parseable record; record i in the UI is the i-th yielded."""
    for li, line in enumerate(lines):
        line = line.strip()
        if line:
            try:
                yield li, json.loads(line)
            except json.JSONDecodeError:
                pass


def _session_payload(log_dir: Path, name: str):
    p = _session_path(log_dir, name)
    if p is None:
        return None
    records = [rec for _, rec in _parse_lines(p.read_text(encoding="utf-8").splitlines())]
    return {"records": records, "mtime": p.stat().st_mtime}


def _edit_result(log_dir: Path, name: str, i: int, old, new: str):
    """Replace record i's "result" if it is still `old`. Returns an error string or None.

    Rewrites in place (same inode) rather than via rename: a running miniagent holds
    the .jsonl/.md open in append mode and would otherwise keep writing to the old file.
    """
    p = _session_path(log_dir, name)
    if p is None:
        return "session not found"
    with p.open("r+", encoding="utf-8") as f:
        lines = f.read().split("\n")
        found = next((x for k, x in enumerate(_parse_lines(lines)) if k == i), None)
        if found is None:
            return "record not found"
        li, rec = found
        if rec.get("type") != "tool_result":
            return "only tool results can be edited"
        if rec.get("result") != old:
            return "record changed since it was loaded; reload and retry"
        rec["result"] = new
        lines[li] = json.dumps(rec, ensure_ascii=False, default=str)
        f.seek(0)
        f.write("\n".join(lines))
        f.truncate()
    md = p.with_suffix(".md")
    if md.exists():
        records = [r for _, r in _parse_lines(lines)]
        md.write_text(f"# Session {p.stem}\n\n" + runlog.render_transcript(records), encoding="utf-8")
    return None


class Handler(webui.JSONHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(PAGE)
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
            self._not_found()

    def do_POST(self):
        if urlparse(self.path).path != "/api/edit":
            return self._not_found()
        try:
            body = self._body()
            name, i, old, new = body["name"], int(body["i"]), body["old"], body["result"]
            if not isinstance(new, str):
                raise ValueError("result must be a string")
        except (ValueError, KeyError, TypeError) as e:
            return self._json({"error": f"bad request: {e}"}, code=400)
        err = _edit_result(LOG_DIR, name, i, old, new)
        if err:
            self._json({"error": err}, code=409)
        else:
            self._json({"ok": True})


def main() -> None:
    global LOG_DIR
    ap = argparse.ArgumentParser(description="Live web viewer for miniagent session logs.")
    ap.add_argument("--dir", default="runs", help="log directory to watch (default: runs)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    LOG_DIR = Path(args.dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    webui.serve(Handler, args.host, args.port, f"{LOG_DIR}/")


if __name__ == "__main__":
    main()

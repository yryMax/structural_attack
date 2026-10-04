"""Web viewer for ToolCase cases: read each case the way the model gets it, and run it through Gemini.

    python -m ui.caseviewer                 # serve ./cases at http://localhost:8766
    python -m ui.caseviewer --dir cases --port 8766

Each cases/*.json is read as an injector.ToolCase {call, body|benign, question}. Pick one in
the sidebar to see it laid out as the conversation the model sees:

  1. the tool call   2. the tool result (one card per record)   3. the user's task

then click "Ask Gemini" (gemini.ask via ToolCase.ask) to add 4. the model's answer. The result
can also be shown as JSON or as the exact prompt sent. The page reloads a case when its file
changes on disk. Shares layout, theme and server code with viewer.py (webui.py).
"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from injector import ToolCase
import gemini
from . import webui

CASE_DIR = Path("cases")  # overridden in main()


CSS = r"""
.step { display: flex; align-items: center; gap: 8px; margin: 22px 0 6px; font-size: 12px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.step:first-of-type { margin-top: 6px; }
.step .n { display: inline-grid; place-items: center; width: 20px; height: 20px; border-radius: 50%;
  background: var(--accent); color: #fff; font-size: 11px; letter-spacing: 0; }
.records { display: grid; gap: 8px; margin-top: 6px; }
.record { border: 1px solid var(--border); border-radius: 8px; background: var(--bg); padding: 8px 10px; }
.record .head { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; margin-bottom: 4px; }
.record .idx { color: var(--muted); font-size: 12px; }
.record .when { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; color: var(--accent); }
.record .what { font-weight: 600; }
.kv { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 3px 14px; font-size: 13px; }
.kv .k { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; padding-top: 1px; }
.kv .v { white-space: pre-wrap; word-break: break-word; min-width: 0; }
.empties { color: var(--muted); font-size: 12px; margin-top: 4px; }
.pl { font-size: 12px; font-weight: 700; color: var(--muted); margin: 10px 0 0; }
.answer-meta { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.answer-meta .stale { color: var(--danger); }
.rec.err { background: var(--dangerbg); color: var(--danger); }
.md code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; background: var(--codebg);
  padding: 0 4px; border-radius: 4px; }
.placeholder { color: var(--muted); font-style: italic; }
"""

BODY = r"""
<div class="layout">
  <aside>
    <h1>Cases</h1>
    <div id="cases"></div>
  </aside>
  <main>
    <div class="toolbar">
      <span class="title" id="title">Pick a case →</span>
      <span class="live" id="model"></span>
      <button class="tb on" id="ask" disabled>Ask Gemini</button>
    </div>
    <div id="view" class="content"><div class="empty">No case selected.</div></div>
  </main>
</div>
"""

JS = r"""
let current = null;          // case file name
let data = null;             // last loaded case payload
let mode = "cards";          // how the tool result is shown: cards | json | prompt
const answers = new Map();   // case name -> [{text, error, model, ms, at, mtime}], newest first
const asking = new Set();    // case names with a request in flight

const TITLE_KEYS = ["summary", "subject", "title", "name", "filename"];

function splitCall(call) {
  const m = /^([\w.]+)\((.*)\)\s*$/s.exec(call || "");
  return m ? [m[1], m[2]] : [call || "", null];
}

// --- values -------------------------------------------------------------------------

const ISO = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})(?::\d{2}(?:\.\d+)?)?(Z|[+-]\d{2}:\d{2})?$/;
function fmtTime(s) {
  const m = ISO.exec(s);
  return m ? m[1] + " " + m[2] + (m[3] ? " (" + m[3] + ")" : "") : s;
}
function timeRange(rec) {
  const a = ISO.exec(typeof rec.start === "string" ? rec.start : ""), b = ISO.exec(typeof rec.end === "string" ? rec.end : "");
  if (!a) return null;
  if (!b) return a[2];
  return a[1] === b[1] ? a[2] + " – " + b[2] : a[1] + " " + a[2] + " – " + b[1] + " " + b[2];
}
const isEmpty = v => v == null || v === "" || (Array.isArray(v) && !v.length) || (typeof v === "object" && !Array.isArray(v) && !Object.keys(v).length);

function valueNode(v) {
  const box = el("div", "v");
  if (typeof v === "string") {
    box.textContent = fmtTime(v.replace(/\r\n/g, "\n"));
  } else if (Array.isArray(v)) {
    for (const x of v) {
      const line = el("span", "sub", x && typeof x === "object" ? "• " + compactArgs(x, 60).replace(/=/g, ": ") : "• " + esc(x));
      box.appendChild(line);
    }
  } else if (v && typeof v === "object") {
    box.textContent = compactArgs(v, 60).replace(/=/g, ": ");
  } else box.textContent = esc(v);
  return box;
}

// --- tool result views ----------------------------------------------------------------

function renderCards(body) {
  const wrap = el("div", "records");
  body.forEach((rec, i) => {
    const card = el("div", "record");
    const titleKey = TITLE_KEYS.find(k => typeof rec[k] === "string" && rec[k]);
    const range = timeRange(rec);
    const head = el("div", "head");
    head.appendChild(el("span", "idx", "#" + (i + 1)));
    if (range) head.appendChild(el("span", "when", range));
    head.appendChild(el("span", "what", titleKey ? rec[titleKey] : (rec.id || "record")));
    card.appendChild(head);

    const kv = el("div", "kv"), empties = [];
    const skip = new Set([titleKey, range ? "start" : null, range ? "end" : null]);
    for (const [k, v] of Object.entries(rec)) {
      if (isEmpty(v)) { empties.push(k); continue; }
      if (skip.has(k)) continue;
      kv.appendChild(el("div", "k", k));
      kv.appendChild(valueNode(v));
    }
    if (range) {  // keep the exact timestamps findable without the header's shorthand
      kv.appendChild(el("div", "k", "start → end"));
      kv.appendChild(el("div", "v mono", fmtTime(rec.start) + " → " + fmtTime(rec.end || "")));
    }
    card.appendChild(kv);
    if (empties.length) card.appendChild(el("div", "empties", "empty: " + empties.join(", ")));
    wrap.appendChild(card);
  });
  return wrap;
}

function renderResult(d) {
  const c = el("div", "rec result");
  const isList = Array.isArray(d.body);
  const cardable = isList && d.body.every(r => r && typeof r === "object" && !Array.isArray(r));
  const n = isList ? d.body.length + (d.body.length === 1 ? " record" : " records") : typeof d.body;
  c.appendChild(el("div", "role", "tool → agent · " + n));

  const m = cardable ? mode : (mode === "cards" ? "json" : mode);
  const json = JSON.stringify(d.body, null, 2);
  if (m === "cards") c.appendChild(renderCards(d.body));
  else if (m === "json") c.appendChild(el("pre", "raw", json));
  else {
    c.appendChild(el("div", "pl", "System instruction"));
    c.appendChild(el("pre", "raw", d.system));
    c.appendChild(el("div", "pl", "Prompt"));
    c.appendChild(el("pre", "raw", d.prompt));
  }

  const bar = el("div", "rawbar");
  const views = [["cards", "Readable"], ["json", "JSON"], ["prompt", "Exact prompt"]];
  for (const [key, label] of views) {
    if (key === "cards" && !cardable) continue;
    const b = el("button", key === m ? "on" : null, label);
    b.onclick = () => { mode = key; render(); };
    bar.appendChild(b);
  }
  bar.appendChild(copyButton(() => m === "prompt" ? d.prompt : json, m === "prompt" ? "Copy prompt" : "Copy JSON"));
  bar.appendChild(el("span", null, (m === "prompt" ? d.prompt.length : json.length) + " chars"));
  c.appendChild(bar);
  return c;
}

// --- answers ----------------------------------------------------------------------------

// just enough markdown for model answers: **bold**, `code`, "* " bullets
function mdNode(text) {
  const box = el("div", "body md");
  const lines = esc(text).replace(/\r\n/g, "\n").trim().split("\n");
  lines.forEach((line, li) => {
    line = line.replace(/^(\s*)[*-]\s+/, "$1• ");
    for (const part of line.split(/(\*\*[^*]+\*\*|`[^`]+`)/)) {
      if (/^\*\*[^*]+\*\*$/.test(part)) box.appendChild(el("strong", null, part.slice(2, -2)));
      else if (/^`[^`]+`$/.test(part)) box.appendChild(el("code", null, part.slice(1, -1)));
      else if (part) box.appendChild(document.createTextNode(part));
    }
    if (li < lines.length - 1) box.appendChild(document.createTextNode("\n"));
  });
  return box;
}

function renderAnswers(d) {
  const frag = document.createDocumentFragment();
  const list = answers.get(d.name) || [];
  if (asking.has(d.name)) {
    const c = el("div", "rec");
    c.appendChild(el("div", "role", "model"));
    c.appendChild(el("div", "placeholder", "asking " + (d.model || "Gemini") + "…"));
    frag.appendChild(c);
  }
  if (!list.length && !asking.has(d.name)) {
    const c = el("div", "rec");
    c.appendChild(el("div", "placeholder", "Not asked yet. Click “Ask Gemini” to send the prompt above and see what the model says. Ask again to compare runs."));
    frag.appendChild(c);
  }
  list.forEach((a, k) => {
    const c = el("div", "rec" + (a.error ? " err" : ""));
    const meta = el("div", "answer-meta", "run " + (list.length - k) + " · " + a.at + (a.model ? " · " + a.model : "") + (a.ms != null ? " · " + (a.ms / 1000).toFixed(1) + "s" : ""));
    if (a.mtime !== d.mtime) meta.appendChild(el("span", "stale", " · case file changed since this run"));
    c.appendChild(meta);
    if (a.error) c.appendChild(el("div", "body", "Error: " + a.error));
    else {
      c.appendChild(mdNode(a.text || "(empty answer)"));
      const bar = el("div", "rawbar");
      bar.appendChild(copyButton(() => a.text));
      c.appendChild(bar);
    }
    frag.appendChild(c);
  });
  return frag;
}

// --- page -------------------------------------------------------------------------------

function step(n, label) {
  const s = el("div", "step");
  s.appendChild(el("span", "n", String(n)));
  s.appendChild(document.createTextNode(label));
  return s;
}

function render() {
  const view = document.getElementById("view");
  const title = document.getElementById("title");
  const ask = document.getElementById("ask");
  view.innerHTML = "";
  title.innerHTML = "";
  if (!data) { title.textContent = "Pick a case →"; ask.disabled = true; return; }

  title.textContent = data.name.replace(/\.json$/, "");
  document.getElementById("model").textContent = data.model || "";
  if (data.error) {
    ask.disabled = true;
    const c = el("div", "rec err");
    c.appendChild(el("div", "role danger", "can't read this case"));
    c.appendChild(el("div", "body", data.error));
    view.appendChild(c);
    return;
  }
  ask.disabled = asking.has(data.name);
  ask.textContent = asking.has(data.name) ? "Asking…" : ((answers.get(data.name) || []).length ? "Ask again" : "Ask Gemini");

  view.appendChild(step(1, "Agent calls a tool"));
  const call = el("div", "rec tool");
  call.appendChild(el("div", "role", "agent → tool"));
  const [fn, args] = splitCall(data.call);
  const line = el("div", "call");
  line.appendChild(el("span", "fn", fn));
  if (args != null) line.appendChild(document.createTextNode("(" + args + ")"));
  call.appendChild(line);
  view.appendChild(call);

  view.appendChild(step(2, "Tool returns"));
  view.appendChild(renderResult(data));

  view.appendChild(step(3, "User asks"));
  const q = el("div", "rec user");
  q.appendChild(el("div", "role", "user"));
  q.appendChild(el("div", "body", data.question));
  view.appendChild(q);

  view.appendChild(step(4, "Model answers"));
  view.appendChild(renderAnswers(data));
}

async function ask() {
  if (!data || asking.has(data.name)) return;
  const d = data, name = d.name;
  asking.add(name);
  render();
  const started = Date.now();
  let entry;
  try {
    const r = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
    const res = await r.json();
    entry = r.ok ? { text: res.answer, model: res.model, ms: res.ms } : { error: res.error };
  } catch (e) {
    entry = { error: String(e), ms: Date.now() - started };
  }
  entry.at = new Date().toTimeString().slice(0, 8);
  entry.mtime = d.mtime;
  if (!answers.has(name)) answers.set(name, []);
  answers.get(name).unshift(entry);
  asking.delete(name);
  if (data && data.name === name) render();
}

let listCache = [];
async function loadCases() {
  const list = await (await fetch("/api/cases")).json();
  listCache = list;
  renderSidebar(document.getElementById("cases"),
    list.map(c => ({
      key: c.name,
      name: c.name.replace(/\.json$/, ""),
      meta: c.error ? "⚠ " + c.error : splitCall(c.call)[0] + " · " + c.records + (c.records === 1 ? " record" : " records") + " · " + c.mtime,
    })),
    current, selectCase);
  return list;
}

async function loadCase(name) {
  const r = await fetch("/api/case?name=" + encodeURIComponent(name));
  const d = await r.json();
  if (current !== name) return;
  data = r.ok ? d : { name, error: d.error };
  render();
}

function selectCase(name) {
  current = name;
  try { history.replaceState(null, "", "#" + encodeURIComponent(name)); } catch (e) {}
  loadCases();
  loadCase(name);
}

document.getElementById("ask").onclick = ask;
document.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); ask(); }
});

(async () => {
  const list = await loadCases();
  const fromHash = decodeURIComponent(location.hash.slice(1));
  const first = list.find(c => c.name === fromHash) || list[0];
  if (first) selectCase(first.name);
})();

// pick up edits to the case files (e.g. injector.py writing a new *_ill.json)
setInterval(async () => {
  const list = await loadCases();
  const cur = list.find(c => c.name === current);
  if (cur && data && cur.mtime_raw !== data.mtime) loadCase(current);
}, 2000);
"""

PAGE = webui.page("ToolCase viewer", CSS, BODY, JS)


def _load_case(path: Path) -> ToolCase:
    d = json.loads(path.read_text(encoding="utf-8"))
    body = d["body"] if "body" in d else d["benign"]   # accept either key
    return ToolCase(d["call"], body, d["question"])


def _cases_payload():
    items = []
    for p in sorted(CASE_DIR.glob("*.json")):
        item = {"name": p.name, "mtime": webui.mtime_str(p), "mtime_raw": p.stat().st_mtime}
        try:
            case = _load_case(p)
            item.update(call=case.call, records=len(case.body) if isinstance(case.body, list) else 1)
        except (OSError, KeyError, ValueError) as e:
            item["error"] = f"bad case file: {e}"
        items.append(item)
    return items


def _case_payload(p: Path):
    case = _load_case(p)
    return {"name": p.name, "call": case.call, "question": case.question, "body": case.body,
            "prompt": case.build_prompt(), "system": gemini.SYSTEM, "model": gemini.MODEL,
            "mtime": p.stat().st_mtime}


class Handler(webui.JSONHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(PAGE)
        elif parsed.path == "/api/cases":
            self._json(_cases_payload())
        elif parsed.path == "/api/case":
            p = webui.safe_file(CASE_DIR, parse_qs(parsed.query).get("name", [""])[0], ".json")
            if p is None:
                return self._json({"error": "not found"}, code=404)
            try:
                self._json(_case_payload(p))
            except (KeyError, ValueError) as e:
                self._json({"error": f"bad case file: {e}"}, code=400)
        else:
            self._not_found()

    def do_POST(self):
        if urlparse(self.path).path != "/api/ask":
            return self._not_found()
        try:
            name = self._body()["name"]
        except (ValueError, KeyError, TypeError) as e:
            return self._json({"error": f"bad request: {e}"}, code=400)
        p = webui.safe_file(CASE_DIR, name, ".json")
        if p is None:
            return self._json({"error": "not found"}, code=404)
        try:
            t0 = time.monotonic()
            answer = _load_case(p).ask()
            self._json({"answer": answer, "model": gemini.MODEL, "ms": round((time.monotonic() - t0) * 1000)})
        except Exception as e:  # network / API / key errors
            self._json({"error": f"{type(e).__name__}: {e}"}, code=500)


def main() -> None:
    global CASE_DIR
    ap = argparse.ArgumentParser(description="Web viewer for ToolCase cases.")
    ap.add_argument("--dir", default="cases", help="case directory (default: cases)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    CASE_DIR = Path(args.dir)
    webui.serve(Handler, args.host, args.port, f"{CASE_DIR}/")


if __name__ == "__main__":
    main()

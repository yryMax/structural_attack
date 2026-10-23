"""Web viewer for AgentDojo attack runs: every injection case, with the injected tool output colored by part.

    python -m ui.injviewer                  # serve ./runs at http://localhost:8767
    python -m ui.injviewer --dir runs --port 8767

Pick a run and a case (suite × user task × injection task) in the sidebar to see the conversation
the agent had. Each tool output that carries an injection is drawn as the text the model read,
split into the parts of a structural attack:

  benign     the tool's real data around the injection
  mimic      injected text that continues the benign data
  close      closes the current tool-response turn ("}],"role":"tool"…}  /  <|im_end|>  /  </>)
  exec       forged assistant / tool turns (the "finished" benign task)
  user       forged user turns
  malicious  the forged turn that carries the injection task's GOAL (GOAL itself underlined)
  system     forged system turns
  reopen     re-opens a tool-response turn so the real tail of the output stays well formed
  instruction an injection with no turn structure (e.g. important_instructions)

Turn boundaries are found for ChatML (<|im_start|>role), GLM (<|role|>), tag style (<role>, </>)
and serialized OpenAI messages ("},{"role":…). "Readable" undoes literal \\n \\t \\" escapes;
"Exact" shows the characters the model got. Shares layout, theme and server code with webui.py.
"""

import argparse
import re
import threading
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json
from . import webui

RUNS_DIR = Path("runs")  # overridden in main()


CSS = r"""
:root {
  --k-benign-bg: #bbf7d0; --k-benign: #14532d;
  --k-mimic: #166534; --k-close: #0284c7; --k-exec: #c2410c; --k-user: #a16207;
  --k-malicious: #dc2626; --k-system: #7c3aed; --k-reopen: #c026d3; --k-instruction: #dc2626;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --k-benign-bg: #14532d; --k-benign: #dcfce7;
    --k-mimic: #86efac; --k-close: #38bdf8; --k-exec: #fb923c; --k-user: #facc15;
    --k-malicious: #f87171; --k-system: #a78bfa; --k-reopen: #e879f9; --k-instruction: #f87171;
  }
}
aside { width: 320px; flex-basis: 320px; }
.filters { display: grid; gap: 6px; margin: 0 4px 10px; }
.filters select, .filters input { width: 100%; font: inherit; font-size: 13px; padding: 5px 8px; border-radius: 7px;
  border: 1px solid var(--border); background: var(--bg); color: var(--fg); }
.filters .count { font-size: 12px; color: var(--muted); }
.content { max-width: 1100px; }
.badge { display: inline-block; font-size: 12px; font-weight: 700; padding: 1px 8px; border-radius: 999px; border: 1px solid var(--border); }
.badge.yes { background: var(--dangerbg); color: var(--danger); border-color: transparent; }
.badge.no { background: var(--result); color: #16a34a; border-color: transparent; }
.badge.na { color: var(--muted); }
.meta-grid { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 3px 14px; font-size: 13px; }
.meta-grid .k { color: var(--muted); }
.goal { color: var(--k-malicious); font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12px; margin: 10px 0 4px; }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.legend i { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.flow { white-space: pre-wrap; word-break: break-word; font: 12.5px/1.6 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin-top: 6px;
  max-height: 60em; overflow-y: auto; }
.seg { border-radius: 3px; }
.seg.benign { background: var(--k-benign-bg); color: var(--k-benign); }
.seg.mimic { background: var(--k-benign-bg); color: var(--k-mimic); text-decoration: underline dotted; text-underline-offset: 3px; }
.seg.close { color: var(--k-close); } .seg.exec { color: var(--k-exec); } .seg.user { color: var(--k-user); }
.seg.malicious { color: var(--k-malicious); } .seg.system { color: var(--k-system); }
.seg.reopen { color: var(--k-reopen); } .seg.instruction { color: var(--k-instruction); }
.seg mark { background: none; color: inherit; font-weight: 700; text-decoration: underline; text-underline-offset: 3px; }
.chip { display: inline-block; font: 600 10px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  padding: 0 5px; margin: 0 3px 0 1px; border-radius: 4px; color: #fff; vertical-align: 1px; user-select: none; }
.nl { opacity: .45; user-select: none; }
details.more summary { cursor: pointer; color: var(--accent); font-size: 12px; }
.vec { font-size: 12px; color: var(--muted); margin-top: 4px; }
"""

BODY = r"""
<div class="layout">
  <aside>
    <h1>Injection cases</h1>
    <div class="filters">
      <select id="run"></select>
      <select id="suite"><option value="">all suites</option></select>
      <select id="outcome">
        <option value="">all outcomes</option>
        <option value="asr">attack succeeded</option>
        <option value="noasr">attack failed</option>
        <option value="util">utility kept</option>
        <option value="noeval">not evaluated</option>
      </select>
      <input id="q" placeholder="filter, e.g. ut_3 it_5">
      <div class="count" id="count"></div>
    </div>
    <div id="cases"></div>
  </aside>
  <main>
    <div class="toolbar">
      <span class="title" id="title">Pick a case →</span>
      <span id="badges"></span>
      <button class="tb on" id="readable">Readable</button>
      <button class="tb" id="exact">Exact</button>
    </div>
    <div id="view" class="content"><div class="empty">No case selected.</div></div>
  </main>
</div>
"""

JS = r"""
let runs = [], run = null, cases = [], current = null, data = null, readable = true;

const KINDS = {
  benign: "benign data", mimic: "mimicked benign data", close: "closes tool turn", exec: "forged execution turns",
  user: "forged user turn", malicious: "malicious task", system: "forged system turn",
  reopen: "re-opens tool turn", instruction: "injected instruction",
};
const css = name => getComputedStyle(document.documentElement).getPropertyValue("--k-" + name).trim();
const short = s => s.replace("user_task_", "ut_").replace("injection_task_", "it_");

// undo literal escapes, however deeply nested (\\\\n, \\n, \n …)
function unescape(s) {
  return s.replace(/\\+u([0-9a-fA-F]{4})/g, (_, h) => String.fromCharCode(parseInt(h, 16)))
          .replace(/\\+n/g, "\n").replace(/\\+t/g, "\t").replace(/\\+r/g, "")
          .replace(/\\+"/g, '"').replace(/\\+'/g, "'").replace(/\\{2,}/g, "\\");
}

// red when the outcome favors the attacker (ASR ✓, utility ✗), green otherwise
function badge(label, v, bad) {
  if (v == null) return el("span", "badge na", label + " –");
  return el("span", "badge " + (v === bad ? "yes" : "no"), label + (v ? " ✓" : " ✗"));
}

// --- sidebar ------------------------------------------------------------------------------

function caseLabel(c) { return c.suite + " · " + short(c.user_task) + " × " + short(c.injection_task); }

function filtered() {
  const suite = document.getElementById("suite").value, oc = document.getElementById("outcome").value;
  const words = document.getElementById("q").value.toLowerCase().split(/\s+/).filter(Boolean);
  return cases.filter(c => {
    if (suite && c.suite !== suite) return false;
    if (oc === "asr" && c.security !== true) return false;
    if (oc === "noasr" && c.security !== false) return false;
    if (oc === "util" && c.utility !== true) return false;
    if (oc === "noeval" && c.utility != null) return false;
    const label = caseLabel(c).toLowerCase() + " " + c.key.toLowerCase();
    return words.every(w => label.includes(w));
  });
}

function drawList() {
  const list = filtered();
  const asr = list.filter(c => c.security === true).length, done = list.filter(c => c.security != null).length;
  document.getElementById("count").textContent =
    list.length + " cases · ASR " + (done ? (100 * asr / done).toFixed(1) + "% (" + asr + "/" + done + ")" : "–");
  renderSidebar(document.getElementById("cases"), list.map(c => ({
    key: c.key, name: caseLabel(c),
    meta: c.utility == null ? "⚠ not evaluated" : "ASR " + (c.security ? "✓" : "✗") + " · utility " + (c.utility ? "✓" : "✗"),
  })), current, selectCase);
}

async function loadRuns() {
  runs = await (await fetch("/api/runs")).json();
  const sel = document.getElementById("run");
  sel.innerHTML = "";
  for (const r of runs) {
    const o = el("option", null, r.name + "  (" + r.attack + ")");
    o.value = r.name;
    sel.appendChild(o);
  }
}

async function selectRun(name, keepCase) {
  run = name;
  document.getElementById("run").value = name;
  document.getElementById("cases").innerHTML = "";
  document.getElementById("count").textContent = "loading…";
  const r = await fetch("/api/cases?run=" + encodeURIComponent(name));
  if (run !== name) return;
  cases = r.ok ? await r.json() : [];
  const suites = [...new Set(cases.map(c => c.suite))].sort();
  const ss = document.getElementById("suite"), keep = ss.value;
  ss.innerHTML = '<option value="">all suites</option>';
  for (const s of suites) { const o = el("option", null, s); o.value = s; ss.appendChild(o); }
  ss.value = suites.includes(keep) ? keep : "";
  const pick = keepCase && cases.find(c => c.key === keepCase) ? keepCase : (filtered()[0] || {}).key;
  if (pick) selectCase(pick); else { current = null; data = null; drawList(); render(); }
}

function selectCase(key) {
  current = key;
  try { history.replaceState(null, "", "#" + encodeURIComponent(run) + "/" + encodeURIComponent(key)); } catch (e) {}
  drawList();
  loadCase(run, key);
}

async function loadCase(r, key) {
  const res = await fetch("/api/case?run=" + encodeURIComponent(r) + "&key=" + encodeURIComponent(key));
  const d = await res.json();
  if (current !== key || run !== r) return;
  data = res.ok ? d : { key, error: d.error };
  render();
}

// --- case ---------------------------------------------------------------------------------

function textNode(s) {
  // in Readable mode, show where a literal newline escape was
  if (!readable) return document.createTextNode(s);
  const frag = document.createDocumentFragment();
  const parts = s.split(/(\\+n)/);
  for (const p of parts) {
    if (/^\\+n$/.test(p)) { frag.appendChild(el("span", "nl", "↵")); frag.appendChild(document.createTextNode("\n")); }
    else if (p) frag.appendChild(document.createTextNode(unescape(p)));
  }
  return frag;
}

function segNode(seg) {
  const span = el("span", "seg " + seg.kind);
  if (seg.kind !== "benign" && seg.kind !== "mimic") {
    const name = !seg.role ? KINDS[seg.kind] : seg.kind === "close" ? "close " + seg.role : seg.role + (seg.kind === "malicious" ? " ⚠" : "");
    const chip = el("span", "chip", name);
    chip.style.background = css(seg.kind);
    chip.title = KINDS[seg.kind];
    span.appendChild(chip);
  }
  let at = 0;
  for (const [a, b] of seg.goal || []) {
    span.appendChild(textNode(seg.text.slice(at, a)));
    const m = el("mark");
    m.appendChild(textNode(seg.text.slice(a, b)));
    span.appendChild(m);
    at = b;
  }
  span.appendChild(textNode(seg.text.slice(at)));
  return span;
}

function flowNode(segs) {
  const box = el("div", "flow");
  for (const s of segs) box.appendChild(segNode(s));
  return box;
}

function legend(segs) {
  const box = el("div", "legend");
  const used = new Set(segs.map(s => s.kind));
  for (const [k, label] of Object.entries(KINDS)) {
    if (!used.has(k)) continue;
    const s = el("span");
    const sw = el("i");
    sw.style.background = k === "benign" || k === "mimic" ? css("benign-bg") : css(k);
    s.appendChild(sw);
    s.appendChild(document.createTextNode(label));
    box.appendChild(s);
  }
  return box;
}

function plain(text, limit = 1500) {
  const box = el("div", "body");
  if (text.length <= limit) { box.textContent = text; return box; }
  box.textContent = text.slice(0, limit);
  const det = el("details", "more");
  det.appendChild(el("summary", null, "show all " + text.length + " chars"));
  det.appendChild(el("div", "body", text.slice(limit)));
  box.appendChild(det);
  return box;
}

function callLine(c) {
  const line = el("div", "call");
  line.appendChild(el("span", "fn", c.function));
  line.appendChild(document.createTextNode("(" + compactArgs(c.args, 120) + ")"));
  return line;
}

function messageNode(m) {
  if (m.role === "system") {
    const c = el("div", "rec");
    const det = el("details", "more");
    det.appendChild(el("summary", null, "system prompt (" + m.text.length + " chars)"));
    det.appendChild(el("div", "body", m.text));
    c.appendChild(det);
    return c;
  }
  if (m.role === "user") {
    const c = el("div", "rec user");
    c.appendChild(el("div", "role", "user"));
    c.appendChild(el("div", "body", m.text));
    return c;
  }
  if (m.role === "assistant") {
    const c = el("div", "rec");
    c.appendChild(el("div", "role", "assistant"));
    if (m.text) c.appendChild(plain(m.text, 4000));
    for (const t of m.tool_calls || []) c.appendChild(callLine(t));
    return c;
  }
  const c = el("div", "rec " + (m.segments ? "" : "tool"));
  const call = m.call ? m.call.function + "(" + compactArgs(m.call.args, 80) + ")" : "";
  c.appendChild(el("div", "role" + (m.segments ? " danger" : ""), "tool → agent · " + call + (m.segments ? " · injected" : "") + (m.error ? " · error" : "")));
  if (m.segments) {
    c.appendChild(legend(m.segments));
    c.appendChild(flowNode(m.segments));
    c.appendChild(el("div", "vec", "injection vector" + (m.vectors.length > 1 ? "s" : "") + ": " + m.vectors.join(", ")));
  } else c.appendChild(plain(m.error || m.text));
  return c;
}

function render() {
  const view = document.getElementById("view"), title = document.getElementById("title"), badges = document.getElementById("badges");
  view.innerHTML = ""; badges.innerHTML = "";
  if (!data) { title.textContent = "Pick a case →"; view.appendChild(el("div", "empty", "No case selected.")); return; }
  if (data.error && !data.messages) {
    title.textContent = data.key;
    const c = el("div", "rec err");
    c.appendChild(el("div", "role danger", "can't read this case"));
    c.appendChild(el("div", "body", data.error));
    view.appendChild(c);
    return;
  }
  title.textContent = run + " · " + data.suite + " · " + short(data.user_task) + " × " + short(data.injection_task);
  badges.appendChild(badge("ASR", data.security, true));
  badges.appendChild(document.createTextNode(" "));
  badges.appendChild(badge("utility", data.utility, false));

  const head = el("div", "rec");
  const g = el("div", "meta-grid");
  const row = (k, v, cls) => { g.appendChild(el("div", "k", k)); g.appendChild(el("div", cls || null, v)); };
  row("attack", data.attack);
  row("injection goal", data.goal || "(unknown)", "goal");
  if (data.duration != null) row("duration", data.duration.toFixed(1) + " s");
  if (data.run_error) row("run error", data.run_error, "goal");
  head.appendChild(g);
  view.appendChild(head);

  for (const m of data.messages) view.appendChild(messageNode(m));

  if (data.unseen.length) {
    const c = el("div", "rec");
    c.appendChild(el("div", "role", "injection vectors not found in any tool output (not read by the agent, or reformatted beyond matching)"));
    for (const u of data.unseen) {
      const det = el("details", "more");
      det.appendChild(el("summary", null, u.vector + " (" + u.length + " chars)"));
      det.appendChild(flowNode(u.segments));
      c.appendChild(det);
    }
    view.appendChild(c);
  }
}

function setMode(r) {
  readable = r;
  document.getElementById("readable").classList.toggle("on", r);
  document.getElementById("exact").classList.toggle("on", !r);
  render();
}

document.getElementById("readable").onclick = () => setMode(true);
document.getElementById("exact").onclick = () => setMode(false);
document.getElementById("run").onchange = e => selectRun(e.target.value);
for (const id of ["suite", "outcome", "q"]) document.getElementById(id).oninput = drawList;
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || !["j", "k"].includes(e.key)) return;
  const list = filtered(), i = list.findIndex(c => c.key === current);
  const next = list[Math.max(0, Math.min(list.length - 1, i + (e.key === "j" ? 1 : -1)))];
  if (next && next.key !== current) selectCase(next.key);
});

(async () => {
  await loadRuns();
  const [hRun, ...rest] = decodeURIComponent(location.hash.slice(1)).split("/");
  const first = runs.find(r => r.name === hRun) || runs[0];
  if (first) selectRun(first.name, rest.join("/"));
})();
"""

PAGE = webui.page("Injection viewer", CSS, BODY, JS)


# --- locating an injection in a tool output -------------------------------------------------

_ESC = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "'": "'", "\\": "\\", "/": "/"}


def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", lambda m: _ESC.get(m.group(1), m.group(0)), s)


# one escape sequence (any number of backslashes deep), or one whitespace char
_CANON = re.compile(r"\\+(u[0-9a-fA-F]{4}|.)|\s", re.S)


def _canon(s: str) -> tuple[str, list[int], list[int]]:
    """s reduced to its visible characters: escapes undone at any depth, and all whitespace dropped,
    real or escaped (\\n, YAML's "\\ " and line continuations). Also each kept char's span in s."""
    out, a, b, pos = [], [], [], 0
    for m in _CANON.finditer(s):
        for i in range(pos, m.start()):
            out.append(s[i]); a.append(i); b.append(i + 1)
        pos = m.end()
        g = m.group(1)
        if g is None or g in "ntr" or g.isspace():
            continue
        out.append(chr(int(g[1:], 16)) if len(g) == 5 else g); a.append(m.start()); b.append(m.end())
    for i in range(pos, len(s)):
        out.append(s[i]); a.append(i); b.append(i + 1)
    return "".join(out), a, b


def locate(text: str, needle: str, start: int = 0):
    """(a, b) such that text[a:b] is needle as it landed in text (searching from text[start:]), or None.

    The environment YAML folds lines and some tools return YAML or a repr(), so the injection can
    reach the model with extra escapes and different whitespace: both sides are compared by their
    visible characters only. YAML folding can still split an escape sequence, so a needle that
    isn't found whole is matched by 40-char anchors near its two ends."""
    hay, a, b = _canon(text)
    n = _canon(needle)[0]
    if not n:
        return None
    s = next((k for k, i in enumerate(a) if i >= start), len(hay))
    j = hay.find(n, s)
    if j >= 0:
        return a[j], b[j + len(n) - 1]
    w, slack = 40, min(len(n) // 2, 400)
    if len(n) < 2 * w:
        return None
    for k in range(0, slack, w):  # first undamaged anchor from the front
        ja = hay.find(n[k:k + w], s)
        if ja >= 0:
            ja = max(s, ja - k)
            break
    else:
        return None
    for k in range(0, slack, w):  # … and from the back, within a plausible length
        jb = hay.find(n[len(n) - k - w:len(n) - k], ja + len(n) // 2, ja + 2 * len(n))
        if jb >= 0:
            return a[ja], b[min(len(hay), jb + w + k) - 1]
    return None


# --- splitting an injection into its parts --------------------------------------------------

# whitespace as it can appear between tokens: real, escaped (\n), or a YAML line fold ("\⏎    \ ")
_G = r"(?:\s|\\+(?=\s)|\\+[ntr](?![a-z]))*"
_Q = r"\\*\""  # a quote, escaped any number of times
# where a forged turn starts: ChatML, GLM, tag style, or the gap between two serialized OpenAI messages
_OPENER = re.compile(
    rf"<\|im_start\|>{_G}(?P<chatml>\w+)"
    r"|<\|(?P<glm>system|user|assistant|observation)\|>"
    r"|<(?P<tag>system|user|assistant|tool_response|tool|observation)>"
    rf"|(?<=\}}){_G},{_G}(?=\{{{_G}{_Q}(?:role|content|tool_call_id){_Q})"
)
_JSON_ROLE = re.compile(rf"{_Q}role{_Q}{_G}:{_G}{_Q}{_G}(\w+)")
# the end of the turn being read: a closing JSON string, or a run of chat-template tags
_JSON_CLOSE = re.compile(rf"{_Q}{_G}\}}{_G}\]|{_Q}{_G},{_G}{_Q}(?:tool_call_id|role|name){_Q}|{_Q}{_G}\}}")
_TAG_TAIL = re.compile(rf"(?:{_G}(?:<\|[^|<>]+\|>|</[\w]*>|<\w+>|\[gMASK\]|<sop>))+{_G}$")
_TOOL_ROLES = {"tool", "tool_response", "observation", "ipython", "function"}


def _turns(text: str) -> list[tuple[int, str | None, bool]]:
    """(start, role, is_json) of each forged turn in text, in order."""
    out = []
    for m in _OPENER.finditer(text):
        role = m.group("chatml") or m.group("glm") or m.group("tag")
        if role is None:  # JSON: the role is a field somewhere in the object
            nxt = _OPENER.search(text, m.end())
            r = _JSON_ROLE.search(text, m.end(), nxt.start() if nxt else len(text))
            out.append((m.end(), r.group(1) if r else None, True))
        else:
            out.append((m.start(), role, False))
    return out


def segment(text: str, goal: str | None) -> list[dict]:
    """Split one injection (as the model read it) into colored parts: [{kind, role, text, goal}]."""
    turns = _turns(text)
    if not turns:
        segs = [{"kind": "instruction", "role": None, "text": text}]
    else:
        segs = []
        head = text[:turns[0][0]]
        if turns[0][2]:
            m = _JSON_CLOSE.search(head)
        else:
            m = _TAG_TAIL.search(head)
        cut = m.start() if m else len(head)
        if head[:cut].strip():
            segs.append({"kind": "mimic", "role": None, "text": head[:cut]})
        if head[cut:]:
            r = _JSON_ROLE.search(head, cut)
            segs.append({"kind": "close", "role": r.group(1) if r else None, "text": head[cut:]})
        for i, (a, role, _) in enumerate(turns):
            b = turns[i + 1][0] if i + 1 < len(turns) else len(text)
            role_n = "tool" if role in _TOOL_ROLES else role
            last = i == len(turns) - 1
            if last and i > 0 and role_n in ("tool", None):
                kind = "reopen"
            elif goal and locate(text[a:b], goal):
                kind = "malicious"
            elif role_n == "user":
                kind = "user"
            elif role_n == "system":
                kind = "system"
            else:
                kind = "exec"
            segs.append({"kind": kind, "role": role_n, "text": text[a:b]})
    for s in segs:
        hit = locate(s["text"], goal) if goal and s["kind"] in ("malicious", "instruction", "user") else None
        s["goal"] = [hit] if hit else []
        if hit and s["kind"] == "user":
            s["kind"] = "malicious"
    return segs


# --- reading runs ----------------------------------------------------------------------------

_goals: dict[tuple[str, str], str] = {}
_goals_lock = threading.Lock()


def _goal(suite: str, injection_task: str) -> str | None:
    """The injection task's GOAL, from agentdojo (slow to import on Lustre, so loaded on first use)."""
    with _goals_lock:
        if not _goals:
            try:
                from agentdojo.task_suite.load_suites import get_suites
                for name, s in get_suites("v1.2.2").items():
                    for tid, t in s.injection_tasks.items():
                        _goals[(name, tid)] = t.GOAL
            except Exception as e:  # agentdojo missing: still show the case, without the goal
                print(f"no injection goals ({type(e).__name__}: {e})")
                _goals[("", "")] = ""
        return _goals.get((suite, injection_task))


_KEY = re.compile(r"[\w.-]+/[\w.-]+/user_task_\d+/[\w.-]+/injection_task_\d+")
_TAIL = re.compile(rb'"(utility|security)":\s*(true|false)')
_summaries: dict[Path, tuple[float, dict]] = {}


def _case_paths(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.glob("*/*/user_task_*/*/injection_task_*.json") if p.parent.name != "none")


def _summary(p: Path, run_dir: Path) -> dict:
    """List entry of one case; reads only the file's tail, where agentdojo writes utility/security."""
    mtime = p.stat().st_mtime
    hit = _summaries.get(p)
    if hit and hit[0] == mtime:
        return hit[1]
    with p.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 2048))
        found = dict((k.decode(), v == b"true") for k, v in _TAIL.findall(f.read()))
    parts = p.relative_to(run_dir).with_suffix("").parts
    s = {"key": "/".join(parts), "suite": parts[1], "user_task": parts[2], "attack": parts[3],
         "injection_task": parts[4], "utility": found.get("utility"), "security": found.get("security")}
    _summaries[p] = (mtime, s)
    return s


def _run_dir(name: str):
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    d = (RUNS_DIR / name).resolve()
    return d if d.parent == RUNS_DIR.resolve() and d.is_dir() else None


def _runs_payload():
    out = []
    for d in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        attacks = sorted({p.name for p in d.glob("*/*/user_task_*/*") if p.name != "none"})
        if attacks:
            out.append({"name": d.name, "attack": ", ".join(attacks)})
    return out


def _text(content) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    return "".join(c.get("content") or c.get("text") or "" for c in content)


def _case_payload(run_dir: Path, key: str):
    d = json.loads((run_dir / (key + ".json")).read_text(encoding="utf-8"))
    suite, it = d["suite_name"], d["injection_task_id"]
    goal = _goal(suite, it)
    # attacks often put the same text in every vector: such vectors can't be told apart, so match each
    # distinct text once and name all its vectors
    injections = {}
    for vec, inj in (d.get("injections") or {}).items():
        injections[inj] = injections.get(inj, ()) + (vec,)
    seen = set()
    messages = []
    for m in d["messages"]:
        text = _text(m.get("content"))
        out = {"role": m["role"], "text": text}
        if m["role"] == "assistant":
            out["tool_calls"] = [{"function": t["function"], "args": t["args"]} for t in m.get("tool_calls") or []]
        if m["role"] == "tool":
            tc = m.get("tool_call") or {}
            out["call"] = {"function": tc.get("function", "?"), "args": tc.get("args") or {}}
            out["error"] = m.get("error")
            hits = []
            for inj, vecs in injections.items():
                at = 0
                while (r := locate(text, inj, at)) is not None:
                    hits.append((r[0], -r[1], vecs))  # longest first among equal starts
                    at = r[1]
            hits.sort()
            if hits:
                segs, pos, vecs = [], 0, []
                for a, b, names in hits:
                    b = -b
                    if a < pos:  # inside an injection already drawn (one vector's text containing another's)
                        continue
                    if a > pos:
                        segs.append({"kind": "benign", "role": None, "text": text[pos:a], "goal": []})
                    segs.extend(segment(text[a:b], goal))
                    pos = b
                    vecs.append(" / ".join(names))
                    seen.update(names)
                if pos < len(text):
                    segs.append({"kind": "benign", "role": None, "text": text[pos:], "goal": []})
                out["segments"], out["vectors"] = segs, vecs
        messages.append(out)
    unseen = [{"vector": " / ".join(n for n in vecs if n not in seen), "length": len(s), "segments": segment(_unescape(s), goal)}
              for s, vecs in injections.items() if not seen.issuperset(vecs)]
    return {"key": key, "suite": suite, "user_task": d["user_task_id"], "injection_task": it,
            "attack": d.get("attack_type"), "goal": goal, "utility": d.get("utility"), "security": d.get("security"),
            "duration": d.get("duration"), "run_error": d.get("error"), "messages": messages, "unseen": unseen}


class Handler(webui.JSONHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        if parsed.path == "/":
            return self._html(PAGE)
        if parsed.path == "/api/runs":
            return self._json(_runs_payload())
        run_dir = _run_dir(q.get("run", [""])[0])
        if parsed.path == "/api/cases":
            if run_dir is None:
                return self._json({"error": "no such run"}, code=404)
            return self._json([_summary(p, run_dir) for p in _case_paths(run_dir)])
        if parsed.path == "/api/case":
            key = q.get("key", [""])[0]
            if run_dir is None or not _KEY.fullmatch(key) or not (run_dir / (key + ".json")).is_file():
                return self._json({"error": "not found"}, code=404)
            try:
                return self._json(_case_payload(run_dir, key))
            except (KeyError, ValueError, TypeError) as e:
                return self._json({"key": key, "error": f"bad result file: {type(e).__name__}: {e}"}, code=400)
        self._not_found()


def main() -> None:
    global RUNS_DIR
    ap = argparse.ArgumentParser(description="Web viewer for the injection cases of AgentDojo attack runs.")
    ap.add_argument("--dir", default="runs", help="directory of run dirs (default: runs)")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    RUNS_DIR = Path(args.dir)
    threading.Thread(target=_goal, args=("", ""), daemon=True).start()  # warm up the agentdojo import
    webui.serve(Handler, args.host, args.port, f"{RUNS_DIR}/")


if __name__ == "__main__":
    main()

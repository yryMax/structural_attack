"""Chat editor for AgentDojo conversations: edit one turn by turn and let an OpenRouter model continue it.

    python -m ui.chateditor                 # serve ./runs at http://localhost:8769
    python -m ui.chateditor --dir runs --port 8769

Load a case of a run from the sidebar (the same list and #run/key links as injviewer.py), or start
a new chat in a suite, from AgentDojo's system prompt. Then add system / user / assistant turns (at the
end, or below the turn whose "+ below" is on), tool calls on an assistant turn (one of the suite's tools)
and a tool response for each pending call, which goes after the other tool turns of its assistant turn.
An assistant turn and its tool turns stay together: no other turn can be inserted between them.
Only the text, and a call's function and args, are edited: call ids, tool_call_id and the call copied
into its tool turn are filled in the way agentdojo's runtime does. A new tool response starts as what
agentdojo's runtime returns for the call: the case's environment (with its injections, set up for its
user task) after all earlier calls of the conversation, run in order.

"Send" on a turn sends the conversation up to that turn to the chosen model, the way agentdojo's
openai-compatible pipeline sends it (its message conversion, the suite's tools, temperature 0), and
puts the reply right after that turn, replacing all turns below it. Run files are never written.

The OpenRouter key is OPENROUTER_API_KEY, or OPENAI_COMPATIBLE_API_KEY when OPENAI_COMPATIBLE_BASE_URL
points at OpenRouter (as in .env for the runs made through it).
"""

import argparse
import functools
import json
import os
import threading
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from . import injviewer, webui

OPENROUTER = "https://openrouter.ai/api/v1"
# tool-calling models on OpenRouter one tier below each lab's flagship as of 2026-11, first one is the default
MODELS = [
    "anthropic/claude-sonnet-5.5",
    "openai/gpt-6.1-terra",
    "google/gemini-3.8-flash",
    "x-ai/grok-4.5",
    "qwen/qwen3.8-max-0902",
    "z-ai/glm-5.3",
    "moonshotai/kimi-k3",
    "deepseek/deepseek-v4.1-flash",
    "mistralai/mistral-large-4-0",
    "minimax/minimax-m3",
    # the two models vllm/serve_multinode.sh serves, here on OpenRouter
    "qwen/qwen3-235b-a22b-2507",
    "z-ai/glm-4.5",
]


CSS = r"""
aside { width: 320px; flex-basis: 320px; }
.filters { display: grid; gap: 6px; margin: 0 4px 14px; }
.filters .row { display: flex; gap: 6px; }
.filters .row select { flex: 1; }
.filters select, .filters input, .toolbar select, .tc select { font: inherit; font-size: 13px; padding: 4px 8px;
  border-radius: 7px; border: 1px solid var(--border); background: var(--bg); color: var(--fg); }
.filters .count, .info { font-size: 12px; color: var(--muted); }
.content { max-width: 1000px; }
.rec .head { display: flex; gap: 8px; align-items: center; margin-bottom: 4px; }
.rec .head .role { margin: 0 auto 0 0; }
.rec.err { background: var(--dangerbg); }
.rec textarea { display: block; width: 100%; padding: 6px 8px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--bg); color: var(--fg); resize: vertical; font: inherit; outline: none; }
.rec textarea:focus { border-color: var(--accent); }
.rec textarea.mono { font: 12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.rec textarea.bad { border-color: var(--danger); }
.tc { border-top: 1px dashed var(--border); margin-top: 8px; padding-top: 6px; display: grid; gap: 4px; }
.tc .line { display: flex; gap: 8px; align-items: center; font-size: 12px; }
.tc .id { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin-right: auto; }
.warn { color: var(--danger); font-size: 12px; }
button.sm { border: 1px solid var(--border); background: var(--bg); color: var(--accent); cursor: pointer;
  font: inherit; font-size: 12px; padding: 2px 8px; border-radius: 6px; }
button.sm:disabled { opacity: .5; cursor: default; }
button.sm.on { background: var(--accent); color: #fff; border-color: var(--accent); }
.rec > button.sm { margin-top: 8px; }
.addbar { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin: 14px 0 40px; }
.addbar.mid { margin: 8px 0; padding: 8px; border: 1px dashed var(--accent); border-radius: 10px; }
.placeholder { color: var(--muted); font-style: italic; }
"""

BODY = r"""
<div class="layout">
  <aside>
    <h1>New chat</h1>
    <div class="filters">
      <div class="row"><select id="suite"></select><button class="tb" id="new" disabled>New</button></div>
    </div>
    <h1>Load from runs</h1>
    <div class="filters">
      <select id="run"></select>
      <input id="q" placeholder="filter, e.g. banking ut_2 it_4">
      <div class="count" id="count"></div>
    </div>
    <div id="cases"></div>
  </aside>
  <main>
    <div class="toolbar">
      <span class="title" id="title"></span>
      <span class="live" id="tools"></span>
      <select id="model"></select>
    </div>
    <div id="view" class="content"></div>
  </main>
</div>
"""

JS = r"""
let meta = null;                             // {models, system, tools: {suite: [OpenAI tool definitions]}}
let runs = [], run = null, cases = [], current = null;
let title = "", suite = null, messages = [];  // the conversation, as agentdojo ChatMessages
let env = {};                                 // {user_task, injections} the tools run with
let after = null;                             // the turn new turns go below (null: at the end)
let busy = null, running = false;             // turn a request was sent from; a tool call is being run
let error = null, gen = 0;                    // last failure; bumped per chat
const info = new WeakMap();                   // reply -> "model · seconds"

const $ = id => document.getElementById(id);
const tools = () => meta.tools[suite] || [];
const text = s => [{ type: "text", content: s }];
const textOf = m => (m.content || []).map(b => b.content || "").join("\n");
const fit = ta => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight + 2, 480) + "px"; };
function btn(label, onclick, cls = "sm") { const b = el("button", cls, label); b.onclick = onclick; return b; }

// --- keeping the conversation well formed -------------------------------------------------

// a tool turn answers a call of the assistant turn before it: drop tool turns whose call is gone,
// and point each at its call so edits to the call show up in it
function sync() {
  let calls = new Map();
  messages = messages.filter(m => {
    if (m.role === "assistant") calls = new Map((m.tool_calls || []).map(t => [t.id, t]));
    if (m.role !== "tool") return true;
    m.tool_call = calls.get(m.tool_call_id);
    return !!m.tool_call;
  });
}
const answered = id => messages.some(m => m.role === "tool" && m.tool_call_id === id);
// index of the turn new turns go below
const slot = () => messages.includes(after) ? messages.indexOf(after) : messages.length - 1;
// unanswered calls of the assistant turn that turn i is, or whose tool turns turn i is in
function pendingAt(i) {
  const a = messages[messages.slice(0, i + 1).findLastIndex(m => m.role !== "tool")];
  return a && a.role === "assistant" ? (a.tool_calls || []).filter(t => !answered(t.id)) : [];
}
// index right after the tool turns of the assistant turn that makes call t, -1 if the call is gone
function blockEnd(t) {
  let i = messages.findIndex(m => (m.tool_calls || []).includes(t));
  if (i < 0) return -1;
  while (messages[i + 1] && messages[i + 1].role === "tool") i++;
  return i + 1;
}

// a new call id shaped like the conversation's last one (as its model makes them), else like OpenAI's
function newId() {
  const like = messages.flatMap(m => m.tool_calls || []).map(t => t.id).filter(Boolean).pop() || "call_" + "x".repeat(24);
  const m = /^(.*[-_:.])?([^-_:.]*)$/.exec(like), tail = m[2] || "x".repeat(24);
  const pool = /^[0-9a-f]+$/.test(tail) ? "0123456789abcdef" : "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  return (m[1] || "") + Array.from(tail, () => pool[Math.random() * pool.length | 0]).join("");
}

// the args a call starts with: each parameter of the tool at its default, or an empty value of its type
const EMPTY = { string: "", integer: 0, number: 0, boolean: false, array: [], object: {} };
function argsFor(name) {
  const t = tools().find(t => t.function.name === name);
  const props = (t && t.function.parameters.properties) || {};
  return structuredClone(Object.fromEntries(Object.entries(props).map(([k, p]) => [k, "default" in p ? p.default : EMPTY[p.type] ?? null])));
}

// --- turns ----------------------------------------------------------------------------------

function area(value, oninput, cls) {
  const ta = el("textarea", cls);
  ta.value = value;
  ta.rows = 1;
  ta.oninput = () => { fit(ta); oninput(ta.value, ta); };
  return ta;
}

function callNode(m, t) {
  const box = el("div", "tc"), line = el("div", "line");
  const sel = el("select");
  for (const f of tools()) sel.appendChild(el("option", null, f.function.name)).value = f.function.name;
  if (!tools().some(f => f.function.name === t.function)) sel.appendChild(el("option", null, t.function + " (not a " + suite + " tool)")).value = t.function;
  sel.value = t.function;
  sel.onchange = () => { t.function = sel.value; t.args = argsFor(sel.value); render(); };
  line.appendChild(sel);
  line.appendChild(el("span", "id", t.id));
  if (!answered(t.id)) line.appendChild(el("span", "warn", "no tool response yet"));
  line.appendChild(btn("✕", () => {
    m.tool_calls = m.tool_calls.filter(x => x !== t);
    if (!m.tool_calls.length) { m.tool_calls = null; m.content ||= text(""); }
    sync(); render();
  }));
  box.appendChild(line);
  box.appendChild(area(JSON.stringify(t.args, null, 2), (v, ta) => {
    try {
      const a = JSON.parse(v);
      if (!a || typeof a !== "object" || Array.isArray(a)) throw 0;
      t.args = a;
      ta.classList.remove("bad");
    } catch (e) { ta.classList.add("bad"); }   // keep the last valid args
  }, "mono"));
  return box;
}

function messageNode(m, i) {
  const c = el("div", "rec " + ({ user: "user", tool: "tool" }[m.role] || ""));
  const head = el("div", "head");
  head.appendChild(el("span", "role", m.role === "tool" ? "tool → agent · " + m.tool_call.function + " · " + m.tool_call_id : m.role));
  if (info.has(m)) head.appendChild(el("span", "info", info.get(m)));
  const below = btn("+ below", () => { after = after === m ? null : m; render(); }, "sm" + (after === m ? " on" : ""));
  below.title = "add new turns right below this one (click again to add at the end)";
  head.appendChild(below);
  const s = btn(busy === i ? "sending…" : "Send ▶", () => send(i));
  s.disabled = busy !== null;
  s.title = "send the conversation up to here; the reply replaces every turn below";
  head.appendChild(s);
  head.appendChild(btn("✕", () => { messages.splice(i, 1); sync(); render(); }));
  c.appendChild(head);
  // an assistant turn that only calls tools has no content (null), like agentdojo's
  c.appendChild(area(textOf(m), v => { m.content = m.role === "assistant" && !v && m.tool_calls ? null : text(v); }));
  if (m.role === "assistant") {
    for (const t of m.tool_calls || []) c.appendChild(callNode(m, t));
    c.appendChild(btn("+ tool call", () => {
      const name = tools()[0].function.name;
      (m.tool_calls ||= []).push({ function: name, args: argsFor(name), id: newId(), placeholder_args: null });
      render();
    }));
  }
  return c;
}

// insert turn m at index `at` (default: below the insertion point); later turns go below it
function add(m, at = slot() + 1) {
  messages.splice(at, 0, m);
  if (after) after = m;
  sync();
  render();
  const ta = $("view").querySelectorAll(".rec > textarea")[messages.indexOf(m)];
  if (ta) ta.focus();
}

function addBar() {
  const i = slot(), inBlock = messages[i + 1] && messages[i + 1].role === "tool";
  const bar = el("div", "addbar" + (after ? " mid" : ""));
  for (const role of ["system", "user", "assistant"]) {
    const b = btn("+ " + role, () => add(role === "assistant" ? { role, content: text(""), tool_calls: null } : { role, content: text("") }), "tb");
    b.disabled = inBlock;
    if (inBlock) b.title = "would split an assistant turn from its tool responses";
    bar.appendChild(b);
  }
  const p = pendingAt(i)[0];
  const tr = btn(running && p ? "running " + p.function + "…" : "+ tool response" + (p ? " to " + p.function : ""), () => respond(p), "tb");
  tr.disabled = !p || running;
  tr.title = "answers the next call of this assistant turn that has no response yet, with what agentdojo's runtime returns";
  bar.appendChild(tr);
  if (after) bar.appendChild(btn("✕ add at the end instead", () => { after = null; render(); }));
  return bar;
}

function render() {
  const view = $("view");
  view.innerHTML = "";
  $("title").textContent = !meta ? "loading agentdojo…" : title || "Pick a case or start a new chat →";
  $("tools").textContent = meta && suite ? suite + " · " + tools().length + " tools" : "";
  if (!meta || !suite) return;
  if (!messages.includes(after)) after = null;   // its turn was deleted or replaced
  messages.forEach((m, i) => {
    view.appendChild(messageNode(m, i));
    if (i === busy) {
      const c = el("div", "rec");
      const below = messages.length - 1 - i;
      c.appendChild(el("div", "placeholder", "asking " + $("model").value + "…" + (below ? " (the reply replaces the " + below + " turns below)" : "")));
      view.appendChild(c);
    }
    if (m === after) view.appendChild(addBar());
  });
  if (error) {
    const c = el("div", "rec err");
    c.appendChild(el("div", "role danger", "request failed"));
    c.appendChild(el("div", "body", error));
    view.appendChild(c);
  }
  if (!after) view.appendChild(addBar());
  view.querySelectorAll("textarea").forEach(fit);
}

async function send(i) {
  if (busy !== null) return;
  const g = gen, model = $("model").value, t0 = Date.now();
  busy = i; error = null;
  render();
  try {
    const r = await fetch("/api/send", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, suite, messages: messages.slice(0, i + 1) }) });
    const res = await r.json();
    if (!r.ok) throw new Error(res.error);
    if (g === gen) {
      info.set(res.message, model + " · " + ((Date.now() - t0) / 1000).toFixed(1) + " s");
      messages.splice(i + 1, Infinity, res.message);
      sync();
    }
  } catch (e) {
    if (g === gen) error = String(e.message || e);
  }
  busy = null;
  render();
}

// add the tool turn answering call p, filled with what agentdojo's runtime returns for it
async function respond(p) {
  const g = gen;
  let content = "";
  running = true; error = null;
  render();
  try {
    const r = await fetch("/api/tool", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ suite, env, messages, call: p.id }) });
    const res = await r.json();
    if (!r.ok) throw new Error(res.error);
    content = res.text;
  } catch (e) {
    if (g === gen) error = "running the tool failed, fill in its response by hand: " + (e.message || e);
  }
  running = false;
  const at = g === gen && !answered(p.id) ? blockEnd(p) : -1;   // after the other answers to its turn's calls
  if (at >= 0) add({ role: "tool", content: text(content), tool_call_id: p.id, tool_call: p, error: null }, at);
  else render();
}

// --- loading --------------------------------------------------------------------------------

function setChat(t, s, ms, e = {}) {
  gen++; title = t; suite = s; messages = ms; env = e; after = null; error = null; busy = null; running = false;
  sync(); render();
}

function newChat() {
  current = null;
  drawList();
  try { history.replaceState(null, "", location.pathname); } catch (e) {}
  const s = $("suite").value;
  setChat("new chat · " + s, s, [{ role: "system", content: text(meta.system) }, { role: "user", content: text("") }]);
}

function drawList() {
  const words = $("q").value.toLowerCase().split(/\s+/).filter(Boolean);
  const list = cases.filter(c => words.every(w => (caseLabel(c) + " " + c.key).toLowerCase().includes(w)));
  $("count").textContent = list.length + " cases";
  renderSidebar($("cases"), list.map(c => ({ key: c.key, name: caseLabel(c), meta: caseMeta(c) })), current, loadCase);
}

async function selectRun(name, keep) {
  run = name;
  $("run").value = name;
  cases = [];
  drawList();
  $("count").textContent = "loading…";
  const r = await fetch("/api/cases?run=" + encodeURIComponent(name));
  if (run !== name) return;
  cases = r.ok ? await r.json() : [];
  drawList();
  if (keep && cases.some(c => c.key === keep)) loadCase(keep);
}

async function loadCase(key) {
  current = key;
  try { history.replaceState(null, "", "#" + encodeURIComponent(run) + "/" + encodeURIComponent(key)); } catch (e) {}
  drawList();
  const r = await fetch("/api/chat?run=" + encodeURIComponent(run) + "&key=" + encodeURIComponent(key));
  const d = await r.json();
  if (current !== key) return;
  if (r.ok) setChat(run + " · " + caseLabel(d), d.suite, d.messages, d.env);
  else { error = d.error; render(); }
}

$("run").onchange = e => selectRun(e.target.value);
$("q").oninput = drawList;
$("new").onclick = newChat;
$("model").onchange = () => { try { localStorage.setItem("model", $("model").value); } catch (e) {} };

(async () => {
  render();
  runs = await (await fetch("/api/runs")).json();
  for (const r of runs) $("run").appendChild(el("option", null, r.name + "  (" + r.attack + ")")).value = r.name;
  const [hRun, ...rest] = decodeURIComponent(location.hash.slice(1)).split("/");
  const first = runs.find(r => r.name === hRun) || runs[0];
  if (first) selectRun(first.name, rest.join("/"));
  meta = await (await fetch("/api/meta")).json();   // waits for agentdojo to import
  for (const s of Object.keys(meta.tools)) $("suite").appendChild(el("option", null, s)).value = s;
  for (const m of meta.models) $("model").appendChild(el("option", null, m)).value = m;
  try { $("model").value = localStorage.getItem("model") || meta.models[0]; } catch (e) {}
  if (!$("model").value) $("model").value = meta.models[0];
  $("new").disabled = false;
  render();
})();
"""

PAGE = webui.page("Chat editor", CSS, BODY, injviewer.CASE_JS + JS)


@functools.cache
def _suites():
    from agentdojo.task_suite.load_suites import get_suites  # slow to import on Lustre
    return get_suites("v1.2.2")


@functools.cache
def _meta():
    from agentdojo.agent_pipeline.agent_pipeline import load_system_message
    from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai
    return {"models": MODELS, "system": load_system_message(None),
            "tools": {name: [_function_to_openai(f) for f in s.tools] for name, s in _suites().items()}}


def _chat_payload(run_dir: Path, key: str):
    p = run_dir / (key + ".json")
    d = json.loads(p.read_text(encoding="utf-8"))
    for m in d["messages"]:  # agentdojo sends a failed call's error instead of its content: make that the content
        if m["role"] == "tool" and m.get("error"):
            m["content"], m["error"] = [{"type": "text", "content": m["error"]}], None
    return {**injviewer._summary(p, run_dir), "messages": d["messages"],
            "env": {"user_task": d["user_task_id"], "injections": d.get("injections") or {}}}


def _run_tool(suite: str, env: dict, messages: list[dict], call_id: str) -> str:
    """What agentdojo's runtime returns to the model for the call `call_id`: the call is run in the suite's
    environment with env's injections, set up for env's user task, after every earlier call in `messages`."""
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
    from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
    from agentdojo.types import get_text_content_as_str
    s = _suites()[suite]
    environment = s.load_and_inject_default_environment(env.get("injections") or {})
    task = s.user_tasks.get(env.get("user_task"))
    if task is not None:  # an injection task run alone gets no set-up, as in run_task_with_pipeline
        environment = task.init_environment(environment)
    runtime, executor = FunctionsRuntime(s.tools), ToolsExecutor()  # YAML output, as the openai-compatible pipeline
    for t in (t for m in messages if m["role"] == "assistant" for t in m.get("tool_calls") or []):
        turn = {"role": "assistant", "content": None, "tool_calls": [FunctionCall(**t)]}
        *_, out, _ = executor.query("", runtime, environment, [turn])
        if t["id"] == call_id:
            return out[-1]["error"] or get_text_content_as_str(out[-1]["content"])
    raise KeyError(f"no tool call {call_id!r} in the conversation")


@functools.cache
def _client():
    import openai
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key and "openrouter.ai" in os.getenv("OPENAI_COMPATIBLE_BASE_URL", ""):
        key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
    if not key:
        raise RuntimeError("no OpenRouter key: set OPENROUTER_API_KEY in .env")
    return openai.OpenAI(base_url=OPENROUTER, api_key=key)


def _send(model: str, suite: str, messages: list[dict]) -> dict:
    """The assistant turn `model` adds to `messages` (agentdojo ChatMessages), made as OpenAILLM.query makes it."""
    from run_agentdojo import use_system_role
    use_system_role()  # the system prompt goes out as role `system`, as in run_agentdojo.py
    from agentdojo.agent_pipeline.llms.openai_llm import (
        _function_to_openai, _message_to_openai, _openai_to_assistant_message, chat_completion_request)
    from agentdojo.functions_runtime import FunctionCall
    for m in messages:
        if m.get("tool_calls"):
            m["tool_calls"] = [FunctionCall(**t) for t in m["tool_calls"]]
        if m.get("tool_call"):
            m["tool_call"] = FunctionCall(**m["tool_call"])
    tools = [_function_to_openai(f) for f in _suites()[suite].tools]
    completion = chat_completion_request(_client(), model, [_message_to_openai(m, model) for m in messages], tools,
                                         reasoning_effort=None, temperature=0.0)
    if not completion.choices:  # OpenRouter reports some provider errors in a 200 response
        raise RuntimeError(getattr(completion, "error", None) or "no choices in the response")
    out = _openai_to_assistant_message(completion.choices[0].message)
    return {**out, "tool_calls": [t.model_dump() for t in out["tool_calls"]] if out["tool_calls"] else None}


class Handler(injviewer.Handler):  # /api/runs and /api/cases as in injviewer
    def do_GET(self):
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        if parsed.path == "/":
            return self._html(PAGE)
        if parsed.path == "/api/meta":
            return self._json(_meta())
        if parsed.path == "/api/chat":
            run_dir, key = injviewer._run_dir(q.get("run", [""])[0]), q.get("key", [""])[0]
            if run_dir is None or not injviewer._KEY.fullmatch(key) or not (run_dir / (key + ".json")).is_file():
                return self._json({"error": "not found"}, code=404)
            try:
                return self._json(_chat_payload(run_dir, key))
            except (KeyError, ValueError, TypeError) as e:
                return self._json({"error": f"bad result file: {type(e).__name__}: {e}"}, code=400)
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/send", "/api/tool"):
            return self._not_found()
        try:
            b = self._body()
            suite, messages = b["suite"], b["messages"]
            if suite not in _suites():
                raise ValueError(f"unknown suite {suite!r}")
            if path == "/api/send" and b["model"] not in MODELS:
                raise ValueError(f"unknown model {b['model']!r}")
        except (ValueError, KeyError, TypeError) as e:
            return self._json({"error": f"bad request: {e}"}, code=400)
        try:
            if path == "/api/send":
                self._json({"message": _send(b["model"], suite, messages)})
            else:
                self._json({"text": _run_tool(suite, b.get("env") or {}, messages, b["call"])})
        except Exception as e:  # network / API / key errors, a malformed turn
            self._json({"error": f"{type(e).__name__}: {e}"}, code=500)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat editor for AgentDojo conversations, continued through OpenRouter.")
    ap.add_argument("--dir", default="runs", help="directory of run dirs (default: runs)")
    ap.add_argument("--port", type=int, default=8769)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    injviewer.RUNS_DIR = Path(args.dir)
    threading.Thread(target=_meta, daemon=True).start()  # warm up the agentdojo import
    webui.serve(Handler, args.host, args.port, f"{injviewer.RUNS_DIR}/")


if __name__ == "__main__":
    main()

"""Playground for AgentDojo: run one case from a web page and watch it run, instead of a batch run_agentdojo.py run.

    python -m ui.playground                 # http://localhost:8770, results in runs/playground/
    python -m ui.playground --dir runs/playground --port 8770

Pick a model (on OpenRouter, with the key chateditor.py uses), an attack (registered by my_attacks,
chatinject_attacks or agentdojo itself; "none" runs the user task benign), a suite, a user task and an
injection task, then Run. The case runs in a thread of this server the way run_agentdojo.py runs a case:
agentdojo set up the same way, agentdojo's pipeline under the same name at temperature 0, and the same
result JSON. It is drawn as injviewer.py draws a case, turn by turn while it runs. Several requests can
run at once.

Each request gets its own run dir, runs/playground/<time>_<model>_<attack>/, laid out like a --logdir of
run_agentdojo.py, plus a request.json that the sidebar's history is read from. Picking an earlier request
shows its result and puts its settings in the form. `python -m ui.injviewer --dir runs/playground` lists
the same requests as runs.

Before a run, its attack's module is imported again if its file changed since it was imported (all of
ATTACK_MODULES are checked when the page loads), so edits to my_attacks.py apply without a server restart.
"""

import argparse
import functools
import importlib
import itertools
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import chateditor, injviewer, webui

PLAYGROUND = Path("runs/playground")  # overridden in main()
# the two models vllm/serve_multinode.sh serves, here on OpenRouter; the first one is the default
MODELS = ["qwen/qwen3-235b-a22b-2507", "z-ai/glm-4.5"]
ATTACK_MODULES = ("my_attacks", "chatinject_attacks")  # listed in this order, before agentdojo's own attacks
UNUSABLE = {"manual"}  # reads its injections from stdin

_running: set[str] = set()  # ids of the requests whose thread has not finished
_imported: dict[str, float] = {}  # attack module -> mtime of the file it was last imported from
_import_lock = threading.Lock()


CSS = r"""
.form { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 6px 10px; align-items: center;
  margin: 0 4px 10px; font-size: 12px; color: var(--muted); }
.form select { width: 100%; font: inherit; font-size: 13px; color: var(--fg); padding: 4px 6px; border-radius: 7px;
  border: 1px solid var(--border); background: var(--bg); }
.form select:disabled { opacity: .5; }
.form button { grid-column: 1 / -1; padding: 7px 10px; font-weight: 600; }
.hint, .warn { margin: 0 4px 14px; font-size: 12px; white-space: pre-wrap; word-break: break-word; }
.hint { color: var(--muted); max-height: 12em; overflow-y: auto; }
.warn { color: var(--danger); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.hint:empty, .warn:empty { display: none; }
.item .meta { white-space: pre-line; }
"""

BODY = r"""
<div class="layout">
  <aside>
    <h1>New request</h1>
    <div class="form">
      <label for="model">model</label><select id="model"></select>
      <label for="attack">attack</label><select id="attack"></select>
      <label for="suite">suite</label><select id="suite"></select>
      <label for="ut">user task</label><select id="ut"></select>
      <label for="it">injection</label><select id="it"></select>
      <button class="tb on" id="go" disabled>loading agentdojo…</button>
    </div>
    <div class="warn" id="err"></div>
    <div class="hint" id="hint"></div>
    <h1>History</h1>
    <div class="filters">
      <input id="q" placeholder="filter, e.g. glm one_turn ut_3">
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
let meta = null, requests = [], current = null, shown = null;   // shown: the /api/case reply drawn last
const $ = id => document.getElementById(id);
const caseTitle = d => d.model + " · " + (d.kind === "attack" ? d.attack : "no attack") + " · " + caseLabel(d);
const oneLine = (s, n) => { s = String(s).replace(/\s+/g, " "); return s.length > n ? s.slice(0, n) + "…" : s; };
function option(parent, value, label, title) {
  const o = parent.appendChild(el("option", null, label));
  o.value = value;
  if (title) o.title = title;
}

// --- the form -------------------------------------------------------------------------------

// list the suite's tasks, keeping the chosen ones if the suite has them
function fillTasks() {
  const s = meta.suites[$("suite").value];
  for (const [id, tasks] of [["ut", s.user_tasks], ["it", s.injection_tasks]]) {
    const sel = $(id), keep = sel.value;
    sel.innerHTML = "";
    for (const [t, text] of Object.entries(tasks)) option(sel, t, short(t) + " · " + oneLine(text, 48), text);
    if (keep in tasks) sel.value = keep;
  }
  hint();
}

// the chosen user task's prompt and injection task's goal
function hint() {
  const s = meta.suites[$("suite").value], benign = $("attack").value === "none";
  $("it").disabled = benign;
  $("hint").textContent = "user task: " + s.user_tasks[$("ut").value]
    + (benign ? "" : "\n\ninjection goal: " + s.injection_tasks[$("it").value]);
}

// put request h's settings in the form, those it still offers
function setForm(h) {
  const has = (id, v) => [...$(id).options].some(o => o.value === v);
  for (const [id, v] of [["model", h.model], ["attack", h.attack], ["suite", h.suite]]) if (has(id, v)) $(id).value = v;
  fillTasks();
  if (has("ut", h.user_task)) $("ut").value = h.user_task;
  if (h.injection_task && has("it", h.injection_task)) $("it").value = h.injection_task;
  hint();
}

async function run() {
  const attack = $("attack").value;
  const body = { model: $("model").value, attack, suite: $("suite").value, user_task: $("ut").value,
                 injection_task: attack === "none" ? null : $("it").value };
  $("go").disabled = true;
  $("err").textContent = "";
  try {
    const r = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error);
    await refresh();
    select(d.id);
  } catch (e) {
    $("err").textContent = String(e.message || e);
  }
  $("go").disabled = false;
}

// --- history --------------------------------------------------------------------------------

const stateOf = h => h.running ? "⏳ running…" : h.error ? "⚠ " + oneLine(h.error, 70) : caseMeta(h);

function drawList() {
  const words = $("q").value.toLowerCase().split(/\s+/).filter(Boolean);
  const list = requests.filter(h => words.every(w => [caseLabel(h), h.model, h.attack, h.id].join(" ").toLowerCase().includes(w)));
  $("count").textContent = list.length + " of " + requests.length + " requests";
  renderSidebar($("cases"), list.map(h => ({ key: h.id, name: caseLabel(h),
    meta: h.created.slice(5, 16) + " · " + h.model.split("/").pop() + " · " + h.attack + "\n" + stateOf(h) })), current, select);
}

async function refresh() {
  const r = await fetch("/api/history");
  if (r.ok) requests = await r.json();
  drawList();
}

function select(id) {
  current = id;
  shown = null;
  try { history.replaceState(null, "", "#" + encodeURIComponent(id)); } catch (e) {}
  drawList();
  const h = requests.find(h => h.id === id);
  if (h && meta) setForm(h);
  return loadCase(id);
}

async function loadCase(id) {
  const r = await fetch("/api/case?id=" + encodeURIComponent(id)), text = await r.text();
  if (current !== id || text === shown) return;
  if (!r.ok && data && data.id === id) return;   // e.g. its result file caught mid-write: keep the one drawn
  shown = text;
  const d = JSON.parse(text), main = document.querySelector("main");
  const atEnd = main.scrollTop + main.clientHeight >= main.scrollHeight - 80;
  data = r.ok ? d : { key: id, error: d.error };
  render();
  if (data.running && atEnd) main.scrollTop = main.scrollHeight;   // follow a running case
}

// while requests run, keep the list and the case drawn up to date
async function poll() {
  try {
    if (requests.some(h => h.running)) {
      await refresh();
      if (data && data.running) await loadCase(current);
    }
  } catch (e) {}
  setTimeout(poll, 1500);
}

$("q").oninput = drawList;
$("suite").onchange = fillTasks;
$("attack").onchange = $("ut").onchange = $("it").onchange = hint;
$("go").onclick = run;

(async () => {
  await refresh();
  const id = decodeURIComponent(location.hash.slice(1));
  const first = requests.find(h => h.id === id) || requests[0];
  if (first) select(first.id);
  poll();
  const r = await fetch("/api/meta");   // waits for agentdojo and the attack modules to import
  const m = await r.json();
  if (!r.ok) { $("err").textContent = m.error; $("go").textContent = "agentdojo did not load"; return; }
  meta = m;
  for (const name of meta.models) option($("model"), name, name);
  option($("attack"), "none", "none (benign run, no injection)");
  const groups = {};
  for (const a of meta.attacks) {
    groups[a.module] ||= $("attack").appendChild(Object.assign(el("optgroup"), { label: a.module }));
    option(groups[a.module], a.name, a.name, a.doc);
  }
  if (meta.attacks.length) $("attack").value = meta.attacks[0].name;
  for (const s of Object.keys(meta.suites)) option($("suite"), s, s);
  fillTasks();
  $("err").textContent = meta.errors.join("\n\n");
  const h = requests.find(h => h.id === current);
  if (h) setForm(h);
  $("go").textContent = "Run ▶";
  $("go").disabled = false;
})();
"""

PAGE = webui.page("AgentDojo playground", injviewer.CSS + CSS, BODY, injviewer.CASE_JS + injviewer.VIEW_JS + JS)


# --- attacks and tasks -----------------------------------------------------------------------

def _import_attacks(modules) -> list[str]:
    """Import each of `modules` not imported yet or whose file changed since it was; the errors."""
    chateditor._suites()  # agentdojo set up before the attack modules import it
    errors = []
    with _import_lock:
        for name in modules:
            module = sys.modules.get(name)
            try:
                if module is None:
                    module = importlib.import_module(name)
                elif os.stat(module.__file__).st_mtime != _imported.get(name):
                    module = importlib.reload(module)
                else:
                    continue
                _imported[name] = os.stat(module.__file__).st_mtime
            except Exception as e:  # a syntax error, a typo…: the attacks it registered before stay
                errors.append(f"importing {name} failed:\n" + "".join(traceback.format_exception_only(e)).rstrip())
    return errors


def _attacks() -> list[dict]:
    """The registered attacks that can run here, ATTACK_MODULES' first: [{name, module, doc}]."""
    from agentdojo.attacks.attack_registry import ATTACKS
    out = [{"name": name, "module": cls.__module__ if cls.__module__ in ATTACK_MODULES else "agentdojo",
            "doc": (cls.__doc__ or "").strip().split("\n")[0]}
           for name, cls in ATTACKS.items() if name not in UNUSABLE]
    order = (*ATTACK_MODULES, "agentdojo")
    return sorted(out, key=lambda a: order.index(a["module"]))  # stable: registration order within a module


@functools.cache
def _tasks() -> dict:
    """{suite: {user_tasks: {id: prompt}, injection_tasks: {id: goal}}}"""
    return {name: {"user_tasks": {i: t.PROMPT for i, t in s.user_tasks.items()},
                   "injection_tasks": {i: t.GOAL for i, t in s.injection_tasks.items()}}
            for name, s in chateditor._suites().items()}


def _meta() -> dict:
    errors = _import_attacks(ATTACK_MODULES)
    return {"models": MODELS, "attacks": _attacks(), "errors": errors, "suites": _tasks()}


# --- running a request -----------------------------------------------------------------------

def _write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _start(b: dict) -> str:
    """Check a run request, give it a run dir and start its thread; its id. ValueError if the request is bad."""
    from agentdojo.attacks.attack_registry import ATTACKS
    suite, attack = chateditor._suites().get(b["suite"]), b["attack"]
    if b["model"] not in MODELS:
        raise ValueError(f"unknown model {b['model']!r}")
    if suite is None:
        raise ValueError(f"unknown suite {b['suite']!r}")
    if b["user_task"] not in suite.user_tasks:
        raise ValueError(f"no user task {b['user_task']!r} in {suite.name}")
    attack_cls = None
    if attack != "none":
        if attack not in ATTACKS or attack in UNUSABLE:
            raise ValueError(f"unknown attack {attack!r}")
        if b.get("injection_task") not in suite.injection_tasks:
            raise ValueError(f"no injection task {b.get('injection_task')!r} in {suite.name}")
        if ATTACKS[attack].__module__ in ATTACK_MODULES and (errors := _import_attacks([ATTACKS[attack].__module__])):
            raise ValueError(errors[0])
        attack_cls = ATTACKS[attack]  # as just imported again
    chateditor._client()  # without an OpenRouter key, fail here rather than in the thread

    now = time.localtime()
    req = {"model": b["model"], "attack": attack, "suite": suite.name, "user_task": b["user_task"],
           "injection_task": b["injection_task"] if attack_cls else None,
           "created": time.strftime("%Y-%m-%d %H:%M:%S", now)}
    stem = time.strftime("%Y%m%d-%H%M%S_", now) + req["model"].split("/")[-1] + "_" + attack
    for n in itertools.count(1):
        d = PLAYGROUND / (stem if n == 1 else f"{stem}-{n}")
        try:
            d.mkdir(parents=True)
            break
        except FileExistsError:
            pass
    _write(d / "request.json", req)
    _running.add(d.name)
    threading.Thread(target=_run, args=(d, req, attack_cls), daemon=True).start()
    return d.name


def _run(d: Path, req: dict, attack_cls) -> None:
    """Thread body: run the request's case as run_agentdojo.py runs one, logged to the run dir d."""
    from agentdojo import benchmark
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    from agentdojo.logging import LOGGER_STACK, OutputLogger
    from run_agentdojo import BENCHMARK_VERSION, PIPELINE_NAME
    LOGGER_STACK.set([])  # the stack's default list is shared by all threads: this one gets its own
    try:
        suite = chateditor._suites()[req["suite"]]
        llm = OpenAILLM(chateditor._client(), req["model"])
        pipeline = AgentPipeline.from_config(PipelineConfig(
            llm=llm, model_id=None, defense=None, system_message_name=None, system_message=None))
        pipeline.name = PIPELINE_NAME  # the result's dir, and the model's name to attacks that address it
        user_task = suite.get_user_task_by_id(req["user_task"])
        with OutputLogger(str(d)):  # the case's TraceLogger writes its result JSON under this logdir, at every step
            if attack_cls is None:
                benchmark.run_task_without_injection_tasks(suite, pipeline, user_task, d, True, BENCHMARK_VERSION)
            else:
                benchmark.run_task_with_injection_tasks(suite, pipeline, user_task, attack_cls(suite, pipeline), d,
                                                        True, [req["injection_task"]], BENCHMARK_VERSION)
    except Exception as e:  # API errors, an attack that fails on this case…
        traceback.print_exc()
        _write(d / "request.json", {**req, "error": f"{type(e).__name__}: {e}"})
    finally:
        _running.discard(d.name)


# --- reading requests ------------------------------------------------------------------------

def _entry(d: Path) -> dict:
    """History entry of the request in run dir d, with the fields injviewer names a case by."""
    req = json.loads((d / "request.json").read_text(encoding="utf-8"))
    paths = injviewer._case_paths(d)  # its result JSON, once the case has started
    case = injviewer._summary(paths[0], d) if paths else {}
    return {"id": d.name, "key": case.get("key"), "kind": "attack" if req["injection_task"] else "benign",
            **{k: req[k] for k in ("model", "attack", "suite", "user_task", "injection_task", "created")},
            "error": req.get("error"), "running": d.name in _running,
            "utility": case.get("utility"), "security": case.get("security")}


def _history() -> list[dict]:
    out = []
    for d in sorted((p for p in PLAYGROUND.iterdir() if p.is_dir()), reverse=True):  # newest first
        try:
            out.append(_entry(d))
        except (OSError, ValueError, KeyError):  # not a request's run dir
            pass
    return out


def _case(d: Path) -> dict:
    """The request's case as injviewer draws one, with its model and whether it is running."""
    e = _entry(d)
    if e["key"]:
        case = injviewer._case_payload(d, e["key"])
    else:  # no result JSON yet: the attack is still making its injections, or failed to
        it = e["injection_task"]
        case = {"key": None, "kind": e["kind"], "suite": e["suite"], "user_task": e["user_task"], "injection_task": it,
                "attack": e["attack"] if it else None, "goal": injviewer._goal(e["suite"], it) if it else None,
                "utility": None, "security": None, "duration": None, "run_error": None, "messages": [], "unseen": []}
    return {**case, "id": e["id"], "model": e["model"], "running": e["running"],
            "run_error": case["run_error"] or e["error"]}


class Handler(webui.JSONHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._html(PAGE)
        if parsed.path == "/api/meta":
            try:
                return self._json(_meta())
            except Exception as e:  # agentdojo failed to import
                return self._json({"error": f"{type(e).__name__}: {e}"}, code=500)
        if parsed.path == "/api/history":
            return self._json(_history())
        if parsed.path == "/api/case":
            d = injviewer._run_dir(parse_qs(parsed.query).get("id", [""])[0])
            if d is None or not (d / "request.json").is_file():
                return self._json({"error": "no such request"}, code=404)
            try:
                return self._json(_case(d))
            except (KeyError, ValueError, TypeError) as e:  # also a result JSON caught while it is written
                return self._json({"error": f"bad result file: {type(e).__name__}: {e}"}, code=503)
        self._not_found()

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self._not_found()
        try:
            self._json({"id": _start(self._body())})
        except (ValueError, KeyError, TypeError) as e:
            self._json({"error": f"bad request: {e}"}, code=400)
        except Exception as e:  # no OpenRouter key, …
            self._json({"error": f"{type(e).__name__}: {e}"}, code=500)


def main() -> None:
    global PLAYGROUND
    ap = argparse.ArgumentParser(description="Run one AgentDojo case at a time from a web page, and look back at earlier ones.")
    ap.add_argument("--dir", default="runs/playground", help="where each request gets its run dir (default: runs/playground)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    PLAYGROUND = injviewer.RUNS_DIR = Path(args.dir)  # injviewer._run_dir finds a request's run dir by its id there
    PLAYGROUND.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_meta, daemon=True).start()  # warm up the agentdojo and attack imports
    webui.serve(Handler, args.host, args.port, f"{PLAYGROUND}/")


if __name__ == "__main__":
    main()

"""Web viewer for vllm/token_proxy.py logs: the exact tokens each chat request became.

    python -m ui.tokenviewer                  # serve ./vllm/token_logs at http://localhost:8768
    python -m ui.tokenviewer --dir vllm/token_logs --port 8768

Pick a log and a request in the sidebar. The prompt is split into turns at the chat template's
turn-header tokens (<|im_start|>, ]~b], <|start|>, <|header_start|>, GLM's <|user|>, ...), and every
added token is marked: special tokens red, other added tokens amber. An injected "<|im_end|>" that
became a real special token stands out from plain text, and forged turns show up as turns of their
own. A request whose prompt holds more turn headers than its messages can render (n_messages + 2:
the template's own system turn and the generation prompt) is flagged ⚑ in the sidebar.

"Tokens" splits plain text at token boundaries (hover a token for its id). The filter box keeps the
requests whose prompt or completion contains the text. A log that token_proxy.py is still writing
is followed live. Each entry is decoded with the tokenizer of the model dir it logged.
"""

import argparse
import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tokenizers import Tokenizer

from . import webui

LOG_DIR = Path("vllm/token_logs")  # overridden in main()

# added tokens that open a turn; the role is the text that follows them
TURN_TOKENS = {"<|im_start|>", "]~b]", "<|start|>", "<|header_start|>", "<start_of_turn>",
               "<|im_system|>", "<|im_user|>", "<|im_assistant|>"}
# added tokens that open a turn and name its role themselves (GLM)
ROLE_TOKENS = {"<|system|>": "system", "<|user|>": "user", "<|assistant|>": "assistant",
               "<|observation|>": "observation"}


@dataclass
class Tok:
    tokenizer: Tokenizer
    added: dict[int, tuple[str, bool]]  # id -> (content, special)
    turn_ids: set[int]


@lru_cache(maxsize=8)
def load_tokenizer(model_dir: str) -> Tok | None:
    path = Path(model_dir) / "tokenizer.json"
    if not path.is_file():
        return None
    tokenizer = Tokenizer.from_file(str(path))
    added = {i: (t.content, t.special) for i, t in tokenizer.get_added_tokens_decoder().items()}
    turn_ids = {i for i, (content, _) in added.items() if content in TURN_TOKENS or content in ROLE_TOKENS}
    return Tok(tokenizer, added, turn_ids)


class LogIndex:
    """Byte offsets and summaries of a log's entries, extended as token_proxy.py appends to it."""

    def __init__(self, path: Path):
        self.path = path
        self.offsets: list[int] = []
        self.summaries: list[dict] = []
        self.end = 0  # bytes indexed so far; a partly written last line is left for later
        self.lock = threading.Lock()

    def update(self) -> None:
        with self.lock:
            size = self.path.stat().st_size
            if size < self.end:  # truncated or replaced
                self.offsets, self.summaries, self.end = [], [], 0
            if size == self.end:
                return
            with self.path.open("rb") as f:
                f.seek(self.end)
                while True:
                    start = f.tell()
                    line = f.readline()
                    if not line.endswith(b"\n"):
                        break
                    self.end = f.tell()
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    self.offsets.append(start)
                    self.summaries.append(summarize(len(self.summaries), entry))

    def read(self, i: int) -> dict:
        with self.path.open("rb") as f:
            f.seek(self.offsets[i])
            return json.loads(f.readline())

    def search(self, query: str) -> list[int]:
        """Entries whose line holds `query` as the log encodes it, i.e. in the prompt or completion text."""
        needle = json.dumps(query, ensure_ascii=False)[1:-1].encode()
        hits = []
        with self.path.open("rb") as f:
            for i, offset in enumerate(self.offsets):
                f.seek(offset)
                if needle in f.readline():
                    hits.append(i)
        return hits


_indexes: dict[Path, LogIndex] = {}
_indexes_lock = threading.Lock()


def log_index(name: str) -> LogIndex | None:
    path = webui.safe_file(LOG_DIR, name, ".jsonl")
    if path is None:
        return None
    with _indexes_lock:
        index = _indexes.setdefault(path, LogIndex(path))
    index.update()
    return index


def summarize(i: int, entry: dict) -> dict:
    tok = load_tokenizer(entry.get("tokenizer") or "")
    ids = entry["prompt_token_ids"]
    turns = sum(1 for t in ids if t in tok.turn_ids) if tok else None
    n_messages = entry.get("n_messages") or 0
    return {
        "i": i,
        "time": entry.get("time", ""),
        "n_messages": n_messages,
        "prompt_tokens": len(ids),
        "completion_tokens": sum(len(c) for c in entry.get("completion_token_ids", [])),
        "finish": ",".join(str(f) for f in entry.get("finish_reason", [])),
        "turns": turns,
        "extra": turns - (n_messages + 2) if turns is not None else 0,
    }


def segments(ids: list[int], tok: Tok, with_pieces: bool) -> list[dict]:
    """Plain-text runs {"t", "n"[, "p"]} and added tokens {"a": id, "t": content, "s": special}.

    A run is decoded whole, so multi-byte characters split across tokens survive; with_pieces adds
    its token pieces [text, ids], a token that ends mid-character joining the next one."""
    out, run = [], []

    def flush() -> None:
        if run:
            seg = {"t": tok.tokenizer.decode(run, skip_special_tokens=False), "n": len(run)}
            if with_pieces:
                seg["p"] = pieces(run, tok.tokenizer)
            out.append(seg)
            run.clear()

    for t in ids:
        if t in tok.added:
            flush()
            content, special = tok.added[t]
            out.append({"a": t, "t": content, "s": special})
        else:
            run.append(t)
    flush()
    return out


def pieces(run: list[int], tokenizer: Tokenizer) -> list[list]:
    out, pending = [], []
    for t in run:
        pending.append(t)
        text = tokenizer.decode(pending, skip_special_tokens=False)
        if text.endswith("�") and len(pending) < 4:
            continue
        out.append([text, pending])
        pending = []
    if pending:
        out.append([tokenizer.decode(pending, skip_special_tokens=False), pending])
    return out


def split_turns(segs: list[dict], tok: Tok) -> list[dict]:
    """The prompt's segments grouped into turns, each opened by a turn-header token."""
    turns = [{"role": "", "segs": []}]  # what precedes the first header, e.g. ]~!b[ or [gMASK]<sop>
    for seg in segs:
        if seg.get("a") in tok.turn_ids:
            turns.append({"role": ROLE_TOKENS.get(seg["t"], ""), "segs": []})
        turns[-1]["segs"].append(seg)
    for turn in turns:
        segs_ = turn["segs"]
        if not turn["role"] and len(segs_) > 1 and segs_[0].get("a") in tok.turn_ids and "a" not in segs_[1]:
            words = segs_[1]["t"].split(None, 1)  # "user\n...", or "user ..." when an injected newline was lost
            turn["role"] = words[0][:40] if words else ""
        turn["n"] = sum(seg.get("n", 1) for seg in segs_)
        turn["added"] = sum(1 for seg in segs_ if "a" in seg)
    return [turn for turn in turns if turn["segs"]]


def entry_payload(index: LogIndex, i: int, with_pieces: bool) -> dict:
    entry = index.read(i)
    meta = {k: entry.get(k) for k in ("time", "id", "model", "n_messages", "finish_reason", "tokenizer")}
    meta.update(index.summaries[i])
    tok = load_tokenizer(entry.get("tokenizer") or "")
    if tok is None:
        return {**meta, "error": f"no tokenizer.json in {entry.get('tokenizer')}: showing the logged text",
                "prompt_text": entry.get("prompt", ""), "completion_text": entry.get("completion", [])}
    prompt = segments(entry["prompt_token_ids"], tok, with_pieces)
    counts: dict[str, int] = {}
    for seg in prompt:
        if "a" in seg:
            counts[seg["t"]] = counts.get(seg["t"], 0) + 1
    return {
        **meta,
        "turns": split_turns(prompt, tok),
        "completions": [segments(ids, tok, with_pieces) for ids in entry.get("completion_token_ids", [])],
        "added_counts": sorted(counts.items(), key=lambda kv: -kv[1]),
    }


def logs_payload() -> list[dict]:
    if not LOG_DIR.is_dir():
        return []
    paths = sorted(LOG_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "size": p.stat().st_size, "mtime": webui.mtime_str(p)} for p in paths]


CSS = r"""
:root { --spbg: #fee2e2; --spfg: #991b1b; --adbg: #fef3c7; --adfg: #92400e; --pc0: #e0e7ff; --pc1: #f1f5f9; --sys: #f5f3ff; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --spbg: #4c1d1d; --spfg: #fecaca; --adbg: #422006; --adfg: #fde68a;
    --pc0: #1e2a44; --pc1: #161b26; --sys: #1d1a2c; }
}
.content { max-width: 1100px; }
aside select, aside input { display: block; width: 100%; margin: 0 0 8px; padding: 6px 8px; border: 1px solid var(--border);
  border-radius: 7px; background: var(--bg); color: var(--fg); font: inherit; }
.count { font-size: 12px; color: var(--muted); margin: 0 4px 6px; }
.legend { font-size: 12px; color: var(--muted); display: flex; gap: 6px; align-items: center; }
.rec.sys { background: var(--sys); }
.rec.req { background: var(--panel); border-style: dashed; }
.thead { cursor: pointer; user-select: none; }
.rec.collapsed .toks, .rec.collapsed .rawbar { display: none; }
.toks { white-space: pre-wrap; word-break: break-word; font: 12.5px/1.65 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.sp, .ad { border-radius: 3px; padding: 0 2px; font-weight: 700; }
.sp { background: var(--spbg); color: var(--spfg); }
.ad { background: var(--adbg); color: var(--adfg); }
.pc { border-radius: 2px; }
.pc.p0 { background: var(--pc0); }
.pc.p1 { background: var(--pc1); }
.kv { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 3px 14px; font-size: 13px; }
.kv .k { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
.kv .v { word-break: break-word; min-width: 0; }
.chips { display: flex; flex-wrap: wrap; gap: 4px; margin: 10px 0 4px; }
.chip { border: 1px solid var(--border); border-radius: 999px; padding: 1px 9px; font: 12px/1.6 inherit; cursor: pointer;
  background: var(--bg); color: var(--fg); }
.chip.user { background: var(--user); } .chip.tool { background: var(--tool); } .chip.sys { background: var(--sys); }
.flagged { color: var(--danger); font-weight: 700; }
"""

BODY = r"""
<div class="layout">
  <aside>
    <h1>Token logs</h1>
    <select id="logSel"></select>
    <input id="q" type="search" placeholder="filter by prompt text" spellcheck="false">
    <div class="count" id="count"></div>
    <div id="entries"></div>
  </aside>
  <main>
    <div class="toolbar">
      <span class="title" id="title">Pick a request →</span>
      <span class="legend"><span class="sp">special</span><span class="ad">added</span></span>
      <button class="tb" id="bTokens" title="split plain text at token boundaries">Tokens</button>
      <button class="tb" id="bCollapse">Collapse all</button>
      <button class="tb" id="bExpand">Expand all</button>
      <button class="tb" id="bCopy">Copy prompt</button>
    </div>
    <div id="view" class="content"><div class="empty">No request selected.</div></div>
  </main>
</div>
"""

JS = r"""
let log = null, entries = [], cur = null, entry = null, query = "", lastSize = null;
let tokens = false, collapsed = new Set();
const enc = encodeURIComponent;

async function api(path) {
  const r = await fetch(path);
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || r.status);
  return d;
}

const roleClass = r => /^(user|human)\b/i.test(r) ? "user"
  : /^(tool|observation|ipython|functions)/i.test(r) ? "tool"
  : /^(system|developer)\b/i.test(r) ? "sys" : "";
const segText = segs => segs.map(s => s.t).join("");

async function loadLogs() {
  const logs = await api("/api/logs");
  const sel = document.getElementById("logSel");
  const keep = sel.value;
  sel.innerHTML = "";
  for (const l of logs) {
    const o = el("option", null, l.name.replace(/\.jsonl$/, "") + " · " + (l.size / 1048576).toFixed(1) + " MB");
    o.value = l.name; sel.appendChild(o);
  }
  if (!logs.length) {
    document.getElementById("entries").innerHTML = "";
    document.getElementById("count").textContent = "no logs: run vllm/token_proxy.py serve";
    return logs;
  }
  if (!log || !logs.some(l => l.name === log)) log = logs[0].name;
  sel.value = log || keep;
  return logs;
}

async function loadEntries() {
  if (!log) return;
  const d = await api("/api/entries?log=" + enc(log) + "&q=" + enc(query));
  lastSize = d.size; entries = d.entries;
  const flagged = entries.filter(e => e.extra > 0).length;
  document.getElementById("count").textContent = entries.length + (query ? " matching" : "") + " requests"
    + (flagged ? " · " + flagged + " ⚑" : "");
  const box = document.getElementById("entries");
  box.innerHTML = "";
  for (const e of entries) {
    const b = el("button", "item" + (e.i === cur ? " active" : ""));
    const name = b.appendChild(el("div", "name", "#" + e.i + " · " + e.time.slice(11)));
    if (e.extra > 0) {
      const f = name.appendChild(el("span", "flagged", "  ⚑ +" + e.extra + " turns"));
      f.title = "more turn headers than " + e.n_messages + " messages can render: likely injected";
    }
    b.appendChild(el("div", "meta", e.n_messages + " msgs · " + (e.turns ?? "?") + " turns · "
      + e.prompt_tokens + " → " + e.completion_tokens + " tok · " + e.finish));
    b.onclick = () => pick(e.i);
    box.appendChild(b);
  }
}

async function pick(i) {
  cur = i;
  history.replaceState(null, "", "#" + enc(log) + "/" + i);
  for (const b of document.querySelectorAll("#entries .item")) b.classList.toggle("active", b.querySelector(".name").textContent.startsWith("#" + i + " "));
  await loadEntry();
}

async function loadEntry() {
  if (cur == null) return;
  entry = await api("/api/entry?log=" + enc(log) + "&i=" + cur + "&tokens=" + (tokens ? 1 : 0));
  render();
}

function renderSegs(box, segs) {
  const frag = document.createDocumentFragment();
  let flip = 0;
  for (const s of segs) {
    if (s.a != null) {
      const sp = el("span", s.s ? "sp" : "ad", s.t);
      sp.title = "id " + s.a + (s.s ? " · special" : " · added, not special");
      frag.appendChild(sp);
    } else if (tokens && s.p) {
      for (const [text, ids] of s.p) {
        const sp = el("span", "pc p" + (flip ^= 1), text.replace(/\n/g, "↵\n"));
        sp.title = "id " + ids.join(" ");
        frag.appendChild(sp);
      }
    } else frag.appendChild(document.createTextNode(s.t));
  }
  box.appendChild(frag);
}

function card(key, cls, label, segs) {
  const c = el("div", "rec " + cls + (collapsed.has(key) ? " collapsed" : ""));
  const head = c.appendChild(el("div", "role thead", (collapsed.has(key) ? "▸ " : "▾ ") + label));
  head.onclick = () => {
    if (collapsed.has(key)) collapsed.delete(key); else collapsed.add(key);
    c.classList.toggle("collapsed");
    head.textContent = (collapsed.has(key) ? "▸ " : "▾ ") + label;
  };
  const body = c.appendChild(el("div", "toks"));
  renderSegs(body, segs);
  const bar = c.appendChild(el("div", "rawbar"));
  bar.appendChild(copyButton(() => segText(segs)));
  return c;
}

function render() {
  const e = entry, view = document.getElementById("view");
  view.innerHTML = "";
  document.getElementById("title").textContent = log.replace(/\.jsonl$/, "") + " · #" + cur;
  document.getElementById("bTokens").className = "tb" + (tokens ? " on" : "");

  const meta = view.appendChild(el("div", "rec req"));
  meta.appendChild(el("div", "role", "request #" + cur));
  const kv = meta.appendChild(el("div", "kv"));
  const row = (k, v, cls) => { kv.appendChild(el("div", "k", k)); kv.appendChild(el("div", "v" + (cls ? " " + cls : ""), v)); };
  row("time", e.time); row("id", e.id); row("model", e.model);
  row("messages", e.n_messages + " sent · " + (e.turns ?? "?") + " turn headers in the prompt");
  if (e.extra > 0) row("⚑", e.extra + " more turn headers than the messages can render: likely injected turns", "flagged");
  row("tokens", e.prompt_tokens + " prompt · " + e.completion_tokens + " completion · finish " + e.finish);
  if (e.added_counts) row("added tokens", e.added_counts.map(([t, n]) => t + " ×" + n).join("   "));
  row("tokenizer", e.tokenizer);
  if (e.error) {
    row("error", e.error, "flagged");
    view.appendChild(card("p", "", "prompt (logged text)", [{ t: e.prompt_text }]));
    (e.completion_text || []).forEach((t, k) => view.appendChild(card("c" + k, "", "completion", [{ t }])));
    return;
  }
  const chips = view.appendChild(el("div", "chips"));
  e.turns.forEach((t, k) => {
    const ch = chips.appendChild(el("button", "chip " + roleClass(t.role), k + " " + (t.role || "prefix")));
    ch.onclick = () => document.getElementById("turn" + k).scrollIntoView({ behavior: "smooth", block: "start" });
  });
  e.turns.forEach((t, k) => {
    const c = card("t" + k, roleClass(t.role), "turn " + k + " · " + (t.role || "before the first turn header")
      + " · " + t.n + " tokens" + (t.added ? " · " + t.added + " added" : ""), t.segs);
    c.id = "turn" + k;
    view.appendChild(c);
  });
  e.completions.forEach((segs, k) => {
    const n = segs.reduce((a, s) => a + (s.n || 1), 0);
    view.appendChild(card("c" + k, "result", "completion" + (e.completions.length > 1 ? " " + k : "") + " · "
      + n + " tokens · finish " + (e.finish_reason || [])[k], segs));
  });
}

document.getElementById("logSel").onchange = ev => { log = ev.target.value; cur = null; entry = null;
  document.getElementById("view").innerHTML = '<div class="empty">No request selected.</div>'; loadEntries(); };
let qTimer = null;
document.getElementById("q").oninput = ev => { clearTimeout(qTimer); qTimer = setTimeout(() => { query = ev.target.value; loadEntries(); }, 300); };
document.getElementById("bTokens").onclick = () => { tokens = !tokens; loadEntry(); };
document.getElementById("bCollapse").onclick = () => { if (!entry || !entry.turns) return;
  entry.turns.forEach((_, k) => collapsed.add("t" + k)); (entry.completions || []).forEach((_, k) => collapsed.add("c" + k)); render(); };
document.getElementById("bExpand").onclick = () => { collapsed.clear(); if (entry) render(); };
document.getElementById("bCopy").onclick = async ev => { if (!entry || !entry.turns) return;
  await copyText(entry.turns.map(t => segText(t.segs)).join("")); ev.target.textContent = "Copied ✓";
  setTimeout(() => { ev.target.textContent = "Copy prompt"; }, 1200); };
document.addEventListener("keydown", ev => {   // ↑ / ↓ step through the listed requests
  if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT" || !entries.length) return;
  if (ev.key !== "ArrowDown" && ev.key !== "ArrowUp") return;
  ev.preventDefault();
  const k = entries.findIndex(e => e.i === cur);
  const next = entries[Math.min(Math.max(k + (ev.key === "ArrowDown" ? 1 : -1), 0), entries.length - 1)];
  if (next) pick(next.i);
});

(async () => {
  const [hashLog, hashI] = decodeURIComponent(location.hash.slice(1)).split("/");
  if (hashLog) log = hashLog;
  await loadLogs();
  await loadEntries();
  if (hashLog === log && hashI !== undefined && hashI !== "") pick(Number(hashI));
  setInterval(async () => {   // follow logs that token_proxy.py is still writing
    const logs = await loadLogs().catch(() => []);
    const l = logs.find(x => x.name === log);
    if (l && l.size !== lastSize) loadEntries();
  }, 3000);
})();
"""

PAGE = webui.page("token log viewer", CSS, BODY, JS)


class Handler(webui.JSONHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        if parsed.path == "/":
            return self._html(PAGE)
        if parsed.path == "/api/logs":
            return self._json(logs_payload())
        if parsed.path not in ("/api/entries", "/api/entry"):
            return self._not_found()
        index = log_index(params.get("log", ""))
        if index is None:
            return self._json({"error": "log not found"}, code=404)
        if parsed.path == "/api/entries":
            query = params.get("q", "")
            keep = set(index.search(query)) if query else None
            return self._json({"size": index.end, "entries": [
                s for s in index.summaries if keep is None or s["i"] in keep]})
        try:
            i = int(params.get("i", ""))
            index.summaries[i]
        except (ValueError, IndexError):
            return self._json({"error": "no such entry"}, code=404)
        self._json(entry_payload(index, i, params.get("tokens") == "1"))


def main() -> None:
    global LOG_DIR
    ap = argparse.ArgumentParser(description="Web viewer for vllm/token_proxy.py token logs.")
    ap.add_argument("--dir", default="vllm/token_logs", help="log directory (default: vllm/token_logs)")
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    LOG_DIR = Path(args.dir)
    webui.serve(Handler, args.host, args.port, f"{LOG_DIR}/")


if __name__ == "__main__":
    main()

"""Session logging for miniagent.

Two outputs per session, written side by side in the log dir:
  - <ts>.jsonl : append-only structured record (source of truth for analysis)
  - <ts>.md    : human-readable transcript, written live as the session runs

Record types: user, model, tool_call, tool_result, and request: the exact JSON body of the
session's last generate_content call as sent over the wire, written once at the end (each
call resends the whole conversation, so the last one holds everything). The .md shows only
its part counts.

The same rendering is used for the live transcript and for re-rendering an old
log from the CLI, so both always agree:

    python -m harness.runlog runs/miniagent/<ts>.jsonl          # print transcript to stdout
    python -m harness.runlog runs/miniagent/<ts>.jsonl --md      # also write runs/miniagent/<ts>.md
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


# --------------------------------------------------------------------------- #
# formatting helpers                                                          #
# --------------------------------------------------------------------------- #

def _ts_short(iso_ts: str) -> str:
    """'2026-09-23T12:29:14.722140' -> '12:29:14'."""
    try:
        return iso_ts.split("T", 1)[1][:8]
    except (IndexError, AttributeError):
        return "??:??:??"


def compact_args(args: dict, max_val: int = 80) -> str:
    """Render tool args as 'key=val, ...', collapsing whitespace and clipping
    long values (email bodies etc.)."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
        s = " ".join(s.split())  # collapse newlines / runs of spaces
        if len(s) > max_val:
            s = s[:max_val] + "…"
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def truncate_block(text: str, max_lines: int = 8, max_chars: int = 500) -> str:
    """Indent a (possibly huge) tool result for the transcript, keeping only the
    first few lines / chars. Full text always lives in the .jsonl."""
    text = (text or "").rstrip()
    total = len(text)
    lines = text.split("\n")
    truncated = len(lines) > max_lines
    clipped = "\n".join(lines[:max_lines])
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars]
        truncated = True
    if truncated:
        clipped = clipped.rstrip() + f"\n… (truncated, {total} chars — full text in .jsonl)"
    return "\n".join("    " + ln for ln in clipped.split("\n"))


PART_META = {"thought", "thoughtSignature", "videoMetadata", "partMetadata"}


def part_kind(part: dict) -> str:
    """A wire-format part's kind: 'text', 'functionCall', 'functionResponse', ... ('thought'
    for a thought-summary text part)."""
    if part.get("thought"):
        return "thought"
    return next((k for k in part if k not in PART_META), "?")


def request_summary(body: str):
    """Content and part counts for a logged request body, or None if it isn't JSON."""
    try:
        req = json.loads(body)
    except (TypeError, ValueError):
        return None
    contents = req.get("contents") or []
    kinds = Counter(part_kind(p) for c in contents for p in c.get("parts") or [])
    return {"contents": len(contents), "parts": sum(kinds.values()), "kinds": dict(kinds)}


def _render_body(rec: dict):
    """Render one record's body (no turn header). Returns None to skip it."""
    t = rec.get("type")

    if t == "user":
        return f"**user:** {rec.get('text', '')}"

    if t == "model":
        calls = rec.get("function_calls")
        text = rec.get("text")
        if calls:
            lines = []
            if text and text.strip():  # intermediate narration before a tool call
                lines.append(f"**model:** {text.strip()}")
            for c in calls:
                lines.append(f"**model → tool:** {c.get('name')}({compact_args(c.get('args') or {})})")
            return "\n\n".join(lines)
        if text is None:  # blocked / max-rounds / no candidate
            reason = rec.get("blocked") or rec.get("error") or "no response"
            return f"⚠️ **model stopped:** {reason}"
        return f"**model:** {text}"

    if t == "tool_result":
        name = rec.get("name", "")
        if rec.get("declined"):
            return f"🚫 **result:** {name} — user declined"
        result = rec.get("result", "")
        stripped = result.lstrip()
        marker = "❌ " if stripped.startswith("ERROR") or stripped.startswith('{"error"') else ""
        return f"{marker}**result:** {name}\n{truncate_block(result)}"

    if t == "request":
        head = f"*→ final request · {len(rec.get('body') or '') / 1024:.1f} KB"
        s = request_summary(rec.get("body"))
        if s is None:
            return head + " · body is not JSON*"
        kinds = ", ".join(f"{k} {v}" for k, v in s["kinds"].items())
        return head + f" · {s['contents']} contents · {s['parts']} parts ({kinds})*"

    # tool_call is redundant with the model's '→ tool' line; skip it.
    return None


def format_record(rec: dict, last_turn):
    """Render a record, emitting a turn header when the turn changes.
    Returns (text, new_last_turn); text is '' for skipped records."""
    body = _render_body(rec)
    if body is None:
        return "", last_turn
    out = []
    turn = rec.get("turn")
    if turn != last_turn:
        out.append(f"## Turn {turn} · {_ts_short(rec.get('ts', ''))}")
        last_turn = turn
    out.append(body)
    return "\n\n".join(out) + "\n\n", last_turn


def render_transcript(records) -> str:
    """Render a whole list of records to a transcript string."""
    out = []
    last_turn = None
    for rec in records:
        text, last_turn = format_record(rec, last_turn)
        if text:
            out.append(text)
    return "".join(out)


# --------------------------------------------------------------------------- #
# live session log                                                            #
# --------------------------------------------------------------------------- #

class RunLog:
    """Append-only JSONL record + live Markdown transcript for one session."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now():%Y%m%d-%H%M%S}"
        self.path = log_dir / f"{stem}.jsonl"
        self.md_path = log_dir / f"{stem}.md"
        self._f = self.path.open("a", encoding="utf-8")
        self._md = self.md_path.open("a", encoding="utf-8")
        self._md.write(f"# Session {stem}\n\n")
        self._md.flush()
        self.turn = 0
        self._last_turn = None

    def write(self, type_: str, **fields):
        record = {"ts": datetime.now().isoformat(), "turn": self.turn, "type": type_, **fields}
        self._f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

        text, self._last_turn = format_record(record, self._last_turn)
        if text:
            self._md.write(text)
            self._md.flush()

    def close(self):
        self._f.close()
        self._md.close()


# --------------------------------------------------------------------------- #
# CLI: re-render an existing .jsonl                                           #
# --------------------------------------------------------------------------- #

def _read_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a session .jsonl log as a readable transcript.")
    ap.add_argument("jsonl", help="path to a runs/miniagent/<ts>.jsonl file")
    ap.add_argument("--md", action="store_true", help="also write <same-stem>.md next to the source")
    args = ap.parse_args()

    stem = Path(args.jsonl).stem
    transcript = f"# Session {stem}\n\n" + render_transcript(_read_jsonl(args.jsonl))

    if args.md:
        out_path = Path(args.jsonl).with_suffix(".md")
        out_path.write_text(transcript, encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(transcript)


if __name__ == "__main__":
    main()

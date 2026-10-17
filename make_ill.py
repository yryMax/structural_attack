"""Generate a target, ill request and payload from a case containing a task.

The case's functionResponse is closed at [INJECTION POINT], followed by all remaining
benign execution turns and the task user turn. The original response is then reopened
with its prefix repeated before the untouched suffix. The remaining execution turns
are included in the payload rather than kept as outer turns in ill.json.
JSON stored inside result is closed before the surrounding response turn.
Top-level task is input metadata only.

The objective is character edit distance after compact JSON serialization and removal
of ASCII spaces, including spaces inside strings. Text surrounding the marker stays
fixed. If the selected string contains JSON, the raw payload is encoded inside its
marked string value, preserving valid JSON through every nested string layer.
payload.txt contains the raw replacement for the innermost string (e.g. mail body).
Templates without markers retain whole-field replacement.

    python make_ill.py
    python make_ill.py --case case.json
    python make_ill.py --template template.json --target target.json
    python make_ill.py --case case.json --path '["contents", 2, "parts", 0, "functionResponse", "response", "result"]'

This optimizes serialized text locally; it does not evaluate model behavior.
"""

import argparse
import copy
import heapq
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from injector import DupDict, loads


INJECTION_POINT = "[INJECTION POINT]"


def compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize(text: str) -> str:
    return text.replace(" ", "")


def find_injection_path(template: dict) -> list:
    """Locate a unique marker in any string value, otherwise use the legacy field.

    Supports the OpenAI chat format (messages[i].content[j].text) and falls back to
    the Gemini format (the last functionResponse result). Pass --path for any other
    string field, including an earlier message or a non-"text" key.
    """
    paths = []

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + [key])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [index])
        elif isinstance(value, str) and INJECTION_POINT in value:
            paths.append(path)

    visit(template, [])
    if len(paths) > 1:
        raise ValueError("multiple fields contain [INJECTION POINT]; select one with --path")
    if paths:
        return paths[0]

    messages = template.get("messages")
    if isinstance(messages, list):
        for i in range(len(messages) - 1, -1, -1):
            content = messages[i].get("content")
            if isinstance(content, list):
                for j in range(len(content) - 1, -1, -1):
                    part = content[j]
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        return ["messages", i, "content", j, "text"]
    contents = template.get("contents", [])
    for i in range(len(contents) - 1, -1, -1):
        parts = contents[i].get("parts", [])
        for j in range(len(parts) - 1, -1, -1):
            if "functionResponse" in parts[j]:
                return ["contents", i, "parts", j, "functionResponse", "response", "result"]
    raise ValueError("no injectable text field found; specify a string field with --path")


def _get(obj, path):
    if not isinstance(path, list) or not path:
        raise ValueError("path must be a nonempty JSON array of keys and indices")
    value = obj
    for key in path:
        if type(key) not in (str, int):
            raise ValueError("path entries must be string keys or integer indices")
        try:
            value = value[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"invalid injection path: {path}") from exc
    return value


def _set(obj, path, value):
    parent = obj
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value


def close_result(value: str) -> str:
    """Close every JSON text layer at the marker without changing its prefix.

    Inside a JSON string, omit the rest of that string's contents but retain its
    closing quote and the original structural suffix, including later fields.
    Apply the same rule recursively if that decoded string contains JSON itself.
    Ordinary text is simply cut at the marker.
    """
    if value.count(INJECTION_POINT) != 1:
        raise ValueError("closing requires exactly one [INJECTION POINT]")
    before, after = value.split(INJECTION_POINT)
    try:
        loads(before)
    except json.JSONDecodeError:
        pass
    else:
        return before
    try:
        loads(value)
    except json.JSONDecodeError:
        # A marker between JSON tokens can make the source unparsable. Removing
        # it restores the original structural suffix without guessing delimiters.
        try:
            loads(before + after)
        except json.JSONDecodeError:
            return before
        return before + after

    # A literal marker in valid JSON must be inside a quoted token. Track quote
    # and escape state instead of counting braces that may occur in string data.
    in_string, escaped, string_start = False, False, None
    for index, char in enumerate(before):
        if escaped:
            escaped = False
        elif in_string and char == "\\":
            escaped = True
        elif char == '"':
            in_string = not in_string
            if in_string:
                string_start = index
    for index, char in enumerate(after):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            token_end = len(before) + len(INJECTION_POINT) + index + 1
            decoded = loads(value[string_start:token_end])
            decoded_before = decoded.split(INJECTION_POINT)[0]
            inner_closed = close_result(decoded)
            completion = inner_closed[len(decoded_before) :]
            closed = before + compact(completion)[1:-1] + after[index:]
            loads(closed)  # Verify the actual result text, including all JSON layers.
            return closed
    raise ValueError("could not close the JSON string containing [INJECTION POINT]")


def build_target(template: dict, task: str, path: list | None = None) -> dict:
    """Close the response, replay the benign continuation, then task and reopen."""
    if not isinstance(task, str) or not task.strip():
        raise ValueError("case task must be a nonempty string")
    if not isinstance(template, dict) or not isinstance(template.get("contents"), list):
        raise ValueError("case contents must be an array of turns")
    for turn in template["contents"]:
        if not isinstance(turn, dict) or not isinstance(turn.get("parts"), list):
            raise ValueError("each case turn must be an object with a parts array")
        if any(not isinstance(part, dict) for part in turn["parts"]):
            raise ValueError("each case part must be an object")
    path = find_injection_path(template) if path is None else path
    value = _get(template, path)
    if (
        len(path) != 7
        or path[0] != "contents"
        or type(path[1]) is not int
        or path[2] != "parts"
        or type(path[3]) is not int
        or path[4:] != ["functionResponse", "response", "result"]
    ):
        raise ValueError("automatic target generation requires a functionResponse.response.result path")
    if not isinstance(value, str) or value.count(INJECTION_POINT) != 1:
        raise ValueError("automatic target generation requires exactly one [INJECTION POINT] in result")

    before, after = value.split(INJECTION_POINT)
    target = copy.deepcopy(template)
    contents = target["contents"]
    # _get already validated indices; normalize negative indices before slicing.
    turn_index = path[1] % len(contents)
    turn = contents[turn_index]
    part_index = path[3] % len(turn["parts"])
    left, resumed = copy.deepcopy(turn), copy.deepcopy(turn)
    left["parts"] = left["parts"][: part_index + 1]
    response_path = ["functionResponse", "response", "result"]
    _set(left["parts"][-1], response_path, close_result(value))
    # Reopen the entire original turn, repeating its prefix so the unchanged
    # suffix still completes the same result, including any nested JSON text.
    _set(resumed["parts"][part_index], response_path, before + after)
    user_turn = {"parts": [{"text": task}], "role": "user"}
    # A benign completion may span model tool calls, tool returns and a final
    # answer. Keep the whole continuation in order before the forged task.
    contents[turn_index:] = [left, *contents[turn_index + 1 :], user_turn, resumed]
    return target


def _trim(prefix: str, target: str, suffix: str) -> tuple[str, str, str, int]:
    """Cancel fixed common edges; this preserves Levenshtein distance exactly."""
    start = 0
    while start < min(len(prefix), len(target)) and prefix[start] == target[start]:
        start += 1
    prefix, target = prefix[start:], target[start:]
    end = 0
    while end < min(len(suffix), len(target)) and suffix[-end - 1] == target[-end - 1]:
        end += 1
    if end:
        suffix, target = suffix[:-end], target[:-end]
    return prefix, target, suffix, start


def edit_distance(left: str, right: str) -> int:
    """Levenshtein distance, with unit insertions, deletions and substitutions."""
    left, right, _, _ = _trim(left, right, "")
    _, right, left, _ = _trim("", right, left)
    if len(left) > len(right):
        left, right = right, left
    row = list(range(len(left) + 1))
    for i, char in enumerate(right, 1):
        next_row = [i]
        for j, other in enumerate(left, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (char != other)))
        row = next_row
    return row[-1]


def _encoded_contents(value: str, layers: int) -> str:
    for _ in range(layers):
        value = compact(value)[1:-1]
    return value


def _escape_tokens(layers: int) -> set[str]:
    if type(layers) is not int or layers < 1:
        raise ValueError("escape_layers must be a positive integer")
    return {_encoded_contents(chr(code), layers) for code in range(32)} | {
        _encoded_contents('"', layers), _encoded_contents("\\", layers)
    }


def _json_string_layers(value: str) -> int:
    """Count JSON documents around the marked decoded string value."""
    layers = 0
    while True:
        try:
            decoded = loads(value)
        except json.JSONDecodeError:
            return layers
        marked_values = []

        def visit(child):
            if isinstance(child, str):
                if INJECTION_POINT in child:
                    marked_values.append(child)
            elif isinstance(child, dict):
                for _, item in child.items():
                    visit(item)
            elif isinstance(child, list):
                for item in child:
                    visit(item)

        visit(decoded)
        if len(marked_values) != 1 or marked_values[0].count(INJECTION_POINT) != 1:
            raise ValueError("JSON injection marker must occur in exactly one string value")
        value = marked_values[0]
        layers += 1


def _automaton(prefix: str, suffix: str, escape_layers: int = 1):
    """Recognize prefix + canonical string contents encoded N times + suffix.

    Edges are (destination, character, inside_field). None matches any unescaped
    string character; an empty character is an epsilon transition.
    """
    edges = [[]]

    def new_state():
        edges.append([])
        return len(edges) - 1

    state = 0
    for char in prefix:
        next_state = new_state()
        edges[state].append((next_state, char, False))
        state = next_state
    field = state
    edges[field].append((field, None, True))
    # json.dumps emits these finite escape tokens; all other characters are raw.
    tokens = _escape_tokens(escape_layers)
    trie = {}
    for token in sorted(tokens):
        state = field
        for i, char in enumerate(token):
            if i == len(token) - 1:
                edges[state].append((field, char, True))
            else:
                key = (state, char)
                if key not in trie:
                    trie[key] = new_state()
                    edges[state].append((trie[key], char, True))
                state = trie[key]
    state = new_state()
    edges[field].append((state, "", False))
    for char in suffix:
        next_state = new_state()
        edges[state].append((next_state, char, False))
        state = next_state
    return edges, state


def _plain(char: str) -> bool:
    return ord(char) >= 32 and char not in ('"', "\\", " ")


def _restore_spaces(chars, gaps, escape_layers: int = 1) -> str:
    """Restore target spaces before complete escape tokens, never within an escape."""
    encoded = "".join(char for char, _ in chars)
    tokens = _escape_tokens(escape_layers)
    pieces = []
    i = 0
    while i < len(chars):
        if encoded[i] == "\\":
            token = next(token for token in tokens if encoded.startswith(token, i))
            length = len(token)
        else:
            length = 1
        origin = next((origin for _, origin in chars[i : i + length] if origin is not None), None)
        if origin is not None:
            pieces.append(" " * gaps[origin])
        pieces.append(encoded[i : i + length])
        i += length
    value = "".join(pieces)
    for _ in range(escape_layers):
        value = json.loads('"' + value + '"')
    return value


def _closest_value(
    prefix: str, target: str, suffix: str, gaps: list[int], escape_layers: int = 1
) -> tuple[str, int]:
    """Shortest edit path through the JSON-string automaton: a global optimum.

    At equal edit distance, prefer preserving target characters over substituting or
    deleting them. This favors escaped quotes over replacing them with arbitrary text.
    """
    edges, accept = _automaton(prefix, suffix, escape_layers)
    weight = len(target) + 1  # primary: edit distance; secondary: lost target characters
    infinity = (len(prefix) + len(suffix) + len(target) + 10) * weight
    costs = [infinity] * len(edges)
    costs[0] = 0
    parents = [[None] * len(edges)]
    for i in range(len(target) + 1):
        # Inserting output characters does not consume input. Dijkstra closes these
        # transitions, including the cycles that encode arbitrary escaped strings.
        heap = [(cost, state) for state, cost in enumerate(costs) if cost < infinity]
        heapq.heapify(heap)
        while heap:
            cost, state = heapq.heappop(heap)
            if cost != costs[state]:
                continue
            for dest, char, field in edges[state]:
                # A raw-character insertion into the self-loop cannot improve a path.
                if char is None:
                    continue
                next_cost = cost + (weight if char else 0)
                if next_cost < costs[dest]:
                    costs[dest] = next_cost
                    parents[i][dest] = (i, state, char, field, None)
                    heapq.heappush(heap, (next_cost, dest))
        if i == len(target):
            break
        next_costs = [infinity] * len(edges)
        next_parents = [None] * len(edges)
        for state, cost in enumerate(costs):
            if cost >= infinity:
                continue
            if cost + weight + 1 < next_costs[state]:
                next_costs[state] = cost + weight + 1
                next_parents[state] = (i, state, "", False, None)
            for dest, char, field in edges[state]:
                if char == "":
                    continue
                emitted = (target[i] if _plain(target[i]) else "?") if char is None else char
                match = emitted == target[i]
                next_cost = cost + (0 if match else weight + 1)
                if next_cost < next_costs[dest]:
                    next_costs[dest] = next_cost
                    next_parents[dest] = (i, state, emitted, field, i if match else None)
        costs = next_costs
        parents.append(next_parents)
    distance = costs[accept] // weight
    chars = []
    i, state = len(target), accept
    while (i, state) != (0, 0):
        prev_i, prev_state, char, field, origin = parents[i][state]
        if field and char:
            chars.append((char, origin))
        i, state = prev_i, prev_state
    chars.reverse()
    return _restore_spaces(chars, gaps, escape_layers), distance


@dataclass
class Injection:
    path: list
    payload: str
    body: dict
    distance: int
    baseline_distance: int
    escape_layers: int = 1


def build_injection(template: dict, target: dict, path: list | None = None) -> Injection:
    path = find_injection_path(template) if path is None else path
    value = _get(template, path)
    if not isinstance(value, str):
        raise TypeError(f"injection path must reference a string: {path}")
    if value.count(INJECTION_POINT) > 1:
        raise ValueError(f"injection field must contain at most one [INJECTION POINT]: {path}")
    if INJECTION_POINT in value:
        field_before, field_after = value.split(INJECTION_POINT)
        inner_layers = _json_string_layers(value)
    else:
        field_before = field_after = ""
        inner_layers = 0
    escape_layers = inner_layers + 1
    marked = copy.deepcopy(template)
    marker = uuid.uuid4().hex
    _set(marked, path, field_before + marker + field_after)
    before, after = compact(marked).split(marker)
    prefix, suffix = normalize(before), normalize(after)
    target_raw = compact(target)
    target_text = normalize(target_raw)
    gaps, spaces = [], 0
    for char in target_raw:
        if char == " ":
            spaces += 1
        else:
            gaps.append(spaces)
            spaces = 0
    prefix, middle, suffix, start = _trim(prefix, target_text, suffix)
    payload, distance = _closest_value(
        prefix, middle, suffix, gaps[start : start + len(middle)], escape_layers
    )
    # When the target's corresponding substring is optimal, preserve its spaces,
    # including trailing spaces canceled with the fixed suffix.
    try:
        target_value = _get(target, path)
    except ValueError:
        target_value = None
    if (
        isinstance(target_value, str)
        and target_value.startswith(field_before)
        and target_value.endswith(field_after)
        and len(target_value) >= len(field_before) + len(field_after)
    ):
        end = len(target_value) - len(field_after)
        target_payload = target_value[len(field_before) : end]
        try:
            for _ in range(inner_layers):
                target_payload = json.loads('"' + target_payload + '"')
        except json.JSONDecodeError:
            pass
        else:
            if normalize(_encoded_contents(target_payload, escape_layers)) == normalize(
                _encoded_contents(payload, escape_layers)
            ):
                payload = target_payload
    body = copy.deepcopy(template)
    replacement = field_before + _encoded_contents(payload, inner_layers) + field_after
    if inner_layers:
        loads(replacement)  # Verify the actual nested result, not only the outer request.
    _set(body, path, replacement)
    # Verify the actual normal serialization, without manually reinterpreting payload.
    actual = edit_distance(normalize(compact(body)), target_text)
    baseline = edit_distance(normalize(compact(template)), target_text)
    if actual != distance or distance > baseline:
        raise RuntimeError("optimizer result failed serialized-distance verification")
    return Injection(list(path), payload, body, distance, baseline, escape_layers)


def build_case(case: dict, path: list | None = None) -> tuple[dict, Injection]:
    """Generate target and injection from a request plus top-level task metadata."""
    if not isinstance(case, dict):
        raise ValueError("case must be a JSON object containing a task and contents")
    task = case.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("case task must be a nonempty string")
    # DupDict.pop/del would leave the task in serialized pairs. Rebuild instead,
    # preserving duplicate keys and their original order in the request.
    template = copy.deepcopy(DupDict([(key, value) for key, value in case.items() if key != "task"]))
    target = build_target(template, task, path)
    path = list(find_injection_path(template) if path is None else path)
    # Resolve indices before removing the continuation so negative paths still
    # select the same response in the shortened injection template.
    path[1] %= len(template["contents"])
    path[3] %= len(template["contents"][path[1]]["parts"])
    # This is the request at the moment the selected tool response arrives.
    # Its later benign execution is replayed in the payload, not outside it.
    del template["contents"][path[1] + 1 :]
    return target, build_injection(template, target, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--case", type=Path, help="case with contents and task (default: case.json)")
    source.add_argument("--template", type=Path, help="legacy mode: template for an existing target")
    parser.add_argument(
        "--target", type=Path, default=Path("target.json"),
        help="generated target output, or existing target input with --template",
    )
    parser.add_argument(
        "--path",
        type=json.loads,
        help="JSON array of keys/indices; defaults to the unique [INJECTION POINT] field, "
        "or the last injectable text field for templates without markers",
    )
    parser.add_argument("--output", type=Path, default=Path("ill.json"))
    parser.add_argument("--payload", type=Path, default=Path("payload.txt"))
    args = parser.parse_args()
    try:
        if args.template is not None:
            template = loads(args.template.read_text(encoding="utf-8"))
            target = loads(args.target.read_text(encoding="utf-8"))
            result = build_injection(template, target, args.path)
        else:
            case_path = args.case if args.case is not None else Path("case.json")
            case = loads(case_path.read_text(encoding="utf-8"))
            target, result = build_case(case, args.path)
        if args.template is None:
            args.target.write_text(
                json.dumps(target, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        args.output.write_text(
            json.dumps(result.body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        args.payload.write_text(result.payload, encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"field: {json.dumps(result.path)}")
    print(f"JSON string encoding layers: {result.escape_layers}")
    print(
        f"edit distance after removing spaces: {result.baseline_distance} -> {result.distance} (minimum)"
    )
    outputs = [args.output, args.payload]
    if args.template is None:
        outputs.insert(0, args.target)
    print(f"wrote {', '.join(map(str, outputs))} ({len(result.payload)} payload characters)")


if __name__ == "__main__":
    main()

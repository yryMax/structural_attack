import copy
import json
import uuid
from dataclasses import dataclass
from pathlib import Path


class DupDict(dict):
    """A JSON object that keeps repeated keys. Lookups see the last value (as json.loads does);
    items() — and so json.dumps — gives back every key/value pair in file order."""

    def __init__(self, pairs=()):
        super().__init__(pairs)
        self.pairs = [[k, v] for k, v in pairs]

    def items(self):
        return [(k, v) for k, v in self.pairs]

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        for p in reversed(self.pairs):   # the occurrence lookups see
            if p[0] == key:
                p[1] = value
                return
        self.pairs.append([key, value])

    def __deepcopy__(self, memo):
        return DupDict([(k, copy.deepcopy(v, memo)) for k, v in self.pairs])


def loads(s: str):
    """json.loads that keeps duplicate keys (see DupDict)."""
    return json.loads(s, object_pairs_hook=DupDict)


def _dumps(obj) -> str:
    # default separators (", " / ": "), single line — the payload style inside the ill field
    return json.dumps(obj, ensure_ascii=False)


def _edges(body: list, index: int, field: str) -> tuple[str, str]:
    # put a unique marker at body[index][field] and cut the dump there, so the cut is at
    # exactly that record's field even if the same text appears elsewhere in the body
    marker = uuid.uuid4().hex
    marked = copy.deepcopy(body)
    marked[index][field] = marker
    before, after = _dumps(marked).split(marker)
    return before, after


@dataclass
class ToolCase:
    """A benign tool return plus the attacker's sample question."""
    call: str            # the tool call, e.g. "gmail.users.messages.get(id='19bbcdf14cdf9c6c')"
    body: list           # the tool result: a list of records
    question: str        # what the attacker asks the agent

    def build_prompt(self) -> str:
        return f"{self.call}\nResult: {_dumps(self.body)}\nTask: {self.question}"

    def to_json(self) -> str:
        return json.dumps({"call": self.call, "body": self.body, "question": self.question},
                          ensure_ascii=False, indent=2)

    def ask(self) -> str:
        """Send build_prompt() to Gemini and return the model's answer."""
        import gemini
        return gemini.ask(self.build_prompt())


def inject_from_target(template: ToolCase, target: ToolCase, point: list) -> ToolCase:
    index, field = point
    if not isinstance(template.body[index][field], str):
        raise TypeError(f"point {point} must reference a string field")

    prefix, suffix = _edges(template.body, index, field)
    target_raw = _dumps(target.body)

    # assert that the target raw's prefix and suffix match the template's prefix and suffix
    if not target_raw.startswith(prefix) or not target_raw.endswith(suffix):
        raise ValueError(f"target body does not match template body at point {point}") 

    payload = target_raw[len(prefix): len(target_raw) - len(suffix)]   # goes into the field
    ill_body = copy.deepcopy(template.body)
    ill_body[index][field] = payload
    return ToolCase(template.call, ill_body, template.question)

def resume_from_ill(ill_case: ToolCase, point: list) -> ToolCase:
    index, field = point
    payload = ill_case.body[index][field]
    if not isinstance(payload, str):
        raise TypeError(f"point {point} must reference a string field")

    prefix, suffix = _edges(ill_case.body, index, field)
    try:
        target_body = loads(prefix + payload + suffix)
    except json.JSONDecodeError as e:
        raise ValueError(f"payload at point {point} does not rebuild valid JSON: {e}") from e
    return ToolCase(ill_case.call, target_body, ill_case.question)

if __name__ == "__main__":
    template = loads(Path("cases/email_template.json").read_text(encoding="utf-8"))
    template_case = ToolCase(template["call"], template["body"], template["question"])

    #ill = loads(Path("cases/email_ill.json").read_text(encoding="utf-8"))
    #ill_case = ToolCase(ill["call"], ill["body"], ill["question"])

    target = loads(Path("cases/email_target.json").read_text(encoding="utf-8"))
    target_case = ToolCase(target["call"], target["body"], target["question"])

    ill_case = inject_from_target(template_case, target_case, point=[0, "body"])

    with open("cases/email_ill.json", "w", encoding="utf-8") as f:
        f.write(ill_case.to_json())

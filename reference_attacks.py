"""Replay the injections logged by an earlier attack run, to check whether its ASR reproduces.

Run with ``-ml reference_attacks --attack from_reference`` and point ``REFERENCE_RUN_DIR`` at the
earlier run's logdir, e.g. ``runs/qwen_qwen3-235b-a22b-2507_reference``. Each case reads
``<suite>/<user_task>/<attack>/<injection_task>.json`` from that directory, either directly or
under its pipeline directory (``openai-compatible/``). A case missing from the reference run fails
instead of running uninjected.

``REFERENCE_FORMAT`` says how the logged ``injections`` reached the environment:

- ``logged`` (default): agentdojo formatted them into ``environment.yaml`` as they are, as in every
  run made with ``run_agentdojo.py``. They are returned unchanged.
- ``appended``: the run was made with ChatInject's agentdojo 0.1.34 fork
  (``~/ChatInject/src/agentdojo``), e.g. ``runs/qwen_qwen3-235b-a22b-2507_reference``. Its
  ``load_and_inject_default_environment`` appends each raw payload, minus control characters other
  than ``\\n\\r\\t``, to the vector's default value on a new line. Rebuilding that value reproduces
  every tool output of the reference's 389 traces except wall-clock timestamps, provided tool
  results are dumped with agentdojo's upstream ``allow_unicode=False``.
"""

import json
import os
from pathlib import Path

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.attack_registry import register_attack
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.task_suite.task_suite import TaskSuite

FORMATS = ("logged", "appended")


def _append_to_default(default: str, payload: str) -> str:
    """The default, a line break unless it is empty or already ends with one, then the payload.

    Defaults are raw text of a double-quoted YAML scalar, where a newline is the two characters
    ``\\n``, so the raw payload is escaped as such a scalar body too.
    """
    separator = "" if not default or default.endswith("\\n") else "\n"
    payload = "".join(char for char in payload if ord(char) >= 32 or char in "\n\r\t")
    return default + json.dumps(separator + payload, ensure_ascii=False)[1:-1]


@register_attack
class FromReferenceAttack(BaseAttack):
    name = "from_reference"

    def __init__(
        self,
        task_suite: TaskSuite,
        target_pipeline: BasePipelineElement,
        reference_dir: Path | None = None,
        reference_format: str | None = None,
    ) -> None:
        super().__init__(task_suite, target_pipeline)
        configured_dir = reference_dir or os.environ.get("REFERENCE_RUN_DIR")
        if not configured_dir:
            raise ValueError("from_reference needs REFERENCE_RUN_DIR set to an earlier attack run's logdir")
        self.reference_dir = Path(configured_dir)
        if not self.reference_dir.is_dir():
            raise FileNotFoundError(f"REFERENCE_RUN_DIR {str(self.reference_dir)!r} is not a directory")
        self.reference_format = reference_format or os.environ.get("REFERENCE_FORMAT", "logged")
        if self.reference_format not in FORMATS:
            raise ValueError(f"REFERENCE_FORMAT must be one of {FORMATS}, not {self.reference_format!r}")

    def attack(self, user_task: BaseUserTask, injection_task: BaseInjectionTask) -> dict[str, str]:
        case = f"{self.task_suite.name}/{user_task.ID}/*/{injection_task.ID}.json"
        paths = sorted(self.reference_dir.glob(case)) + sorted(self.reference_dir.glob(f"*/{case}"))
        if not paths:
            raise FileNotFoundError(f"No reference result for {case} in {self.reference_dir}")
        if len(paths) > 1:
            raise ValueError(f"Several reference results for {case}: {[str(path) for path in paths]}")

        injections = json.loads(paths[0].read_text(encoding="utf-8"))["injections"]
        defaults = self.task_suite.get_injection_vector_defaults()
        unknown = injections.keys() - defaults.keys()
        if not injections or unknown:
            raise ValueError(f"{paths[0]} has no usable injections (unknown vectors: {sorted(unknown)})")
        if self.reference_format == "appended":
            return {vector: _append_to_default(defaults[vector], payload) for vector, payload in injections.items()}
        return injections

"""Summarize an AgentDojo run directory into Markdown tables. Read-only, no API calls.

    python -m metrics.summarize runs/qwen3-235b-a22b-2507
    python -m metrics.summarize runs/qwen3-235b-a22b-2507_chatinject

Works on runs still in progress: result JSONs without `utility` (task running, or crashed)
are skipped and counted as incomplete. Results are grouped by path
<run_dir>/<pipeline>/<suite>/<task>/<attack|none>/<injection_task|none>.json into:
  - benign:              user_task_*/none/none.json             -> Utility
  - injection as user:   injection_task_*/none/none.json        -> injection tasks the model can do
  - attacked:            user_task_*/<attack>/injection_task_*  -> Utility under attack, ASR
ASR is agentdojo's `security` field (True = the injection task was executed).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suites


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}% ({k}/{n})" if n else "–"


def load_results(run_dir: Path) -> tuple[dict, int]:
    """Returns ({(category, attack, suite): [result dicts]}, number of incomplete files)."""
    groups = defaultdict(list)
    incomplete = 0
    for path in sorted(run_dir.glob("*/*/*/*/*.json")):
        suite, task, attack = path.parts[-4], path.parts[-3], path.parts[-2]
        try:
            result = json.loads(path.read_text())
        except ValueError:
            result = {}
        if "utility" not in result:
            incomplete += 1
            continue
        if attack == "none":
            category = "injection_as_user" if task.startswith("injection_task") else "benign"
        else:
            category = "attacked"
        groups[(category, attack, suite)].append(result)
    return groups, incomplete


def table(title: str, columns: list[str], rows: list[list[str]]) -> str:
    lines = [f"## {title}", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def summarize(run_dir: Path, version: str) -> str:
    groups, incomplete = load_results(run_dir)
    suites = get_suites(version)
    expected = {
        "benign": {s: len(t.user_tasks) for s, t in suites.items()},
        "injection_as_user": {s: len(t.injection_tasks) for s, t in suites.items()},
        "attacked": {s: len(t.user_tasks) * len(t.injection_tasks) for s, t in suites.items()},
    }
    metrics = {
        "benign": [("Utility", "utility")],
        "injection_as_user": [("Injection tasks solved", "utility")],
        "attacked": [("Utility under attack", "utility"), ("ASR", "security")],
    }
    titles = {"benign": "benign", "injection_as_user": "injection tasks run as user tasks", "attacked": "attacked"}

    out = []
    for category, attack in sorted({(c, a) for c, a, _ in groups}):
        rows = []
        totals = defaultdict(int)
        for suite in suites:
            results = groups.get((category, attack, suite), [])
            if not results:
                continue
            row = [suite, f"{len(results)}/{expected[category][suite]}"]
            for _, field in metrics[category]:
                k = sum(bool(r[field]) for r in results)
                row.append(pct(k, len(results)))
                totals[field] += k
            totals["n"] += len(results)
            rows.append(row)
        overall = ["**Overall**", f"{totals['n']}/{sum(expected[category].values())}"]
        overall += [pct(totals[field], totals["n"]) for _, field in metrics[category]]
        rows.append(overall)
        name = titles[category] if attack == "none" else f"{attack} ({titles[category]})"
        out.append(table(name, ["suite", "done/expected"] + [m for m, _ in metrics[category]], rows))
    out.append(f"incomplete files skipped: {incomplete}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--benchmark-version", default="v1.2.2")
    args = ap.parse_args()
    print(summarize(args.run_dir, args.benchmark_version))


if __name__ == "__main__":
    main()

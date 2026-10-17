"""Print ASR and Utility under attack of a finished AgentDojo attack run.

    python -m metrics.summarize runs/qwen3-235b-a22b-2507_chatinject

Reads <run_dir>/<pipeline>/<suite>/user_task_*/<attack>/<injection_task>.json. ASR is agentdojo's
`security` (True = the injection task was executed). Benign results (none/none.json) are ignored.
"""

import argparse
import json
from pathlib import Path


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f} ({k}/{n})"


def check_complete(run_dir: Path, expected: int = 949) -> bool:
    """True if the run has `expected` valid attacked records (JSON with a `utility` field)."""
    valid = 0
    for p in run_dir.glob("*/*/user_task_*/*/*.json"):
        if p.parent.name == "none":
            continue
        try:
            valid += "utility" in json.loads(p.read_text())
        except ValueError:
            pass
    return valid == expected


def get_results(run_dir: Path, if_detailed: bool = False) -> list[dict]:
    paths = [p for p in run_dir.glob("*/*/user_task_*/*/*.json") if p.parent.name != "none"]
    compact_results = []
    for p in paths:
        r = json.loads(p.read_text())
        c = {
            "suite": r.get("suite_name"),
            "security": r.get("security"),
            "utility": r.get("utility"),
        }
        compact_results.append(c)

    groups = {"overall": compact_results}
    if if_detailed:
        by_suite = {}
        for c in compact_results:
            by_suite.setdefault(c["suite"], []).append(c)
        groups = {**dict(sorted(by_suite.items())), **groups}

    rows = []
    for suite, results in groups.items():
        n = len(results)
        rows.append({
            "suite": suite,
            "n": n,
            "asr": pct(sum(bool(r["security"]) for r in results), n),
            "utility": pct(sum(bool(r["utility"]) for r in results), n),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-d", "--detailed", action="store_true")
    args = ap.parse_args()
    if not check_complete(args.run_dir):
        print(f"WARNING: {args.run_dir} does not have all expected valid JSONs (utility field).")
    for row in get_results(args.run_dir, args.detailed):
        print(f"{row['suite']}: ASR {row['asr']}, Utility {row['utility']}")


if __name__ == "__main__":
    main()

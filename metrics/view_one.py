"""Print ASR and Utility under attack of a finished AgentDojo attack run.

    python metrics/view_one.py runs/qwen3-235b-a22b-2507_chatinject [-d] [-s banking -s slack ...]

Reads <run_dir>/<pipeline>/<suite>/user_task_*/<attack>/<injection_task>.json. ASR is agentdojo's
`security` (True = the injection task was executed). Benign results (none/none.json) are ignored.
"""

import argparse
import json
import sys
from pathlib import Path

# attacked records per suite in a full v1.2.2 run (user tasks x injection tasks)
EXPECTED = {"workspace": 560, "slack": 105, "travel": 140, "banking": 144}


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f} ({k}/{n})"


def attacked_paths(run_dir: Path, suites: list[str]) -> list[Path]:
    """Attacked result JSONs, optionally only those of `suites`."""
    return sorted(p for p in run_dir.glob("*/*/user_task_*/*/*.json")
                  if p.parent.name != "none" and (not suites or p.relative_to(run_dir).parts[1] in suites))


def find_incomplete(run_dir: Path, suites: list[str]) -> tuple[int, list[str]]:
    """Count valid attacked records (JSON with a `utility` field) and name the ones that exist but
    are incomplete: a crashed case leaves a JSON without `utility`, a corrupted one does not parse."""
    valid, incomplete = 0, []
    for p in attacked_paths(run_dir, suites):
        try:
            ok = "utility" in json.loads(p.read_text())
        except ValueError:
            ok = False
        if ok:
            valid += 1
        else:
            incomplete.append(str(p.relative_to(run_dir)))
    return valid, incomplete


def get_results(run_dir: Path, suites: list[str], if_detailed: bool = False) -> list[dict]:
    compact_results = []
    for p in attacked_paths(run_dir, suites):
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
    ap.add_argument("-s", "--suite", dest="suites", action="append", default=[], choices=sorted(EXPECTED),
                    help="only count this suite (repeatable); default all")
    ap.add_argument("--expected", type=int, default=None,
                    help="attacked records in a full run; default from the selected suites (949 for all)")
    args = ap.parse_args()
    if not args.run_dir.is_dir():
        sys.exit(f"{args.run_dir}: no such directory")
    if args.expected is None:
        args.expected = sum(n for suite, n in EXPECTED.items() if not args.suites or suite in args.suites)
    valid, incomplete = find_incomplete(args.run_dir, args.suites)
    if valid != args.expected:
        absent = args.expected - valid - len(incomplete)
        print(f"WARNING: {args.run_dir} has {valid}/{args.expected} valid records: "
              f"{len(incomplete)} incomplete (counted as utility=False, security=False), "
              f"{absent} not found (left out of the totals)")
        for name in incomplete:
            print(f"  incomplete: {name}")
    for row in get_results(args.run_dir, args.suites, args.detailed):
        print(f"{row['suite']}: ASR {row['asr']}, Utility {row['utility']}")


if __name__ == "__main__":
    main()

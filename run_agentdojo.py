"""Run the AgentDojo benchmark with one subprocess per case, N at a time.

agentdojo's own --max-workers only parallelizes across suites (and is broken in 0.1.35),
so a full attack run is ~950 sequential cases. This launches
`python -m agentdojo.scripts.benchmark` once per (suite, user_task[, injection_task]),
skipping cases whose result JSON already exists, so it is safe to stop and re-run.

    python run_agentdojo.py --model-id qwen/qwen3-235b-a22b-2507 \
        --logdir runs/qwen3-235b-a22b-2507 -j 16                        # benign
    python run_agentdojo.py --model-id qwen/qwen3-235b-a22b-2507 \
        --logdir runs/qwen3-235b-a22b-2507_important_instr \
        --attack important_instructions -s workspace -j 16               # attack
    python run_agentdojo.py --model-id qwen/qwen3-235b-a22b-2507 \
        --logdir runs/qwen3-235b-a22b-2507_chatinject \
        -ml chatinject_attacks --attack chat_inject_qwen3 -j 16          # custom attack

Per-case stdout/stderr goes to <logdir>/_logs/. A failed case is retried --retries times; cases that
still fail are listed in <logdir>/_failed.txt and re-attempted on the next run (a crashed case's
partial result JSON, which has no `utility`, does not count as done).
Summarize results with `python -m metrics.summarize <logdir>`.
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from agentdojo.task_suite.load_suites import get_suites

PIPELINE_NAME = "openai-compatible"  # fixed pipeline name for --model OPENAI_COMPATIBLE


def list_cases(version: str, suites: list[str], attack: str | None) -> list[tuple[str, str, str | None]]:
    cases = []
    for suite_name, suite in get_suites(version).items():
        if suites and suite_name not in suites:
            continue
        for user_task_id in suite.user_tasks:
            if attack is None:
                cases.append((suite_name, user_task_id, None))
            else:
                for injection_task_id in suite.injection_tasks:
                    cases.append((suite_name, user_task_id, injection_task_id))
    return cases


def result_path(logdir: Path, case: tuple[str, str, str | None], attack: str | None) -> Path:
    suite_name, user_task_id, injection_task_id = case
    return logdir / PIPELINE_NAME / suite_name / user_task_id / (attack or "none") / f"{injection_task_id or 'none'}.json"


def is_done(path: Path) -> bool:
    """A case is done only if its result JSON is complete. agentdojo writes the file while a
    task runs, so a crashed case leaves a JSON without `utility` that must be re-run."""
    try:
        return "utility" in json.loads(path.read_text())
    except (OSError, ValueError):
        return False


def log_path(args: argparse.Namespace, case: tuple[str, str, str | None]) -> Path:
    suite_name, user_task_id, injection_task_id = case
    return args.logdir / "_logs" / f"{suite_name}__{user_task_id}__{injection_task_id or 'none'}.log"


def last_error(log_file: Path) -> str:
    """Last `SomethingError: ...` / `SomethingException: ...` line of a case log."""
    errors = re.findall(r"^[\w.]+(?:Error|Exception):.*$", log_file.read_text(errors="replace"), re.MULTILINE)
    return errors[-1].strip() if errors else "unknown error"


def case_name(case: tuple[str, str, str | None]) -> str:
    return "/".join(c for c in case if c)


def run_case(args: argparse.Namespace, case: tuple[str, str, str | None]) -> tuple[tuple, int]:
    suite_name, user_task_id, injection_task_id = case
    cmd = [
        sys.executable, "-m", "agentdojo.scripts.benchmark",
        "--model", "OPENAI_COMPATIBLE",
        "--model-id", args.model_id,
        "--logdir", str(args.logdir),
        "--benchmark-version", args.benchmark_version,
        "-s", suite_name,
        "-ut", user_task_id,
    ]
    for module in args.modules_to_load:
        cmd += ["-ml", module]
    if args.attack is not None:
        cmd += ["--attack", args.attack, "-it", injection_task_id]
    with log_path(args, case).open("w") as f:
        returncode = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False).returncode
    return case, returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--attack", default=None)
    ap.add_argument("-ml", "--module-to-load", dest="modules_to_load", action="append", default=[],
                    help="module registering custom attacks, e.g. chatinject_attacks")
    ap.add_argument("-s", "--suite", dest="suites", action="append", default=[])
    ap.add_argument("-j", "--jobs", type=int, default=16)
    ap.add_argument("--benchmark-version", default="v1.2.2")
    ap.add_argument("--retries", type=int, default=2, help="re-runs of a failed case before giving up")
    args = ap.parse_args()

    (args.logdir / "_logs").mkdir(parents=True, exist_ok=True)
    cases = list_cases(args.benchmark_version, args.suites, args.attack)
    todo = [c for c in cases if not is_done(result_path(args.logdir, c, args.attack))]
    print(f"{len(cases)} cases, {len(cases) - len(todo)} already done, {len(todo)} to run with {args.jobs} jobs")

    failed = []
    finished = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        pending = {pool.submit(run_case, args, c): (c, 1) for c in todo}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                case, attempt = pending.pop(future)
                _, returncode = future.result()
                if returncode == 0:
                    finished += 1
                    print(f"[{finished}/{len(todo)}] {case_name(case)} ok", flush=True)
                elif attempt <= args.retries:
                    print(f"{case_name(case)} failed (attempt {attempt}): {last_error(log_path(args, case))}; retrying",
                          flush=True)
                    pending[pool.submit(run_case, args, case)] = (case, attempt + 1)
                else:
                    finished += 1
                    failed.append((case, last_error(log_path(args, case))))
                    print(f"[{finished}/{len(todo)}] {case_name(case)} FAILED after {attempt} attempts", flush=True)

    failed_file = args.logdir / "_failed.txt"
    failed_file.write_text("".join(f"{case_name(case)}\t{error}\n" for case, error in failed))
    print(f"done: {len(todo) - len(failed)} ok, {len(failed)} failed (listed in {failed_file})")
    for case, error in failed:
        print(f"  {case_name(case)}: {error}")


if __name__ == "__main__":
    main()

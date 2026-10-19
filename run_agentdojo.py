"""Run the AgentDojo v1.2.2 benchmark with one subprocess per case, N at a time.

agentdojo's own --max-workers only parallelizes across suites (and is broken in 0.1.35),
so a full attack run is ~950 sequential cases. This runs agentdojo's benchmark CLI once per
(suite, user_task[, injection_task]) in a forked child process, skipping cases whose result JSON
already exists, so it is safe to stop and re-run. agentdojo is imported once in the parent and
inherited by every fork: a fresh `python -m agentdojo.scripts.benchmark` spends over a minute
importing ~24k files from Lustre, per case.

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
import importlib
import json
import logging
import multiprocessing as mp
import os
import re
import sys
from multiprocessing.connection import wait
from pathlib import Path

from agentdojo.attacks.attack_registry import ATTACKS
from agentdojo.scripts import benchmark
from agentdojo.task_suite.load_suites import get_suites
from rich.logging import RichHandler

PIPELINE_NAME = "openai-compatible"  # fixed pipeline name for --model OPENAI_COMPATIBLE
BENCHMARK_VERSION = "v1.2.2"


def list_cases(suites: list[str], attack: str | None) -> list[tuple[str, str, str | None]]:
    cases = []
    for suite_name, suite in get_suites(BENCHMARK_VERSION).items():
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


def load_endpoint(model_id: str) -> None:
    """Point agentdojo at the vLLM server for `model_id` if one is up: vllm/serve_*.sh write its URL
    and key to vllm/endpoint-<served name>.env. They override the environment and .env, which may
    still hold a previous job's node and key."""
    path = Path(__file__).parent / "vllm" / f"endpoint-{model_id}.env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.startswith("export "):
            key, _, value = line.removeprefix("export ").partition("=")
            os.environ[key] = value
    print(f"endpoint {os.environ['OPENAI_COMPATIBLE_BASE_URL']} from {path}")


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


def run_case(args: argparse.Namespace, case: tuple[str, str, str | None]) -> None:
    """Body of a forked child: send stdout/stderr to the case log and run the benchmark CLI,
    which exits with its status code."""
    suite_name, user_task_id, injection_task_id = case
    cli_args = [
        "--model", "OPENAI_COMPATIBLE",
        "--model-id", args.model_id,
        "--logdir", str(args.logdir),
        "--benchmark-version", BENCHMARK_VERSION,
        "-s", suite_name,
        "-ut", user_task_id,
    ]
    for module in args.modules_to_load:
        cli_args += ["-ml", module]
    if args.attack is not None:
        cli_args += ["--attack", args.attack, "-it", injection_task_id]
    log_fd = os.open(log_path(args, case), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    # same logging setup as `python -m agentdojo.scripts.benchmark`
    logging.basicConfig(format="%(message)s", level=logging.INFO, datefmt="%H:%M:%S",
                        handlers=[RichHandler(show_path=False, markup=True)], force=True)
    benchmark.main(cli_args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--attack", default=None)
    ap.add_argument("-ml", "--module-to-load", dest="modules_to_load", action="append", default=[],
                    help="module registering custom attacks, e.g. chatinject_attacks")
    ap.add_argument("-s", "--suite", dest="suites", action="append", default=[])
    ap.add_argument("-j", "--jobs", type=int, default=16)
    args = ap.parse_args()
    load_endpoint(args.model_id)

    for module in args.modules_to_load:  # import attack modules once too, so forks inherit them
        importlib.import_module(module)
    if args.attack is not None and args.attack not in ATTACKS:
        sys.exit(f"unknown attack {args.attack!r} (missing -ml?); registered: {', '.join(sorted(ATTACKS))}")

    (args.logdir / "_logs").mkdir(parents=True, exist_ok=True)
    cases = list_cases(args.suites, args.attack)
    todo = [c for c in cases if not is_done(result_path(args.logdir, c, args.attack))]
    print(f"{len(cases)} cases, {len(cases) - len(todo)} already done, {len(todo)} to run with {args.jobs} jobs")

    fork = mp.get_context("fork")
    queue = [(c, 1) for c in todo]
    running = {}  # process sentinel -> (process, case, attempt)
    failed = []
    finished = 0
    while queue or running:
        while queue and len(running) < args.jobs:
            case, attempt = queue.pop(0)
            sys.stdout.flush()  # don't let the child inherit unflushed parent output
            proc = fork.Process(target=run_case, args=(args, case))
            proc.start()
            running[proc.sentinel] = (proc, case, attempt)
        for sentinel in wait(list(running)):
            proc, case, attempt = running.pop(sentinel)
            proc.join()
            if proc.exitcode == 0:
                finished += 1
                print(f"[{finished}/{len(todo)}] {case_name(case)} ok", flush=True)
            elif attempt <= 3:
                queue.append((case, attempt + 1))
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

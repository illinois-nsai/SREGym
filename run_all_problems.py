#!/usr/bin/env python3
"""Run all SREGym problems with a fixed agent/model and per-problem logs."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

DEFAULT_AGENT = "stratus"
DEFAULT_MODEL = "qwen-8b"
REPO_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = REPO_ROOT / "sregym" / "conductor" / "problems" / "registry.py"
TASKLIST_PATH = REPO_ROOT / "sregym" / "conductor" / "tasklist.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all problems via `uv run python main.py --problem <problem> ...` "
            "and write logs to ~/logs/<problem>.log"
        )
    )
    parser.add_argument("--agent", default=DEFAULT_AGENT, help=f"Agent name (default: {DEFAULT_AGENT})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--logs-dir",
        default="~/logs",
        help="Directory to write per-problem logs (default: ~/logs)",
    )
    parser.add_argument(
        "--no-force-build",
        action="store_true",
        help="Do not pass --force-build to main.py",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining problems when one command fails",
    )
    parser.add_argument(
        "--tasklist-only",
        action="store_true",
        help="Use tasklist selection from registry instead of all problems.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based problem index to start from (default: 1).",
    )
    parser.add_argument(
        "--start-from",
        help="Problem ID to start from.",
    )
    return parser.parse_args()


def load_all_problem_ids() -> list[str]:
    source = REGISTRY_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(REGISTRY_PATH))

    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "PROBLEM_REGISTRY"
            ):
                continue
            if not isinstance(node.value, ast.Dict):
                raise RuntimeError("self.PROBLEM_REGISTRY is not a dictionary literal")

            problem_ids: list[str] = []
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    problem_ids.append(key.value)

            if not problem_ids:
                raise RuntimeError("No problem IDs found in PROBLEM_REGISTRY")
            return problem_ids

    raise RuntimeError(f"Could not locate self.PROBLEM_REGISTRY in {REGISTRY_PATH}")


def load_tasklist_problem_ids() -> list[str]:
    if not TASKLIST_PATH.exists():
        return load_all_problem_ids()

    problem_ids: list[str] = []
    in_all_section = False
    in_problems_section = False

    for raw_line in TASKLIST_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if stripped == "all:":
            in_all_section = True
            in_problems_section = False
            continue

        if in_all_section and not raw_line.startswith(" "):
            in_all_section = False
            in_problems_section = False

        if in_all_section and stripped == "problems:":
            in_problems_section = True
            continue

        if in_problems_section:
            if stripped.startswith("- "):
                problem_ids.append(stripped[2:].strip())
                continue
            if not raw_line.startswith(" "):
                break

    if not problem_ids:
        raise RuntimeError(f"No problems found in {TASKLIST_PATH}")
    return problem_ids


def get_problem_ids(tasklist_only: bool) -> list[str]:
    return load_tasklist_problem_ids() if tasklist_only else load_all_problem_ids()


def select_problem_range(problem_ids: list[str], start_index: int, start_from: str | None) -> list[str]:
    if start_index < 1:
        raise ValueError("--start-index must be at least 1")

    if start_from:
        if start_from not in problem_ids:
            raise ValueError(f"Problem ID not found: {start_from}")
        start_pos = problem_ids.index(start_from)
    else:
        start_pos = start_index - 1

    if start_pos >= len(problem_ids):
        raise ValueError(f"Start position {start_pos + 1} exceeds problem count {len(problem_ids)}")

    return problem_ids[start_pos:]


def main() -> int:
    args = parse_args()

    logs_dir = Path(args.logs_dir).expanduser().resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    problem_ids = get_problem_ids(tasklist_only=args.tasklist_only)
    try:
        problem_ids = select_problem_range(problem_ids, args.start_index, args.start_from)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    total = len(problem_ids)
    if total == 0:
        print("No problems found.", file=sys.stderr)
        return 1

    print(f"Running {total} problems. Logs: {logs_dir}")

    failures: list[tuple[str, int]] = []

    for idx, problem in enumerate(problem_ids, start=1):
        log_path = logs_dir / f"{problem}.log"
        cmd = [
            "uv",
            "run",
            "python",
            "main.py",
            "--problem",
            problem,
            "--agent",
            args.agent,
            "--model",
            args.model,
        ]
        if not args.no_force_build:
            cmd.append("--force-build")

        print(f"[{idx}/{total}] {problem} -> {log_path}")

        with log_path.open("w", encoding="utf-8") as log_file:
            rc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False).returncode

        if rc != 0:
            failures.append((problem, rc))
            print(f"FAILED: {problem} (exit={rc})")
            if not args.continue_on_error:
                break

    if failures:
        print("\nFailures:")
        for problem, rc in failures:
            print(f"- {problem}: exit={rc}")
        return 1

    print("\nAll runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    module: str
    required_markers: tuple[str, ...]
    forbidden_markers: tuple[str, ...]


TASKS = (
    TaskSpec(
        "task1_leakage",
        "Leakage-safe preprocessing",
        "tasks.task1_leakage.solution",
        ("Pipeline", "StandardScaler"),
        ("fit_transform(X)",),
    ),
    TaskSpec(
        "task2_validation",
        "Time-aware validation",
        "tasks.task2_validation.solution",
        ("TimeSeriesSplit",),
        ("train_test_split", "shuffle=True"),
    ),
    TaskSpec(
        "task3_unstable_training",
        "Stable and reproducible training",
        "tasks.task3_unstable_training.solution",
        ("Pipeline", "StandardScaler", "random_state=42"),
        ("RandomState(None)", "eta0=1.0"),
    ),
)


@dataclass
class TaskResult:
    key: str
    title: str
    score: float
    repeat_score: float
    deterministic: bool
    finite: bool
    structure_passed: bool
    elapsed_ms: float
    status: str
    details: list[str]


def evaluate_task(spec: TaskSpec) -> TaskResult:
    source_path = Path(spec.module.replace(".", "/") + ".py")
    source = source_path.read_text(encoding="utf-8")
    details: list[str] = []
    missing = [marker for marker in spec.required_markers if marker not in source]
    present_forbidden = [marker for marker in spec.forbidden_markers if marker in source]
    structure_passed = not missing and not present_forbidden
    if missing:
        details.append(f"missing markers: {', '.join(missing)}")
    if present_forbidden:
        details.append(f"forbidden markers: {', '.join(present_forbidden)}")

    started = time.perf_counter()
    result: Any = importlib.import_module(spec.module).run()
    elapsed_ms = (time.perf_counter() - started) * 1000
    repeat_result: Any = importlib.import_module(spec.module).run()
    score = float(result)
    repeat_score = float(repeat_result)
    finite = math.isfinite(score) and math.isfinite(repeat_score)
    deterministic = score == repeat_score
    passed = structure_passed and finite and deterministic
    if not finite:
        details.append("score is not finite")
    if not deterministic:
        details.append("repeat run returned a different score")
    return TaskResult(
        spec.key,
        spec.title,
        score,
        repeat_score,
        deterministic,
        finite,
        structure_passed,
        elapsed_ms,
        "PASS" if passed else "FAIL",
        details,
    )


def run_benchmark() -> list[TaskResult]:
    return [evaluate_task(task) for task in TASKS]


def as_report(results: list[TaskResult]) -> dict[str, Any]:
    passed = sum(item.status == "PASS" for item in results)
    return {
        "benchmark": "ml-leakage-benchmark",
        "tasks": [asdict(item) for item in results],
        "summary": {"passed": passed, "total": len(results), "status": "PASS" if passed == len(results) else "FAIL"},
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Benchmark report", "", "| Task | Score | Repeat | Deterministic | Structure | Status |", "| --- | ---: | ---: | :---: | :---: | :---: |"]
    for task in report["tasks"]:
        lines.append(f"| {task['title']} | {task['score']:.4f} | {task['repeat_score']:.4f} | {'yes' if task['deterministic'] else 'no'} | {'yes' if task['structure_passed'] else 'no'} | **{task['status']}** |")
    summary = report["summary"]
    lines.extend(["", f"**Summary:** {summary['passed']}/{summary['total']} tasks passed."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ML leakage benchmark solutions.")
    parser.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    parser.add_argument("--output", type=Path, help="Write the report to a file instead of stdout.")
    args = parser.parse_args()
    report = as_report(run_benchmark())
    if args.format == "json":
        output = json.dumps(report, indent=2) + "\n"
    elif args.format == "markdown":
        output = markdown_report(report)
    else:
        output = "\n".join(f"{task['status']}  {task['title']}: {task['score']:.4f} (repeat {task['repeat_score']:.4f})" for task in report["tasks"])
        output += f"\n\nSummary: {report['summary']['passed']}/{report['summary']['total']} passed.\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0 if report["summary"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

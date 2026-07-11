"""Generate the complete 45-task Mutmut research benchmark."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TypedDict


class ProjectConfig(TypedDict):
    """One registered mutation seed project."""

    project_id: str
    source_root: str
    source_paths: list[str]
    test_paths: list[str]
    hidden_test_root: str
    task_prefix: str


def load_projects(root: Path) -> list[ProjectConfig]:
    """Load the three committed seed-project registrations."""
    path = root / "benchmark_seeds" / "projects.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("benchmark_seeds/projects.json must contain a list.")
    projects: list[ProjectConfig] = []
    for item in raw:
        if not isinstance(item, dict):
            raise SystemExit("Every seed-project entry must be an object.")
        projects.append(ProjectConfig(**item))
    expected = {
        "mutmut_algorithms",
        "mutmut_collections",
        "mutmut_textdata",
    }
    found = {item["project_id"] for item in projects}
    if found != expected or len(projects) != 3:
        raise SystemExit(
            "Step 16C requires exactly the three registered seed projects."
        )
    return projects


def run_project(
    *,
    root: Path,
    project: ProjectConfig,
    tasks_per_project: int,
    max_per_function: int,
    mutmut_timeout_seconds: int,
    test_timeout_seconds: int,
) -> None:
    """Generate one validated project slice."""
    project_id = project["project_id"]
    output_root = root / "generated_benchmarks" / project_id
    work_root = root / "artifacts" / "mutmut_generation_step16c"

    if output_root.exists():
        shutil.rmtree(output_root)

    command = [
        sys.executable,
        "scripts/generate_mutmut_benchmark.py",
        "--source-root",
        project["source_root"],
        "--project-id",
        project_id,
        "--hidden-test-root",
        str(Path(project["source_root"]) / project["hidden_test_root"]),
        "--output-root",
        str(Path("generated_benchmarks") / project_id),
        "--work-root",
        str(Path("artifacts") / "mutmut_generation_step16c"),
        "--task-prefix",
        project["task_prefix"],
        "--max-tasks",
        str(tasks_per_project),
        "--max-per-function",
        str(max_per_function),
        "--mutmut-timeout-seconds",
        str(mutmut_timeout_seconds),
        "--test-timeout-seconds",
        str(test_timeout_seconds),
        "--force",
    ]
    for source_path in project["source_paths"]:
        command.extend(["--source-path", source_path])
    for test_path in project["test_paths"]:
        command.extend(["--test-path", test_path])
    command.extend(["--pytest-pythonpath", "src"])

    print("=" * 72, flush=True)
    print(f"GENERATE {project_id}", flush=True)
    print("COMMAND " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"Generation failed for {project_id} with exit code {completed.returncode}."
        )

    manifests = sorted(output_root.glob("*/task.json"))
    if len(manifests) != tasks_per_project:
        raise SystemExit(
            f"{project_id} generated {len(manifests)} tasks; "
            f"{tasks_per_project} were required."
        )
    print(
        f"PROJECT_COMPLETE {project_id} TASKS={len(manifests)} "
        f"WORK_ROOT={work_root / project_id}",
        flush=True,
    )


def main() -> None:
    """Generate all three deterministic research slices."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-per-project", type=int, default=15)
    parser.add_argument("--max-per-function", type=int, default=3)
    parser.add_argument("--mutmut-timeout-seconds", type=int, default=1800)
    parser.add_argument("--test-timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    if args.tasks_per_project != 15:
        raise SystemExit("Step 16C requires exactly 15 tasks per project.")
    if args.max_per_function < 1:
        raise SystemExit("--max-per-function must be positive.")

    root = Path.cwd().resolve(strict=True)
    version = subprocess.run(
        ["mutmut", "--version"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    version_text = (version.stdout or version.stderr).strip()
    if version.returncode != 0 or "3.6.0" not in version_text:
        raise SystemExit(f"Mutmut 3.6.0 is required; found {version_text!r}.")

    work_root = root / "artifacts" / "mutmut_generation_step16c"
    if work_root.exists():
        shutil.rmtree(work_root)

    projects = load_projects(root)
    for project in projects:
        run_project(
            root=root,
            project=project,
            tasks_per_project=args.tasks_per_project,
            max_per_function=args.max_per_function,
            mutmut_timeout_seconds=args.mutmut_timeout_seconds,
            test_timeout_seconds=args.test_timeout_seconds,
        )

    manifests = sorted((root / "generated_benchmarks").glob("mutmut_*/*/task.json"))
    if len(manifests) != 45:
        raise SystemExit(
            f"Expected 45 generated Mutmut manifests, found {len(manifests)}."
        )
    print("=" * 72, flush=True)
    print("MUTMUT_RESEARCH_TASKS=45", flush=True)


if __name__ == "__main__":
    main()

"""Validation for all registered Mutmut seed projects."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(".").resolve()
REGISTRY_PATH = PROJECT_ROOT / "benchmark_seeds" / "projects.json"


def load_projects() -> list[dict[str, object]]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def run_suite(
    project: dict[str, object], suite: str
) -> subprocess.CompletedProcess[str]:
    source_root = PROJECT_ROOT / str(project["source_root"])
    suite_value = project[suite]
    if isinstance(suite_value, list):
        target = source_root / str(suite_value[0])
    else:
        target = source_root / str(suite_value)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(source_root / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            str(target),
        ],
        cwd=source_root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def test_seed_registry_has_three_independent_projects() -> None:
    projects = load_projects()
    identifiers = [str(project["project_id"]) for project in projects]

    assert identifiers == [
        "mutmut_algorithms",
        "mutmut_collections",
        "mutmut_textdata",
    ]
    assert len(set(identifiers)) == 3

    for project in projects:
        source_root = PROJECT_ROOT / str(project["source_root"])
        assert source_root.is_dir()
        assert (source_root / "src").is_dir()
        assert (source_root / str(project["test_paths"][0])).is_dir()
        assert (source_root / str(project["hidden_test_root"])).is_dir()


@pytest.mark.parametrize(
    "project",
    load_projects(),
    ids=lambda item: str(item["project_id"]),
)
def test_registered_visible_seed_suite_passes(project: dict[str, object]) -> None:
    result = run_suite(project, "test_paths")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout


@pytest.mark.parametrize(
    "project",
    load_projects(),
    ids=lambda item: str(item["project_id"]),
)
def test_registered_hidden_seed_suite_passes(project: dict[str, object]) -> None:
    result = run_suite(project, "hidden_test_root")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout

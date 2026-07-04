from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from patchpilot.benchmark import load_manifest

MANIFESTS = sorted(Path("benchmarks").glob("*/task.json"))


def _failure_count(output: str) -> int:
    match = re.search(r"(\d+) failed", output)
    if match is None:
        return 0
    return int(match.group(1))


@pytest.mark.parametrize("manifest_path", MANIFESTS, ids=lambda p: p.parent.name)
def test_benchmark_manifest_and_initial_failure_count(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    repository = Path(manifest.repository_root)

    assert repository.is_dir()
    assert (repository / "src").is_dir()
    assert (repository / "tests").is_dir()
    assert manifest.allowed_paths == ["src"]
    assert manifest.forbidden_paths == ["tests"]

    result = subprocess.run(
        manifest.test_command,
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0
    assert _failure_count(output) == manifest.expected_initial_failures

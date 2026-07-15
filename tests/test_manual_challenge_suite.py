"""Structural regression tests for the 53-task research benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from patchpilot.benchmark import load_manifest

CHALLENGE_ROOT = Path("challenge_benchmarks")
RESEARCH_CATALOG = Path("generated_benchmarks/research_suite.json")


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_manual_challenge_catalog_has_eight_hard_tasks() -> None:
    manifests = sorted(CHALLENGE_ROOT.glob("*/task.json"))
    assert len(manifests) == 8

    task_ids: set[str] = set()
    for path in manifests:
        manifest = load_manifest(path)
        task_ids.add(manifest.task_id)
        assert manifest.difficulty == "hard"
        assert manifest.allowed_paths == ["src"]
        assert manifest.forbidden_paths == ["tests"]
        assert manifest.hidden_test_root is not None
        assert manifest.expected_hidden_tests is not None
        assert manifest.expected_hidden_tests >= 3

        repository = Path(manifest.repository_root)
        hidden_root = Path(manifest.hidden_test_root)
        assert repository.is_dir()
        assert hidden_root.is_dir()
        assert not hidden_root.is_relative_to(repository)

        provenance = read_json(path.parent / "provenance.json")
        assert provenance["origin_type"] == "manual_challenge"
        assert provenance["task_id"] == manifest.task_id
        assert int(provenance["changed_line_count"]) >= 3
        assert provenance["defect_diff"]
        assert provenance["defect_patterns"]
        clean = provenance["clean_reference_validation"]
        assert isinstance(clean, dict)
        assert clean["visible_passed"] is True
        assert clean["hidden_passed"] is True

    assert len(task_ids) == 8


def test_primary_research_catalog_has_53_unique_tasks() -> None:
    catalog = read_json(RESEARCH_CATALOG)
    assert catalog["task_count"] == 53
    assert catalog["mutmut_task_count"] == 45
    assert catalog["manual_challenge_task_count"] == 8
    assert catalog["sanity_task_count_excluded"] == 12

    manifest_paths = catalog["manifest_paths"]
    assert isinstance(manifest_paths, list)
    assert len(manifest_paths) == 53
    assert len(set(manifest_paths)) == 53
    assert all(Path(str(path)).is_file() for path in manifest_paths)


def test_sanity_benchmarks_remain_separate() -> None:
    assert len(list(Path("benchmarks").glob("*/task.json"))) == 12
    catalog = read_json(RESEARCH_CATALOG)
    manifest_paths = [str(path) for path in catalog["manifest_paths"]]
    assert not any(path.startswith("benchmarks/") for path in manifest_paths)

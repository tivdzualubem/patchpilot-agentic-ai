"""Static integrity checks for the generated Mutmut research suite."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from patchpilot.benchmark.manifest import load_manifest
from patchpilot.benchmark.provenance import load_mutmut_provenance

PROJECT_COUNTS = {
    "mutmut_algorithms": 15,
    "mutmut_collections": 15,
    "mutmut_textdata": 15,
}


def research_manifests() -> list[Path]:
    """Return only the three mutation-generated project slices."""
    root = Path("generated_benchmarks")
    manifests: list[Path] = []
    for project_id in PROJECT_COUNTS:
        manifests.extend(sorted((root / project_id).glob("*/task.json")))
    return manifests


def test_mutmut_research_suite_has_expected_composition() -> None:
    manifests = research_manifests()
    assert len(manifests) == 45

    projects: Counter[str] = Counter()
    task_ids: set[str] = set()
    for path in manifests:
        manifest = load_manifest(path)
        provenance = load_mutmut_provenance(path.parent / "provenance.json")
        task_ids.add(manifest.task_id)
        projects[provenance.source_project] += 1

        repository = Path(manifest.repository_root).resolve(strict=True)
        assert manifest.hidden_test_root is not None
        hidden_root = Path(manifest.hidden_test_root).resolve(strict=True)
        assert hidden_root != repository
        assert not hidden_root.is_relative_to(repository)
        assert provenance.mutmut_version == "3.6.0"
        assert provenance.mutant_status == "killed"
        assert len(provenance.source_root_sha256) == 64
        assert len(provenance.exported_repository_sha256) == 64
        assert len(provenance.hidden_tests_sha256) == 64
        assert provenance.visible_initial_failures >= 1
        assert provenance.hidden_initial_failures >= 1

    assert len(task_ids) == 45
    assert dict(projects) == PROJECT_COUNTS


def test_mutmut_research_catalog_matches_manifests() -> None:
    catalog = json.loads(
        Path("generated_benchmarks/mutmut_research_suite.json").read_text(
            encoding="utf-8"
        )
    )

    assert catalog["suite_id"] == "patchpilot-mutmut-research-v1"
    assert catalog["benchmark_kind"] == "mutmut"
    assert catalog["task_count"] == 45
    assert catalog["mutmut_version"] == "3.6.0"
    assert catalog["project_breakdown"] == PROJECT_COUNTS
    assert catalog["sanity_tasks_excluded"] == 12
    assert len(catalog["tasks"]) == 45


def test_sanity_tasks_remain_separate() -> None:
    sanity_manifests = sorted(Path("benchmarks").glob("*/task.json"))
    research_ids = {load_manifest(path).task_id for path in research_manifests()}
    sanity_ids = {load_manifest(path).task_id for path in sanity_manifests}

    assert len(sanity_manifests) == 12
    assert research_ids.isdisjoint(sanity_ids)

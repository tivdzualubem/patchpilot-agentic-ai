"""Validated benchmark catalog loading for PatchPilot evaluations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from patchpilot.benchmark import load_manifest


@dataclass(frozen=True)
class CatalogTask:
    """One benchmark task selected from a validated catalog."""

    task_id: str
    manifest_path: Path
    origin_type: str
    difficulty: str
    defect_category: str


@dataclass(frozen=True)
class BenchmarkCatalog:
    """A validated ordered benchmark catalog."""

    suite_id: str
    catalog_path: Path | None
    catalog_sha256: str | None
    tasks: tuple[CatalogTask, ...]
    declared_task_count: int
    metadata: dict[str, object]

    @property
    def task_count(self) -> int:
        """Return the validated task count."""
        return len(self.tasks)

    def composition(self) -> dict[str, dict[str, int]]:
        """Return deterministic counts by key benchmark attributes."""
        result: dict[str, dict[str, int]] = {}
        for field in ("origin_type", "difficulty", "defect_category"):
            counts: dict[str, int] = {}
            for task in self.tasks:
                value = str(getattr(task, field))
                counts[value] = counts.get(value, 0) + 1
            result[field] = dict(sorted(counts.items()))
        return result


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve_project_file(
    project_root: Path,
    raw_path: object,
    *,
    field_name: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{field_name} must be a non-empty relative path.")

    relative = Path(raw_path)
    if relative.is_absolute():
        raise ValueError(f"{field_name} must be relative: {raw_path}")

    resolved = (project_root / relative).resolve(strict=True)
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"{field_name} escapes the project root: {raw_path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _catalog_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_benchmark_catalog(
    project_root: Path,
    catalog_path: Path | str,
) -> BenchmarkCatalog:
    """Load and validate one ordered benchmark catalog."""
    project_root = project_root.resolve(strict=True)
    raw_catalog = Path(catalog_path)
    if not raw_catalog.is_absolute():
        raw_catalog = project_root / raw_catalog
    resolved_catalog = raw_catalog.resolve(strict=True)
    if not resolved_catalog.is_relative_to(project_root):
        raise ValueError(f"Catalog escapes the project root: {catalog_path}")

    payload = _read_json_object(resolved_catalog)
    suite_id = payload.get("suite_id")
    if not isinstance(suite_id, str) or not suite_id:
        raise ValueError("Catalog suite_id must be a non-empty string.")

    task_count = payload.get("task_count")
    if not isinstance(task_count, int) or isinstance(task_count, bool):
        raise ValueError("Catalog task_count must be an integer.")
    if task_count < 1:
        raise ValueError("Catalog task_count must be positive.")

    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("Catalog tasks must be a list.")
    if len(raw_tasks) != task_count:
        raise ValueError(
            f"Catalog declares {task_count} tasks but contains {len(raw_tasks)}."
        )

    raw_manifest_paths = payload.get("manifest_paths")
    if raw_manifest_paths is not None:
        if not isinstance(raw_manifest_paths, list):
            raise ValueError("Catalog manifest_paths must be a list.")
        task_paths = [
            row.get("manifest_path") if isinstance(row, dict) else None
            for row in raw_tasks
        ]
        if raw_manifest_paths != task_paths:
            raise ValueError("Catalog manifest_paths must exactly match tasks order.")

    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    tasks: list[CatalogTask] = []

    for index, raw_task in enumerate(raw_tasks, start=1):
        if not isinstance(raw_task, dict):
            raise ValueError(f"Catalog task {index} must be an object.")

        task_id = raw_task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"Catalog task {index} has an invalid task_id.")
        if task_id in seen_ids:
            raise ValueError(f"Duplicate catalog task_id: {task_id}")

        manifest_path = _resolve_project_file(
            project_root,
            raw_task.get("manifest_path"),
            field_name=f"tasks[{index}].manifest_path",
        )
        if manifest_path.name != "task.json":
            raise ValueError(
                f"Catalog manifest must be named task.json: {manifest_path}"
            )
        if manifest_path in seen_paths:
            raise ValueError(f"Duplicate catalog manifest: {manifest_path}")

        manifest = load_manifest(manifest_path)
        if manifest.task_id != task_id:
            raise ValueError(
                f"Catalog task_id {task_id!r} does not match "
                f"manifest task_id {manifest.task_id!r}."
            )

        origin_type = raw_task.get("origin_type")
        if not isinstance(origin_type, str) or not origin_type:
            raise ValueError(f"Catalog task {task_id} has invalid origin_type.")

        difficulty = raw_task.get("difficulty")
        if not isinstance(difficulty, str) or not difficulty:
            raise ValueError(f"Catalog task {task_id} has invalid difficulty.")

        defect_category = raw_task.get("defect_category")
        if not isinstance(defect_category, str) or not defect_category:
            raise ValueError(f"Catalog task {task_id} has invalid defect_category.")

        if manifest.difficulty != difficulty:
            raise ValueError(
                f"Difficulty mismatch for {task_id}: "
                f"{difficulty!r} != {manifest.difficulty!r}."
            )
        if manifest.defect_category != defect_category:
            raise ValueError(
                f"Defect category mismatch for {task_id}: "
                f"{defect_category!r} != {manifest.defect_category!r}."
            )

        seen_ids.add(task_id)
        seen_paths.add(manifest_path)
        tasks.append(
            CatalogTask(
                task_id=task_id,
                manifest_path=manifest_path,
                origin_type=origin_type,
                difficulty=difficulty,
                defect_category=defect_category,
            )
        )

    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"tasks", "manifest_paths"}
    }
    return BenchmarkCatalog(
        suite_id=suite_id,
        catalog_path=resolved_catalog,
        catalog_sha256=_catalog_digest(resolved_catalog),
        tasks=tuple(tasks),
        declared_task_count=task_count,
        metadata=metadata,
    )


def load_primary_research_catalog(
    project_root: Path,
    catalog_path: Path | str = "generated_benchmarks/research_suite.json",
) -> BenchmarkCatalog:
    """Load the canonical 53-task primary research benchmark."""
    project_root = project_root.resolve(strict=True)
    catalog = load_benchmark_catalog(project_root, catalog_path)
    if catalog.suite_id != "patchpilot-primary-research-benchmark":
        raise ValueError(
            "Expected the patchpilot-primary-research-benchmark catalog, "
            f"found {catalog.suite_id!r}."
        )
    if catalog.task_count != 53:
        raise ValueError(
            f"Primary research catalog must contain 53 tasks, "
            f"found {catalog.task_count}."
        )

    composition = catalog.composition()["origin_type"]
    expected = {"manual_challenge": 8, "mutmut": 45}
    if composition != expected:
        raise ValueError(
            "Primary research origin composition must be "
            f"{expected}, found {composition}."
        )

    if any(
        task.manifest_path.relative_to(project_root).parts[0] == "benchmarks"
        for task in catalog.tasks
    ):
        raise ValueError(
            "Sanity benchmarks must not appear in the primary research catalog."
        )
    return catalog


def discover_manifest_root(
    project_root: Path,
    manifest_root: Path | str,
) -> BenchmarkCatalog:
    """Build a validated custom catalog from a one-level task directory."""
    project_root = project_root.resolve(strict=True)
    root = Path(manifest_root)
    if not root.is_absolute():
        root = project_root / root
    root = root.resolve(strict=True)
    if not root.is_relative_to(project_root):
        raise ValueError(f"Manifest root escapes the project: {manifest_root}")

    manifests = sorted(root.glob("*/task.json"))
    if not manifests:
        raise FileNotFoundError(f"No benchmark manifests were found under {root}.")

    tasks: list[CatalogTask] = []
    seen_ids: set[str] = set()
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        if manifest.task_id in seen_ids:
            raise ValueError(f"Duplicate task_id: {manifest.task_id}")
        seen_ids.add(manifest.task_id)
        tasks.append(
            CatalogTask(
                task_id=manifest.task_id,
                manifest_path=manifest_path.resolve(strict=True),
                origin_type="custom",
                difficulty=manifest.difficulty,
                defect_category=manifest.defect_category,
            )
        )

    return BenchmarkCatalog(
        suite_id=f"custom:{root.relative_to(project_root).as_posix()}",
        catalog_path=None,
        catalog_sha256=None,
        tasks=tuple(tasks),
        declared_task_count=len(tasks),
        metadata={"manifest_root": root.relative_to(project_root).as_posix()},
    )


def select_catalog_tasks(
    catalog: BenchmarkCatalog,
    *,
    task_ids: list[str] | None = None,
    limit: int | None = None,
) -> tuple[CatalogTask, ...]:
    """Select tasks while preserving canonical catalog order."""
    selected = list(catalog.tasks)

    if task_ids:
        requested = list(dict.fromkeys(task_ids))
        if len(requested) != len(task_ids):
            raise ValueError("--task-id values must be unique.")
        requested_set = set(requested)
        available = {task.task_id for task in catalog.tasks}
        missing = sorted(requested_set - available)
        if missing:
            raise ValueError(
                "Requested task IDs are not in the selected catalog: "
                + ", ".join(missing)
            )
        selected = [task for task in catalog.tasks if task.task_id in requested_set]

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1.")
        selected = selected[:limit]

    if not selected:
        raise ValueError("Task selection is empty.")
    return tuple(selected)

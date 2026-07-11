"""Tests for mutation benchmark provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from patchpilot.benchmark.provenance import (
    MutationOperatorFamily,
    MutmutProvenance,
    classify_mutation,
    load_mutmut_provenance,
)


@pytest.mark.parametrize(
    ("diff", "expected"),
    [
        (
            "--- src/example.py\n"
            "+++ src/example.py\n"
            "@@ -1 +1 @@\n"
            "-return left + right\n"
            "+return left - right\n",
            MutationOperatorFamily.ARITHMETIC,
        ),
        (
            "--- src/example.py\n"
            "+++ src/example.py\n"
            "@@ -1 +1 @@\n"
            "-if value < 2:\n"
            "+if value <= 2:\n",
            MutationOperatorFamily.BOUNDARY,
        ),
        (
            "--- src/example.py\n"
            "+++ src/example.py\n"
            "@@ -1 +1 @@\n"
            "-value = abs(value)\n"
            "+value = None\n",
            MutationOperatorFamily.CONSTANT,
        ),
    ],
)
def test_classifies_mutmut_diff(
    diff: str,
    expected: MutationOperatorFamily,
) -> None:
    assert classify_mutation(diff) is expected


def make_payload(diff: str) -> dict[str, object]:
    return {
        "source_project": "example-project",
        "source_root_sha256": "a" * 64,
        "generator_commit": "bcb1c2d",
        "mutmut_version": "3.6.0",
        "generation_command": ["python", "scripts/generate_mutmut_benchmark.py"],
        "selection_rank": 1,
        "selected_from_total": 10,
        "selected_from_killed": 7,
        "mutant_name": "example.core.x_add__mutmut_1",
        "mutant_status": "killed",
        "mutated_function": "add",
        "source_file": "src/example/core.py",
        "source_line": 12,
        "operator_family": "arithmetic",
        "mutation_diff": diff,
        "mutation_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "test_command": ["python", "-m", "pytest", "-q"],
        "visible_tests_pass_on_clean": True,
        "hidden_tests_pass_on_clean": True,
        "visible_initial_failures": 1,
        "hidden_initial_failures": 1,
        "hidden_test_count": 2,
        "difficulty": "easy",
    }


def test_loads_valid_provenance(tmp_path: Path) -> None:
    diff = "--- a.py\n+++ a.py\n@@ -1 +1 @@\n-x + y\n+x - y\n"
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(make_payload(diff)), encoding="utf-8")

    record = load_mutmut_provenance(path)

    assert isinstance(record, MutmutProvenance)
    assert record.mutmut_version == "3.6.0"
    assert record.operator_family is MutationOperatorFamily.ARITHMETIC


def test_rejects_mismatched_diff_hash() -> None:
    diff = "--- a.py\n+++ a.py\n@@ -1 +1 @@\n-x + y\n+x - y\n"
    payload = make_payload(diff)
    payload["mutation_diff_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="does not match"):
        MutmutProvenance.model_validate(payload)

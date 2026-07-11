"""Unit tests for deterministic Mutmut benchmark generation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from patchpilot.benchmark.provenance import MutationOperatorFamily

SCRIPT_PATH = Path("scripts/generate_mutmut_benchmark.py")


@pytest.fixture(scope="module")
def generator_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "patchpilot_mutmut_generator_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parses_show_output(generator_module: ModuleType) -> None:
    mutant = generator_module.MutantResult(
        name="example.core.x_add__mutmut_1",
        status="killed",
    )
    output = (
        "# example.core.x_add__mutmut_1: killed\n"
        "--- src/example/core.py\n"
        "+++ src/example/core.py\n"
        "@@ -12,1 +12,1 @@\n"
        "-    return left + right\n"
        "+    return left - right\n"
    )

    descriptor = generator_module.parse_mutmut_show(mutant, output)

    assert descriptor.function == "add"
    assert descriptor.source_file == "src/example/core.py"
    assert descriptor.source_line == 12
    assert descriptor.operator_family is MutationOperatorFamily.ARITHMETIC


def test_selection_is_deterministic_and_function_balanced(
    generator_module: ModuleType,
) -> None:
    descriptor = generator_module.MutantDescriptor
    items = [
        descriptor(
            name="pkg.core.x_beta__mutmut_2",
            status="killed",
            function="beta",
            source_file="src/core.py",
            source_line=20,
            operator_family=MutationOperatorFamily.CONSTANT,
            mutation_diff="--- a\n+++ a\n@@ -1 +1 @@\n-x = 1\n+x = 2\n",
        ),
        descriptor(
            name="pkg.core.x_alpha__mutmut_2",
            status="killed",
            function="alpha",
            source_file="src/core.py",
            source_line=10,
            operator_family=MutationOperatorFamily.BOUNDARY,
            mutation_diff="--- a\n+++ a\n@@ -1 +1 @@\n-x < 1\n+x <= 1\n",
        ),
        descriptor(
            name="pkg.core.x_alpha__mutmut_1",
            status="killed",
            function="alpha",
            source_file="src/core.py",
            source_line=9,
            operator_family=MutationOperatorFamily.ARITHMETIC,
            mutation_diff="--- a\n+++ a\n@@ -1 +1 @@\n-x + 1\n+x - 1\n",
        ),
    ]

    selected = generator_module.select_diverse_mutants(
        list(reversed(items)),
        max_tasks=3,
        max_per_function=2,
    )

    assert [item.name for item in selected] == [
        "pkg.core.x_alpha__mutmut_1",
        "pkg.core.x_beta__mutmut_2",
        "pkg.core.x_alpha__mutmut_2",
    ]


def test_pytest_count_handles_mixed_outcomes(
    generator_module: ModuleType,
) -> None:
    assert (
        generator_module.pytest_test_count("2 failed, 3 passed, 1 skipped in 0.02s")
        == 6
    )

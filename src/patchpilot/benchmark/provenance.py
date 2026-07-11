"""Validated provenance records for mutation-generated benchmarks."""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MutationOperatorFamily(StrEnum):
    """Normalized mutation families used for dataset balancing."""

    ARITHMETIC = "arithmetic"
    BOOLEAN = "boolean"
    BOUNDARY = "boundary"
    COLLECTION = "collection"
    COMPARISON = "comparison"
    CONSTANT = "constant"
    EXCEPTION = "exception"
    RETURN_VALUE = "return_value"
    STATEMENT = "statement"
    UNKNOWN = "unknown"


class MutmutProvenance(BaseModel):
    """Complete reproducibility metadata for one exported killed mutant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    benchmark_kind: Literal["mutmut"] = "mutmut"
    source_project: str = Field(min_length=2, max_length=100)
    source_root_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_commit: str = Field(min_length=1, max_length=100)
    mutmut_version: str = Field(min_length=1, max_length=100)
    generation_command: list[str] = Field(min_length=1)
    selection_rank: int = Field(ge=1)
    selected_from_total: int = Field(ge=1)
    selected_from_killed: int = Field(ge=1)
    mutant_name: str = Field(min_length=3, max_length=500)
    mutant_status: Literal["killed"]
    mutated_function: str = Field(min_length=1, max_length=300)
    source_file: str = Field(min_length=1, max_length=500)
    source_line: int | None = Field(default=None, ge=1)
    operator_family: MutationOperatorFamily
    mutation_diff: str = Field(min_length=1)
    mutation_diff_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    test_command: list[str] = Field(min_length=3)
    visible_tests_pass_on_clean: bool
    hidden_tests_pass_on_clean: bool
    visible_initial_failures: int = Field(ge=1)
    hidden_initial_failures: int = Field(ge=1)
    hidden_test_count: int = Field(ge=1)
    difficulty: Literal["easy", "medium", "hard"]

    @model_validator(mode="after")
    def validate_mutation_diff_hash(self) -> Self:
        """Require the stored digest to match the exact mutation diff."""
        digest = hashlib.sha256(self.mutation_diff.encode("utf-8")).hexdigest()
        if digest != self.mutation_diff_sha256:
            raise ValueError("mutation_diff_sha256 does not match mutation_diff.")
        if self.selected_from_killed > self.selected_from_total:
            raise ValueError("selected_from_killed cannot exceed selected_from_total.")
        if self.selection_rank > self.selected_from_killed:
            raise ValueError("selection_rank cannot exceed selected_from_killed.")
        return self


def load_mutmut_provenance(path: Path) -> MutmutProvenance:
    """Load one validated mutation provenance record."""
    if not path.is_file():
        raise FileNotFoundError(f"Provenance file does not exist: {path}")
    return MutmutProvenance.model_validate_json(path.read_text(encoding="utf-8"))


def mutation_diff_sha256(diff: str) -> str:
    """Return the SHA-256 digest of an exact mutation diff."""
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def _changed_lines(diff: str) -> tuple[list[str], list[str]]:
    removed: list[str] = []
    added: list[str] = []
    for line in diff.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            removed.append(line[1:].strip())
        elif line.startswith("+"):
            added.append(line[1:].strip())
    return removed, added


def classify_mutation(diff: str) -> MutationOperatorFamily:
    """Classify a Mutmut unified diff into a stable operator family."""
    removed, added = _changed_lines(diff)
    before = "\n".join(removed)
    after = "\n".join(added)
    combined = f"{before}\n{after}"

    boundary_pairs = (
        ("<", "<="),
        (">", ">="),
        ("<=", "<"),
        (">=", ">"),
    )
    if any(left in before and right in after for left, right in boundary_pairs):
        return MutationOperatorFamily.BOUNDARY

    if re.search(r"\b(?:raise|except)\b", combined):
        return MutationOperatorFamily.EXCEPTION

    if re.search(r"\b(?:and|or|not)\b", combined):
        return MutationOperatorFamily.BOOLEAN

    if re.search(r"(?:==|!=|<=|>=|<|>)", combined):
        return MutationOperatorFamily.COMPARISON

    arithmetic_pattern = r"(?:\+|-|\*|/|//|%|\*\*)"
    if re.search(arithmetic_pattern, before) and re.search(
        arithmetic_pattern,
        after,
    ):
        return MutationOperatorFamily.ARITHMETIC

    if re.search(
        r"\b(?:append|extend|insert|pop|remove|sorted|list|set|dict|tuple)\b",
        combined,
    ):
        return MutationOperatorFamily.COLLECTION

    if any(
        token in combined for token in ("None", "True", "False", '""', "''")
    ) or re.search(r"(?<![A-Za-z_])\d+(?![A-Za-z_])", combined):
        return MutationOperatorFamily.CONSTANT

    if (
        removed
        and added
        and all(line.startswith("return") for line in [*removed, *added])
    ):
        return MutationOperatorFamily.RETURN_VALUE

    if removed or added:
        return MutationOperatorFamily.STATEMENT
    return MutationOperatorFamily.UNKNOWN

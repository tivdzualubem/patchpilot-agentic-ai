#!/usr/bin/env bash
set -euo pipefail

ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
    pwd
)"
RAW="$ROOT/results/final-research-evaluation-6524a37/raw/runs.csv"
PUBLISHED="$ROOT/results/final-research-evaluation-6524a37"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [output-directory]" >&2
    exit 2
fi

if [[ $# -eq 1 ]]; then
    OUTPUT="$1"
    mkdir -p "$OUTPUT"
    CLEANUP=0
else
    OUTPUT="$(mktemp -d /tmp/patchpilot-reproduced-analysis.XXXXXX)"
    CLEANUP=1
fi

cleanup() {
    if [[ "$CLEANUP" -eq 1 ]]; then
        rm -rf "$OUTPUT"
    fi
}
trap cleanup EXIT

cd "$ROOT"

python scripts/reproduce_final_analysis.py \
    --runs-csv "$RAW" \
    --output-dir "$OUTPUT"

python - "$OUTPUT" "$PUBLISHED" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

generated = Path(sys.argv[1])
published = Path(sys.argv[2])

tables = {
    "condition_summary.csv": ["condition"],
    "failure_taxonomy.csv": [
        "condition",
        "terminal_failure_category",
    ],
    "paired_latency_wilcoxon_holm.csv": [
        "condition_a",
        "condition_b",
    ],
    "pairwise_exact_mcnemar_holm.csv": [
        "condition_a",
        "condition_b",
    ],
    "parse_failure_association.csv": ["condition"],
    "stratified_results.csv": [
        "grouping",
        "level",
        "condition",
    ],
    "task_level_hidden_outcomes.csv": ["task_id"],
    "visible_hidden_disagreements.csv": [
        "task_id",
        "condition",
    ],
}

for filename, keys in tables.items():
    actual = pd.read_csv(generated / filename)
    expected = pd.read_csv(published / filename)

    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)

    assert list(actual.columns) == list(expected.columns), filename
    assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=1e-9,
        atol=1e-10,
        obj=filename,
    )
    print(f"REPRODUCED_TABLE={filename}")

analysis = json.loads(
    (generated / "final_statistical_analysis.json").read_text(
        encoding="utf-8"
    )
)
assert analysis["total_runs"] == 212
assert analysis["tasks"] == 53
assert analysis["conditions"] == 4
assert abs(analysis["cochran_q"] - 11.0) < 1e-12
assert abs(
    analysis["cochran_q_p"] - 0.011725875578
) < 1e-12
assert analysis["scope"][
    "runtime_verification_ablation_executed"
] is False

expected_figures = {
    "benchmark_composition.png",
    "hidden_verified_rates_ci95.png",
    "mean_latency_by_condition.png",
    "mutmut_hidden_verified_rates.png",
    "parse_failure_success_rates.png",
}
actual_figures = {
    path.name
    for path in (generated / "figures").glob("*.png")
    if path.stat().st_size > 0
}
assert actual_figures == expected_figures

print("ALL_PUBLISHED_TABLES_REPRODUCED=1")
print("ALL_FINAL_ANALYSIS_FIGURES_REGENERATED=1")
print("VERIFICATION_ABLATION_SCOPE_EXPLICIT=1")
PY

echo "ONE_COMMAND_REPRODUCTION_VERIFIED=1"
echo "REPRODUCTION_OUTPUT=$OUTPUT"

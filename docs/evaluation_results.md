# PatchPilot Evaluation Results

This document records the current full-agent evaluation result for PatchPilot.

## Evaluation Setup

- Benchmark: PatchPilot-Bench v0
- Tasks: 12 local mutation-seeded Python repair tasks
- Condition: `full-agent-live-qwen`
- Model backend: Ollama / Qwen2.5-Coder
- Result directory: `artifacts/evaluation/20260704-091749`

The benchmark tasks cover arithmetic, geometry, list processing, number handling, statistics, and string-processing defects.

## Summary

| Metric | Value |
| --- | ---: |
| Runs | 12 |
| Successful repairs | 12 |
| Repair rate | 100.0% |
| Full-suite passes | 12 |
| Full-suite pass rate | 100.0% |
| Invalid patch runs | 0 |
| Invalid patch rate | 0.0% |
| Failures | 0 |
| Escalations | 0 |
| Budget exhaustions | 0 |
| Mean steps | 5.75 |
| Mean tool calls | 5.75 |
| Mean patch attempts | 1.25 |
| Mean elapsed seconds | 40.14 |

## Per-Task Results

| Task | Status | Success | Full Suite Passed | Steps | Tool Calls | Patch Attempts | Seconds |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| calculator-001 | succeeded |  | True | 5 | 5 | 1 | 27.01 |
| calculator-002 | succeeded |  | True | 5 | 5 | 1 | 28.09 |
| calculator-003 | succeeded |  | True | 5 | 5 | 1 | 32.77 |
| calculator-004 | succeeded |  | True | 5 | 5 | 1 | 27.32 |
| geometry-001 | succeeded |  | True | 5 | 5 | 1 | 38.00 |
| geometry-002 | succeeded |  | True | 8 | 8 | 2 | 77.40 |
| lists-001 | succeeded |  | True | 5 | 5 | 1 | 29.10 |
| lists-002 | succeeded |  | True | 5 | 5 | 1 | 25.55 |
| numbers-001 | succeeded |  | True | 5 | 5 | 1 | 32.75 |
| stats-001 | succeeded |  | True | 8 | 8 | 2 | 72.84 |
| strings-001 | succeeded |  | True | 5 | 5 | 1 | 26.24 |
| strings-002 | succeeded |  | True | 8 | 8 | 2 | 64.60 |

## Interpretation

The full PatchPilot agent repaired all 12 benchmark tasks in this run. Every successful repair required executable verification through the full pytest suite before the agent could finish. No run ended in escalation, failure, invalid patch state, or budget exhaustion.

This result should be reported as the full-agent condition. Baseline and ablation conditions are still required before making comparative claims.

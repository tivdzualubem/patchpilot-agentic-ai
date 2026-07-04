# PatchPilot Evaluation Comparison

This document compares the full PatchPilot agent against two reduced conditions.

## Conditions

| Condition | Description |
| --- | --- |
| `full-agent-live-qwen` | Full bounded PatchPilot workflow with verification-driven retry budget up to 3 patch attempts |
| `one-shot-live-qwen` | One-shot baseline: reproduce failure, inspect once, apply one patch, verify once, then stop |
| `no-retry-live-qwen` | Reduced-budget ablation with only 1 patch attempt and tighter step/tool budget |

## Summary

| Metric | Full Agent | One-Shot Baseline | No-Retry Ablation |
| --- | ---: | ---: | ---: |
| Runs | 12 | 12 | 12 |
| Successful repairs | 12 | 8 | 8 |
| Repair rate | 100.0% | 66.7% | 66.7% |
| Full-suite pass rate | 100.0% | 66.7% | 66.7% |
| Invalid patch rate | 0.0% | 0.0% | 0.0% |
| Budget exhaustions | 0 | 0 | 4 |
| Escalations | 0 | 4 | 0 |
| Mean steps | 5.75 | 5.00 | 5.33 |
| Mean tool calls | 5.75 | 5.00 | 5.33 |
| Mean patch attempts | 1.25 | 1.00 | 1.00 |
| Mean elapsed seconds | 40.14 | 30.59 | 37.07 |

## Result Directories

- Full agent: `artifacts/evaluation/20260704-091749`
- One-shot baseline: `artifacts/evaluation/20260704-135312`
- No-retry ablation: `artifacts/evaluation/20260704-094853`

## Interpretation

The full PatchPilot agent repaired all 12 PatchPilot-Bench v0 tasks. The one-shot baseline repaired 8 of 12 tasks and escalated the remaining 4 after its single patch failed verification. The no-retry ablation also repaired 8 of 12 tasks, but its failures appeared as budget exhaustion under the tighter execution limits.

These results support the project claim that verified iterative repair improves reliability over a single-pass repair attempt. The full agent succeeded on tasks that required a second patch attempt, while the one-shot baseline stopped after the first failed verification.

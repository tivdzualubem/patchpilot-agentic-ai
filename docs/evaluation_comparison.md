# PatchPilot Evaluation Comparison

This document compares the full PatchPilot agent against the no-retry ablation.

## Conditions

| Condition | Description |
| --- | --- |
| `full-agent-live-qwen` | Full bounded PatchPilot workflow with retry budget up to 3 patch attempts |
| `no-retry-live-qwen` | Ablation with only 1 patch attempt and tighter step/tool budget |

## Summary

| Metric | Full Agent | No-Retry Ablation |
| --- | ---: | ---: |
| Runs | 12 | 12 |
| Successful repairs | 12 | 8 |
| Repair rate | 100.0% | 66.7% |
| Full-suite pass rate | 100.0% | 66.7% |
| Invalid patch rate | 0.0% | 0.0% |
| Budget exhaustions | 0 | 4 |
| Escalations | 0 | 0 |
| Mean steps | 5.75 | 5.33 |
| Mean tool calls | 5.75 | 5.33 |
| Mean patch attempts | 1.25 | 1.00 |
| Mean elapsed seconds | 40.14 | 37.07 |

## Result Directories

- Full agent: `artifacts/evaluation/20260704-091749`
- No-retry ablation: `artifacts/evaluation/20260704-094853`

## Interpretation

The full PatchPilot agent repaired all 12 PatchPilot-Bench v0 tasks. The no-retry ablation repaired 8 of 12 tasks and exhausted its budget on 4 tasks.

This shows that bounded retry capacity and continued verification-driven repair are important for tasks that need more than one patch attempt. The comparison supports the project claim that software repair benefits from an agentic loop rather than a single-pass repair attempt.

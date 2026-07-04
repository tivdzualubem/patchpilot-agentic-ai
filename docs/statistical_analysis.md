# PatchPilot Statistical Analysis

This document records the reproducible statistical comparison between `full-agent-live-qwen` and `no-retry-live-qwen` on PatchPilot-Bench v0.

## Inputs

- Full-agent runs CSV: `artifacts/evaluation/20260704-091749/runs.csv`
- Baseline/ablation runs CSV: `artifacts/evaluation/20260704-094853/runs.csv`
- Paired tasks: 12

## Repair Success

| Condition | Successes | Repair Rate |
| --- | ---: | ---: |
| `full-agent-live-qwen` | 12/12 | 100.0% |
| `no-retry-live-qwen` | 8/12 | 66.7% |

## Paired Success Table

| Outcome | Count |
| --- | ---: |
| Both succeeded | 8 |
| Full agent only succeeded | 4 |
| Baseline/ablation only succeeded | 0 |
| Both failed | 0 |

## Exact McNemar Test

Because both conditions ran on the same benchmark tasks, repair success is compared as paired binary outcomes.

| Test | Value |
| --- | ---: |
| Discordant pairs | 4 |
| Exact McNemar two-sided p-value | 0.1250 |

## Interpretation

The full agent repaired 12/12 tasks, while the baseline/ablation repaired 8/12 tasks.
The paired effect size is 4 tasks repaired only by the full agent versus 0 tasks repaired only by the baseline/ablation.

Because the benchmark currently contains 12 tasks, p-values should be interpreted cautiously.
The strongest evidence is the paired success difference and the fact that the full agent succeeds on tasks where the reduced condition exhausts its budget.

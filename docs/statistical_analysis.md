# PatchPilot Statistical Analysis

This document records the statistical comparison between the full PatchPilot agent and the no-retry ablation on PatchPilot-Bench v0.

## Paired Success Table

| Outcome | Count |
| --- | ---: |
| Both succeeded | 8 |
| Full agent only succeeded | 4 |
| No-retry ablation only succeeded | 0 |
| Both failed | 0 |

## Exact McNemar Test

Because each condition ran on the same benchmark tasks, repair success can be compared as paired binary outcomes. The discordant task count is 4: the full agent succeeded where the ablation failed on 4 tasks, while the ablation succeeded where the full agent failed on 0 tasks.

| Test | Value |
| --- | ---: |
| Exact McNemar two-sided p-value | 0.1250 |

## Interpretation

The full agent repaired 12/12 tasks, while the no-retry ablation repaired 8/12 tasks. The observed effect favours the full agent, especially on tasks requiring more than one patch attempt.

Because the benchmark currently contains 12 tasks, the statistical test should be interpreted cautiously. The main evidence is the paired effect size: four tasks were repaired only by the full agent, and none were repaired only by the ablation. This supports the project claim that bounded retry and verification-driven repair improve robustness over a one-pass variant.

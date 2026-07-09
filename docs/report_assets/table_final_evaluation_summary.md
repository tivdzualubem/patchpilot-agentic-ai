| benchmark | condition | runs | successes | repair_rate | full_suite_pass_rate | invalid_patch_rate | budget_exhaustions | escalations | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Controlled | Full agent | 12 | 12 | 100.0% | 100.0% | 0.0% |  |  | Internal controlled benchmark; saturated. |
| Controlled | One-shot | 12 | 12 | 100.0% | 100.0% | 0.0% |  |  | Internal controlled benchmark; saturated. |
| Controlled | No-retry | 12 | 12 | 100.0% | 100.0% | 0.0% |  |  | Internal controlled benchmark; saturated. |
| Mutmut | Full agent | 20 | 8 | 40.0% | 40.0% | 0.0% | 5 | 7 | Primary real mutmut-generated benchmark. |
| Mutmut | One-shot | 20 | 6 | 30.0% | 30.0% | 0.0% | 0 | 14 | Ablation baseline. |
| Mutmut | No-retry | 20 | 5 | 25.0% | 25.0% | 0.0% | 12 | 3 | Ablation baseline. |
| QuixBugs | Full agent | 8 | 3 | 37.5% | 37.5% | 0.0% | 3 | 2 | External smoke benchmark. |
| SWE-bench Lite | Official harness smoke | 1 | 0 | 0.0% | 0.0% |  |  |  | Feasibility attempt; local WSL/Docker I/O blocked full run. |

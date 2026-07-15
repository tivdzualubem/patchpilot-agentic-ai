# PatchPilot Final Statistical Analysis

## Evaluation integrity

- Experiment: `final-research-evaluation-6524a37-20260714-090253`
- Commit: `6524a37d802fc83faa9db0111facaf491da4e618`
- Complete matrix: **212/212 runs** across **53 tasks × 4 conditions**
- Primary endpoint: **hidden-verified repair success**
- Benchmark composition: **45 Mutmut tasks + 8 manual hard challenges**
- Model: `qwen2.5-coder:3b`, temperature 0, seed 42

The primary endpoint is hidden verification because it is objective, execution-based, and not exposed to the agent during repair.

## Main outcome

| Condition | Hidden verified | 95% Wilson CI | Visible | Mean time | Median time | Parse-failure runs |
|---|---:|---:|---:|---:|---:|---:|
| One-shot | 37/53 (69.8%) | 56.5%–80.5% | 38/53 (71.7%) | 86.5s | 74.4s | 9 |
| Fixed workflow | 37/53 (69.8%) | 56.5%–80.5% | 38/53 (71.7%) | 105.7s | 91.3s | 15 |
| Tool agent, no reflection | 31/53 (58.5%) | 45.1%–70.7% | 33/53 (62.3%) | 350.1s | 320.2s | 21 |
| Full reflective agent | 34/53 (64.2%) | 50.7%–75.7% | 36/53 (67.9%) | 249.6s | 238.7s | 16 |

### Global paired test

Cochran's Q across the four matched conditions:

- Q(3) = **11.000**
- p = **0.0117**

The global null that all four conditions have the same hidden-verified success probability is rejected at α = 0.05. This establishes heterogeneity somewhere in the four-condition set, but it does not identify a specific superior pair.

### Exact paired McNemar tests with Holm correction

| Comparison (A vs B) | A only | B only | Risk difference | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|
| One-shot vs Fixed workflow | 0 | 0 | +0.0 pp | 1.0000 | 1.0000 |
| One-shot vs Tool agent, no reflection | 6 | 0 | +11.3 pp | 0.0312 | 0.1875 |
| One-shot vs Full reflective agent | 3 | 0 | +5.7 pp | 0.2500 | 1.0000 |
| Fixed workflow vs Tool agent, no reflection | 6 | 0 | +11.3 pp | 0.0312 | 0.1875 |
| Fixed workflow vs Full reflective agent | 3 | 0 | +5.7 pp | 0.2500 | 1.0000 |
| Tool agent, no reflection vs Full reflective agent | 3 | 6 | -5.7 pp | 0.5078 | 1.0000 |

No pairwise comparison remains significant after Holm correction across all six planned comparisons. The strongest unadjusted contrast is the baseline versus the non-reflective tool agent: six tasks were solved only by the baseline and none only by the non-reflective agent (exact p = 0.0313), but Holm-adjusted p = 0.1875.

## Stratified performance

### By benchmark origin

| Condition | Mutmut (n=45) | Manual hard (n=8) |
|---|---:|---:|
| One-shot | 37/45 (82.2%) | 0/8 (0.0%) |
| Fixed workflow | 37/45 (82.2%) | 0/8 (0.0%) |
| Tool agent, no reflection | 31/45 (68.9%) | 0/8 (0.0%) |
| Full reflective agent | 34/45 (75.6%) | 0/8 (0.0%) |

All four conditions scored **0/8 on the manual hard challenges**. Therefore, differences in the overall score come entirely from the 45 Mutmut tasks. This is a clear capability ceiling for the current 3B local model and scaffold on multi-line, stateful, and protocol-level defects.

### By difficulty

| Condition | Easy (n=33) | Medium (n=12) | Hard (n=8) |
|---|---:|---:|---:|
| One-shot | 26/33 (78.8%) | 11/12 (91.7%) | 0/8 (0.0%) |
| Fixed workflow | 26/33 (78.8%) | 11/12 (91.7%) | 0/8 (0.0%) |
| Tool agent, no reflection | 21/33 (63.6%) | 10/12 (83.3%) | 0/8 (0.0%) |
| Full reflective agent | 25/33 (75.8%) | 9/12 (75.0%) | 0/8 (0.0%) |

The unusual pattern in which medium tasks outperform easy tasks for the two baselines reflects the selected task composition, not a general claim that medium defects are intrinsically easier.

## Efficiency and trajectory quality

Mean latency:

- One-shot: **86.5s**
- Fixed workflow: **105.7s**
- Full reflective agent: **249.6s**
- Tool agent without reflection: **350.1s**

Friedman's paired test on task-level latency:

- χ²(3) = **143.785**
- p = **5.77e-31**

All six paired Wilcoxon latency comparisons remain significant after Holm correction. The ordering is consistent: one-shot is fastest, followed by fixed workflow, full reflective agent, then the non-reflective tool agent.

The fixed workflow produced exactly the same hidden-success vector as one-shot on all 53 tasks, while taking about **1.22×** as long. In this evaluation, retry capability added no observed repair benefit because no baseline task was recovered by the fixed workflow after a one-shot failure.

The agent-runtime defect addressed before this evaluation is resolved at the trajectory level:

- Budget exhaustions: **0**
- Invalid-patch runs: **0**
- No-progress-loop runs: **0**

## Parsing remains the dominant failure mode

A run with at least one decision parse failure almost never achieved hidden-verified success:

| Condition | Success with parse failure | Success without parse failure |
|---|---:|---:|
| One-shot | 11.1% | 81.8% |
| Fixed workflow | 6.7% | 94.7% |
| Tool agent, no reflection | 4.8% | 93.8% |
| Full reflective agent | 0.0% | 91.9% |

This association is descriptive rather than causal because difficulty and condition affect both parsing and success. Nevertheless, the trajectory data clearly localize the remaining bottleneck: model output reliability, especially on hard tasks, rather than tool-loop convergence or unsafe patching.

## Hidden verification findings

There were six visible/hidden disagreements:

- `clamp-mutmut-10` passed visible tests but failed hidden verification in all four conditions.
- `parse-key-values-mutmut-19` disagreed only for the non-reflective agent.
- `parse-csv-line-mutmut-1` disagreed only for the reflective agent.

This supports keeping hidden tests as the primary endpoint. Visible success alone would overstate system performance.

## Reflection ablation limitation

The reflective condition recorded:

- Reflection events: **0**
- Hypothesis revisions: **0**

Therefore, the experiment does **not** provide direct evidence that reflection caused an improvement. The reflective policy label identifies the configured condition, but the measured trajectories did not contain completed reflection/revision cycles. The observed 34 versus 31 hidden successes should be described as a difference between configured agent conditions, not as a demonstrated causal effect of reflection.

## Defensible conclusions

1. **The repaired agent runtime is operational.** The final evaluation completed all 212 runs without budget exhaustion, invalid-patch failures, or no-progress loops.
2. **Simple scaffolds were strongest under this local model.** One-shot and fixed workflow tied at 37/53 hidden-verified repairs and outperformed both tool-agent variants descriptively.
3. **Extra agency increased cost without a statistically confirmed outcome gain.** Agentic conditions were 2.9–4.0× slower than one-shot on mean latency.
4. **The reflective configuration improved over the non-reflective configuration descriptively, but not significantly.** It gained six tasks and lost three in their direct paired comparison; exact McNemar p = 0.5078.
5. **The hard benchmark remains unsolved.** All conditions failed all eight manual challenges, identifying the main capability gap for future work.
6. **Claims must stay system-specific.** These results characterize PatchPilot with `qwen2.5-coder:3b`, its prompts, tools, budgets, benchmark selection, and hidden tests; they are not a general verdict on reflection or agentic repair.

## Report and UI implications

The final report and dashboard should:

- use hidden-verified repair rate as the headline metric;
- show visible/hidden disagreement explicitly;
- separate Mutmut and manual-hard results;
- report confidence intervals and paired tests;
- display latency, model calls, parse failures, and trajectory safety metrics;
- state that reflection events were not observed;
- avoid claiming statistically significant pairwise superiority;
- present the 0/8 hard-challenge result as a limitation and future-work target.

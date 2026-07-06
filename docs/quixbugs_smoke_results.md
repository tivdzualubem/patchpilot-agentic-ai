# QuixBugs External Smoke Results

PatchPilot was also evaluated on a supplementary public benchmark smoke test using QuixBugs Python tasks.

## Setup

- Benchmark source: QuixBugs Python programs
- Wrapper script: `scripts/run_quixbugs_smoke.py`
- Task count: 8 fixed tasks
- Model: `qwen2.5-coder:1.5b`
- Result directory: `artifacts/external/quixbugs/20260705-185529`
- Correct reference implementations were not used by the repair agent.
- Tests and QuixBugs reference solutions were forbidden paths.

## Summary

| Metric | Value |
| --- | ---: |
| Runs | 8 |
| Successful repairs | 3 |
| Repair rate | 37.5% |
| Full-suite passes | 3 |
| Full-suite pass rate | 37.5% |
| Invalid patch runs | 0 |
| Invalid patch rate | 0.0% |
| Escalations | 2 |
| Budget exhaustions | 3 |
| Mean steps | 6.38 |
| Mean tool calls | 6.38 |
| Mean patch attempts | 1.13 |
| Mean elapsed seconds | 74.34 |

## Interpretation

This external smoke test is not the primary benchmark. It is used as a generalization check beyond PatchPilot-Bench v0.

The result shows that PatchPilot can repair some unseen public benchmark defects while preserving its safety constraints. It also exposes a current capability limit: the patch synthesizer is intentionally conservative and currently targets one-line source replacements. Multi-line algorithmic repairs remain future work.

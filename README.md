---
title: PatchPilot Agentic AI
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# PatchPilot: A Safety-Constrained Tool-Using Agent for Python Debugging and Repair

PatchPilot is a bounded agentic system for repairing small Python repositories with executable pytest suites. A language-model policy proposes structured actions, while the runtime validates tool arguments, restricts editable paths, protects tests, applies controlled patches, records traces, and accepts a repair only after executable verification.

This repository contains the implementation, benchmark corpus, four-condition evaluation, statistical analysis, live Streamlit demonstration, and final research report.

## Research snapshot

The final paired evaluation used **53 tasks** under **four conditions**, producing **212 completed runs** with `qwen2.5-coder:3b`.

| Condition | Hidden-verified repairs | Rate | Mean runtime |
| --- | ---: | ---: | ---: |
| One-shot | 37/53 | 69.8% | 86.53 s |
| Fixed workflow | 37/53 | 69.8% | 105.74 s |
| Full reflective agent | 34/53 | 64.2% | 249.64 s |
| Tool agent without reflection | 31/53 | 58.5% | 350.12 s |

The global paired comparison produced Cochran's \(Q(3)=11.00\), \(p=0.0117\). No pairwise exact McNemar comparison remained significant after Holm correction. The configured reflective condition recorded zero completed reflection events in the final traces, so the results do **not** establish a causal benefit from reflection.

All four conditions failed the eight manually authored hard challenges. Results therefore describe this benchmark and configuration rather than general software-repair capability.

Detailed evidence:

- [Final research report](docs/PatchPilot_Final_Report.pdf)
- [Reproducible report source](docs/final_report/)
- [Final statistical analysis](results/final-research-evaluation-6524a37/PatchPilot_Final_Statistical_Analysis.md)
- [Machine-readable final results](results/final-research-evaluation-6524a37/)
- [Final evaluation figures](docs/report_assets/final_evaluation/)

## System architecture

```text
Task manifest
  -> isolated workspace
  -> structured policy action
  -> schema and path validation
  -> bounded tool execution
  -> observation and trace update
  -> patch verification
  -> verified success or bounded termination
```

The model does not receive unrestricted shell or repository access. PatchPilot exposes a compact tool interface for:

- running tests;
- searching and reading source files;
- applying and restoring source patches;
- recording completion only after verification.

Task manifests define editable and forbidden paths. The executor validates tool-specific arguments, blocks protected-file edits, enforces budgets, and records machine-readable run traces.

## Repository layout

```text
benchmark_seeds/                         Seed projects used for generated tasks
benchmarks/                              Controlled benchmark tasks
challenge_benchmarks/                    Manually authored hard challenges
generated_benchmarks/                    Generated tasks and research catalog
generated_benchmarks/research_suite.json Canonical 53-task research suite
demo/                                    Streamlit live-repair demonstration
docs/PatchPilot_Final_Report.pdf         Final polished research report
docs/final_report/                       LaTeX source, assets, and build script
docs/report_assets/                      Committed evaluation and UI figures
results/final-research-evaluation-6524a37/
                                         Final statistics and task-level outcomes
scripts/                                 Demo, benchmark, evaluation, and analysis tools
src/patchpilot/                          PatchPilot package
tests/                                   Unit and integration tests
```

## Requirements

- Python 3.12
- Git
- Ollama for live model-backed runs
- `qwen2.5-coder:3b`
- Streamlit dependencies for the graphical demo
- Docker only for the optional containerized frontend

Install the validated local model:

```bash
ollama pull qwen2.5-coder:3b
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,analysis]"
python -m pip install -r demo/requirements-demo.txt
```

## Verified quality gate

The final repository gate completed with:

```text
274 tests passed
Ruff passed
mypy passed on 32 source files
git diff --check passed
```

Run the same code-quality checks:

```bash
python -m ruff check .
python -m pytest -q
python -m mypy src
git diff --check
```

## Live repair demonstration

Start Ollama and confirm that the model is available:

```bash
ollama list
```

Launch the Streamlit application:

```bash
streamlit run demo/streamlit_app.py \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.port 8501
```

Open `http://localhost:8501`.

The demonstration lets a user:

1. select a prepared defective task;
2. inspect its source and regression tests;
3. run PatchPilot with the local model;
4. review executable verification, the agent trajectory, and the generated patch.

The hosted frontend is available at:

```text
https://lubem-patchpilot-agentic-ai.hf.space/
```

A local Ollama environment remains the reliable path for live repair execution.

### Command-line demo

```bash
python scripts/run_demo_task.py \
  --task-id calculator-001 \
  --manifest-path benchmarks/calculator-001/task.json \
  --model qwen2.5-coder:3b
```

A verified successful result includes:

```text
status: succeeded
full_suite_passed: true
```

Runtime workspaces and raw traces are written under `artifacts/`, which is intentionally ignored by Git.

## Research corpus

The canonical suite is defined by:

```text
generated_benchmarks/research_suite.json
```

It contains:

- 45 Mutmut-derived tasks;
- 8 manually authored hard challenges;
- 53 primary research tasks in total.

Twelve controlled sanity tasks are maintained separately and are not part of the final 53-task primary comparison.

Validate the corpus:

```bash
python scripts/validate_research_benchmarks.py
python scripts/validate_mutmut_research_suite.py
```

## Reproducing the four-condition evaluation

The evaluation runner defaults to the canonical research catalog and `qwen2.5-coder:3b`.

```bash
python scripts/run_evaluation.py \
  --condition all \
  --model qwen2.5-coder:3b \
  --catalog generated_benchmarks/research_suite.json \
  --output-root artifacts/evaluation \
  --experiment-id reproduction-3b
```

The four primary conditions are:

1. `one-shot`
2. `fixed-workflow`
3. `tool-agent-no-reflection`
4. `full-reflective-agent`

The runner writes an experiment configuration, condition-level CSV and JSON summaries, a combined `runs.csv`, and trace artifacts beneath the selected output root.

The published final experiment is:

```text
final-research-evaluation-6524a37-20260714-090253
evaluated commit: 6524a37d802fc83faa9db0111facaf491da4e618
```

## Reproducing the statistical analysis

After an evaluation has produced a combined `runs.csv`:

```bash
python scripts/analyze_final_evaluation.py \
  artifacts/evaluation/<EXPERIMENT_ID>/runs.csv \
  --output-dir results/reproduced-analysis
```

The analysis produces condition summaries, task-level outcomes, failure taxonomy, exact paired McNemar tests with Holm correction, latency comparisons, and a machine-readable JSON summary.

The committed final analysis is under:

```text
results/final-research-evaluation-6524a37/
```

## Evaluation interpretation

The final results support several bounded observations:

- One-shot and fixed workflow had the highest hidden-verified success rate in this experiment.
- The reflective configuration outperformed the no-reflection tool agent descriptively, but the trace data recorded no completed reflection events.
- Agentic conditions required substantially more runtime than the baselines.
- Hidden verification exposed a small number of visible/hidden disagreements.
- The manually authored hard subset remained unsolved by every condition.

These findings do not establish general superiority of one architecture, reflection as a causal mechanism, or production-level repair performance.

## Safety and observability

PatchPilot's runtime includes:

- Pydantic-validated tool-specific arguments;
- canonicalization of supported argument aliases;
- manifest-defined editable and forbidden paths;
- protected regression tests;
- isolated task workspaces;
- step, patch, tool-call, and runtime budgets;
- patch restoration and bounded termination;
- visible and hidden executable verification;
- machine-readable actions, observations, outcomes, and metadata.

A zero invalid-patch count in the final evaluation means no run ended with a counted invalid-patch outcome under the implemented checks. It is not a claim that a language model cannot propose incorrect or unsafe code.

## Final report

The polished 17-page report is committed at:

```text
docs/PatchPilot_Final_Report.pdf
```

Its reproducible source package is at:

```text
docs/final_report/
```

Where a LaTeX installation is available:

```bash
cd docs/final_report
./build_report.sh
```

## Limitations

- The primary corpus contains 53 small Python repair tasks.
- The eight manually authored hard tasks were unsolved by all conditions.
- The evaluation used one local model and one fixed seed.
- The configured reflective condition did not record completed reflection events.
- Results are scaffold-, prompt-, model-, budget-, and benchmark-specific.
- The system targets bounded source repair, not arbitrary repository maintenance.
- Live model-backed execution depends on a local Ollama service.
- The project does not claim completed SWE-bench performance or production readiness.

## Citation

For academic discussion, cite the repository and final report:

```text
Tivdzua Lubem Noah and Gisele Wiykiynyuy.
PatchPilot: A Safety-Constrained Tool-Using Agent for Python Debugging and Repair.
Master's Agentic AI Project Report, 2026.
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

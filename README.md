---
title: PatchPilot Agentic AI
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# PatchPilot: A Tool-Using Agent for Python Debugging and Repair

PatchPilot is a bounded Agentic AI system for Python debugging and repair. Given a small Python repository with failing tests, it reproduces the failure, inspects source code through restricted tools, applies a controlled patch, verifies the result with pytest, and reports success only when the current repository revision passes the configured test command.

The project focuses on safe, auditable tool use rather than unrestricted code generation. The language model proposes actions, while the PatchPilot runtime validates tool calls, restricts editable paths, applies patches, rolls back failed edits, enforces budgets, and records machine-readable traces for each run.

## Highlights

- Test-driven Python repair with a bounded tool loop.
- Tools for running tests, searching code, reading files, applying patches, restoring files, and finishing runs.
- Isolated workspace per benchmark run.
- Manifest-level `allowed_paths` and `forbidden_paths` safety boundaries.
- Patch rollback after failed verification.
- Step, tool-call, patch-attempt, and runtime budgets.
- Local Ollama/Qwen model backend.
- Streamlit live demo for both controlled and mutmut-generated tasks.
- Reproducible benchmark generation, evaluation scripts, result CSVs, figures, and tables.

## Architecture

```text
Benchmark task
  -> isolated workspace
  -> structured LLM repair policy
  -> validated tool action
  -> restricted tool executor
  -> observation and trace update
  -> verified success, escalation, or budget exhaustion
```

The model does not directly edit repository files. PatchPilot validates and executes all actions through controlled tools.

## Repository layout

```text
benchmark_seeds/                         Seed projects for generated benchmarks
benchmarks/                              Controlled repair benchmark tasks
generated_benchmarks/mutmut_algorithms/  Mutmut-generated repair benchmark
demo/                                    Streamlit live demo
docs/                                    Documentation and report assets
docs/report_assets/                      Generated figures and tables
results/                                 Committed machine-readable result summaries
scripts/                                 Benchmark, demo, evaluation, and asset scripts
src/patchpilot/                          PatchPilot package
tests/                                   Unit and integration tests
```

## Requirements

- Python 3.12
- Git
- Ollama for live local model runs
- `qwen2.5-coder:1.5b` for the default local demo
- Docker only for the optional containerized frontend

Install the default model:

```bash
ollama pull qwen2.5-coder:1.5b
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the quality gate:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src
git diff --check
```

Current verified status:

```text
107 tests passed
ruff passed
mypy passed
git diff --check passed
```

## Live repair demo from the CLI

Controlled task:

```bash
python scripts/run_demo_task.py \
  --task-id calculator-001 \
  --manifest-path benchmarks/calculator-001/task.json \
  --model qwen2.5-coder:1.5b \
  --output-root artifacts/demo_smoke_controlled
```

Mutmut-generated task:

```bash
python scripts/run_demo_task.py \
  --task-id mutmut-alg-mutmut-algorithms-core-x-add-mutmut-1 \
  --manifest-path generated_benchmarks/mutmut_algorithms/mutmut-alg-mutmut-algorithms-core-x-add-mutmut-1/task.json \
  --model qwen2.5-coder:1.5b \
  --output-root artifacts/demo_smoke_mutmut
```

A successful run returns JSON with `status: succeeded`, `full_suite_passed: true`, and one or more changed source files. Runtime outputs are written under `artifacts/`, which is intentionally ignored by Git.

## Streamlit live demo

Install demo dependencies:

```bash
python -m pip install -r demo/requirements-demo.txt
```

Run the UI:

```bash
streamlit run demo/streamlit_app.py \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.port 8501
```

Open:

```text
http://localhost:8501
```

The UI supports both controlled tasks and mutmut-generated tasks. It lets a user select a task, inspect source and tests, run PatchPilot live, view the tool timeline, inspect the patch diff, and confirm pytest verification.

Hosted frontend:

```text
https://lubem-patchpilot-agentic-ai.hf.space/
```

The Hugging Face Space loads the frontend. Live repair requires access to an Ollama server, so local execution is the reliable live demonstration path.

## Docker frontend

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

Live repair inside Docker requires the container to reach a running Ollama service.

## Evaluation summary

PatchPilot is evaluated on four tracks.

| Benchmark | Purpose | Result |
| --- | --- | --- |
| Controlled benchmark | End-to-end sanity check | Full, one-shot, and no-retry all reached 12/12 |
| Mutmut-generated benchmark | Primary ablation benchmark | Full 8/20, one-shot 6/20, no-retry 5/20 |
| QuixBugs smoke | External generalization check | 3/8 repaired |
| SWE-bench Lite | Official-harness feasibility check | Attempted; full run blocked by local WSL/Docker I/O instability |

Primary mutmut result:

```text
Full agent: 8/20 = 40%
One-shot: 6/20 = 30%
No-retry: 5/20 = 25%
Invalid patch rate: 0% for all three conditions
```

The controlled benchmark is saturated and is used as a reliability check, not as a superiority claim. The mutmut-generated benchmark is the main ablation comparison.

Committed result summaries are in `results/`. Generated report figures and tables are in `docs/report_assets/`.

## Reproducing evaluations

Run the controlled benchmark:

```bash
python scripts/run_evaluation.py \
  --condition full-live-qwen \
  --output-root artifacts/evaluation_controlled_full

python scripts/run_evaluation.py \
  --condition one-shot-live-qwen \
  --output-root artifacts/evaluation_controlled_oneshot

python scripts/run_evaluation.py \
  --condition no-retry-live-qwen \
  --output-root artifacts/evaluation_controlled_noretry
```

Regenerate the mutmut benchmark from the seed project:

```bash
python scripts/generate_mutmut_benchmark.py \
  --source-root benchmark_seeds/mutmut_algorithms \
  --source-path src/mutmut_algorithms \
  --test-path tests \
  --pytest-pythonpath src \
  --output-root generated_benchmarks/mutmut_algorithms \
  --task-prefix mutmut-alg \
  --max-tasks 20 \
  --force
```

Run mutmut evaluations:

```bash
python scripts/run_evaluation.py \
  --condition full-live-qwen \
  --manifest-root generated_benchmarks/mutmut_algorithms \
  --output-root artifacts/evaluation_mutmut_algorithms

python scripts/run_evaluation.py \
  --condition one-shot-live-qwen \
  --manifest-root generated_benchmarks/mutmut_algorithms \
  --output-root artifacts/evaluation_mutmut_algorithms_oneshot

python scripts/run_evaluation.py \
  --condition no-retry-live-qwen \
  --manifest-root generated_benchmarks/mutmut_algorithms \
  --output-root artifacts/evaluation_mutmut_algorithms_noretry
```

Summarize paired mutmut outcomes:

```bash
python scripts/summarize_mutmut_ablation.py \
  --full artifacts/evaluation_mutmut_algorithms/<TIMESTAMP>/runs.csv \
  --one-shot artifacts/evaluation_mutmut_algorithms_oneshot/<TIMESTAMP>/runs.csv \
  --no-retry artifacts/evaluation_mutmut_algorithms_noretry/<TIMESTAMP>/runs.csv \
  --output-csv results/mutmut_paired_outcomes.csv
```

Run the QuixBugs smoke benchmark:

```bash
python scripts/run_quixbugs_smoke.py
```

Regenerate result tables and report figures:

```bash
python scripts/generate_report_assets.py
```

## Statistical comparison

The mutmut benchmark uses paired task outcomes. Exact paired McNemar/binomial sign-test results are stored in `results/statistical_tests.csv`.

| Comparison | First-only successes | Second-only successes | p-value |
| --- | ---: | ---: | ---: |
| Full agent vs one-shot | 2 | 0 | 0.5000 |
| Full agent vs no-retry | 3 | 0 | 0.2500 |
| One-shot vs no-retry | 1 | 0 | 1.0000 |

The full agent has the highest mutmut repair rate, but the paired tests are not statistically significant at p < 0.05 because the benchmark contains 20 tasks.

## Safety design

PatchPilot uses a restricted execution boundary:

- tool calls are schema-validated;
- edits are limited to allowed source paths;
- tests are protected from modification;
- patch attempts are budgeted;
- repeated no-progress actions are blocked;
- failed patches can be restored before retry;
- success requires a passing pytest run;
- run traces are written for auditability.

## Committed vs ignored files

Committed:

- source code;
- benchmark definitions;
- mutmut seed and generated benchmark tasks;
- evaluation and demo scripts;
- result summary CSVs;
- final report assets.

Ignored:

- `artifacts/`;
- raw workspaces;
- raw traces;
- temporary cloned repositories;
- Python cache files;
- Docker/runtime caches.

## Known limitations

- PatchPilot targets small Python repositories with pytest suites.
- The repair policy is conservative and bounded.
- The controlled benchmark is saturated.
- The mutmut benchmark has limited statistical power at 20 tasks.
- QuixBugs is used as a smoke test, not a full benchmark suite.
- SWE-bench Lite full evaluation was not completed because the local WSL/Docker environment became unstable.
- Live repair depends on a local Ollama model being available.

## License

This project is released under the MIT License. See `LICENSE` for details.

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

PatchPilot is a bounded Agentic AI system for autonomous Python debugging and repair. It treats software repair as an agent-environment loop: reproduce the failure, inspect relevant code, form a repair hypothesis, apply a controlled patch, verify with tests, and stop only when the current repository revision is executable and verified.

The project focuses on safe, auditable agent execution rather than unrestricted code generation. The language model proposes repair actions inside a constrained runtime; the runtime validates tool calls, restricts file access, applies patches safely, records traces, enforces budgets, and confirms success through pytest.

## Key Features

- Single-agent Plan-Act-Observe-Reflect-Verify workflow
- Restricted repository tools for listing, reading, searching, testing, patching, diff inspection, and rollback
- Isolated benchmark workspaces for repair attempts
- Path restrictions to prevent edits outside allowed source files
- Patch validation and rollback support
- Loop guard against repeated identical tool actions
- Execution budgets for steps, tool calls, patch attempts, and runtime
- Auditable JSON traces for every repair run
- Ollama/Qwen local-model integration
- Final success allowed only after full-suite verification

## Current Status

PatchPilot has a working end-to-end live repair workflow using a local Ollama model.

Verified live workflow:

```text
run_tests -> search_code -> read_file -> apply_patch -> run_tests -> finish
```

The current local quality gate passes:

```text
pytest: 86 passed
ruff: passed
mypy: passed
git diff --check: passed
```

A successful live run repairs the seeded calculator benchmark and finishes only after the test suite passes.

## Architecture

```text
User / Benchmark Task
        |
        v
Agent State
        |
        v
Structured LLM Policy
        |
        v
Validated Tool Action
        |
        v
Restricted Tool Executor
        |
        +--> Repository tools
        +--> Test runner
        +--> Patch manager
        +--> Rollback / diff tools
        |
        v
Tool Observation
        |
        v
Trace Recorder
        |
        v
Verified Finish / Escalation / Failure
```

PatchPilot is intentionally designed so the model does not directly mutate the repository. The model proposes actions, while the runtime validates and executes them.

## Model Backend

The live local pilot uses Ollama with Qwen2.5-Coder.

The default live script currently uses:

```text
qwen2.5-coder:1.5b
```

This model was selected for reliable CPU-only execution in a low-memory local environment. The pipeline is model-swappable: stronger local or hosted code models can be connected through the same text-generation interface.

## Repository Layout

```text
benchmarks/             Seed repair tasks
docs/                   Project documentation
scripts/                Demo and live-run scripts
src/patchpilot/agent/   Agent policy, executor, loop control, tracing
src/patchpilot/tools/   Repository, test, and patch tools
src/patchpilot/models/  Model backends
src/patchpilot/schemas/ Shared state and tool schemas
tests/                  Unit and integration tests
```

## Quickstart

Create and activate a Python 3.12 virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the local quality gate:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src
git diff --check
```

## Running the Live Ollama Demo

Install and start Ollama, then pull the local model:

```bash
ollama pull qwen2.5-coder:1.5b
```

Run the live repair demo:

```bash
python scripts/run_live_qwen.py
```

A successful run prints the agent status, tool steps, changed files, workspace path, and trace path.

Expected successful flow:

```text
STATUS=succeeded
STEP_1=run_tests|error
STEP_2=search_code|ok
STEP_3=read_file|ok
STEP_4=apply_patch|ok
STEP_5=run_tests|ok
STEP_6=finish|ok
```


## Interactive Demo

PatchPilot includes a Streamlit prototype that allows users to select a benchmark task, inspect the broken source and tests, run the agent locally, view the tool-use trace, inspect the patch diff, and confirm final pytest verification.

Install demo dependencies:

```bash
python -m pip install -r demo/requirements-demo.txt
```

Run the demo:

```bash
streamlit run demo/streamlit_app.py --server.headless true --browser.gatherUsageStats false
```

Open:

```text
http://localhost:8501
```

The live repair button requires Ollama and the selected model to be available locally.

## Docker Demo

Build and run the demo container:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

The Docker demo packages the repository and Streamlit frontend. Live Ollama execution requires access to a running Ollama service from the container environment.

## Safety Design

PatchPilot uses a restricted execution boundary:

- tool calls are schema-validated before execution;
- file access is limited to repository-relative paths;
- benchmark tests are protected from modification;
- patch attempts are budgeted;
- repeated identical actions are rejected;
- success requires full-suite verification on the current revision;
- every action and observation is recorded in an auditable trace.

## Evaluation Plan

The intended evaluation compares the full PatchPilot agent against:

1. one-shot patch generation;
2. a fixed one-pass debugging workflow;
3. a reduced tool-using agent without reflection.

Primary metrics:

- full repair rate;
- full regression-test pass rate;
- invalid patch rate;
- average repair attempts;
- tool-call count;
- execution time;
- rollback frequency;
- budget exhaustion rate.

## Project Scope

PatchPilot targets small Python repositories with executable pytest suites. It is designed for research and demonstration of bounded agentic software repair, not unrestricted production code modification.

## License

License information will be added before final release.

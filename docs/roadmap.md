# PatchPilot Roadmap

PatchPilot currently has a working bounded repair agent, a successful live Ollama/Qwen repair run, and a passing local quality gate.

## Completed

- Bounded single-agent repair loop
- Restricted repository tools
- Isolated benchmark workspaces
- Patch application and rollback support
- Repeated-action loop guard
- Auditable JSON traces
- Ollama/Qwen model backend
- Live repair script
- Public README

## Remaining Work

### Evaluation

- Expand the benchmark to 12-18 validated mutation-seeded Python repair tasks.
- Implement one-shot patch generation baseline.
- Implement fixed one-pass debugging baseline.
- Implement reduced-agent/no-reflection ablation.
- Collect repair rate, regression pass rate, invalid patch rate, attempts, tool calls, runtime, rollback frequency, and budget exhaustion.

### Demo

- Add a FastAPI backend wrapper for running benchmark tasks.
- Add a Streamlit frontend for selecting tasks, viewing traces, inspecting diffs, and seeing verification status.
- Package a local demo and evaluate whether Hugging Face Spaces or another hosted option is suitable for deployment.

### Report

- Write the final technical report.
- Include architecture, safety design, benchmark methodology, results, limitations, and future work.
- Add trace screenshots or trace excerpts from successful and failed runs.

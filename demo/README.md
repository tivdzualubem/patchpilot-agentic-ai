# PatchPilot Demo

The demo is an interactive prototype, not only a results website.

It supports:

- selecting a benchmark repair task;
- viewing the broken source and regression tests;
- running PatchPilot locally against the selected task;
- displaying the tool-use trace;
- displaying the generated patch diff;
- showing final pytest verification status;
- reviewing evaluation and statistical evidence.

## Run locally

```bash
python -m pip install -r demo/requirements-demo.txt
streamlit run demo/streamlit_app.py --server.headless true
```

Open:

```text
http://localhost:8501
```

## Run one live repair from CLI

```bash
python scripts/run_demo_task.py --task-id calculator-001
```

## Docker

```bash
docker compose up --build
```

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY benchmarks ./benchmarks
COPY docs ./docs
COPY scripts ./scripts
COPY demo ./demo

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[dev]" \
    && python -m pip install --no-cache-dir -r demo/requirements-demo.txt

EXPOSE 8501

CMD ["streamlit", "run", "demo/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]

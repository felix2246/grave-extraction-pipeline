FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    libgl1 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY . .
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--no-dev", "python", "scripts/run_pipeline.py"]

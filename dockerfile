FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

COPY pyproject.toml ./

RUN uv sync

COPY . .

CMD ["uv", "run", "python", "utils/pipeline.py"]

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

COPY pyproject.toml .

COPY uv.lock .

RUN uv sync --frozen --no-dev

COPY . .

CMD ["uv", "run", "python", "utils/pipeline.py"]

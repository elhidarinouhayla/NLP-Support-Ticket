FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

# Copie des fichiers de dependances
COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --frozen --no-dev

# Copie du code source
COPY utils/ ./utils/

# Creation des repertoires 
RUN mkdir -p /app/data/raw /app/data/processed /app/models

# Variables d'environnement
ENV CHROMA_DB_PATH="/app/data/chroma_db"

CMD ["uv", "run", "python", "utils/pipeline.py"]
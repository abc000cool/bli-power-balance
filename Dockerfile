# Reproducibility fallback: identical results without a local Python setup.
#   docker build -t blipb .
#   docker run --rm blipb uv run pytest
#   docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/figures:/app/figures \
#       blipb uv run python studies/atlas.py
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --extra dev

COPY tests ./tests
COPY studies ./studies
COPY validation ./validation
COPY SPEC.md ./

CMD ["uv", "run", "pytest"]

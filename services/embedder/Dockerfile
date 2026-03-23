FROM ghcr.io/prakharmodi26/hokiehelp-base:latest

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

# Base image already has torch, sentence-transformers, qdrant-client, minio, etc.
# This only installs the package itself — takes seconds.
RUN pip install --no-cache-dir --no-deps .

CMD ["python", "-m", "embedder.main"]

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

# Install CUDA-enabled PyTorch + service deps
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir .

# Pre-download the embedding model so the CronJob doesn't re-download 1.3GB each run
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

CMD ["python", "-m", "embedder.main"]

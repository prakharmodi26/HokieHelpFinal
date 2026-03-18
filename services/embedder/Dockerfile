FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

# Install service deps (torch is already in base image with CUDA support)
RUN pip install --no-cache-dir .

# Pre-download the embedding model so the CronJob doesn't re-download 1.3GB each run
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

CMD ["python", "-m", "embedder.main"]

FROM python:3.11-slim

# Common Python dependencies shared by chatbot and embedder
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    sentence-transformers>=2.2.0 \
    qdrant-client>=1.7.0 \
    openai>=1.0.0 \
    minio>=7.2.0 \
    fastapi>=0.115.0 \
    "uvicorn[standard]>=0.30.0"

# Pre-download the embedding model so it's baked into the image.
# No HuggingFace downloads at pod startup.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

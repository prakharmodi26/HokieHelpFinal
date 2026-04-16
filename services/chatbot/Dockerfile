FROM --platform=linux/amd64 ghcr.io/prakharmodi26/hokiehelp-base:latest

LABEL org.opencontainers.image.source=https://github.com/prakharmodi26/HokieHelpFinal

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

# Base image already has torch, sentence-transformers, qdrant-client, etc.
# Install ollama (new dep not in base image) then the package itself.
RUN pip install --no-cache-dir ollama && pip install --no-cache-dir --no-deps .

EXPOSE 8000

CMD ["python", "-m", "chatbot.main"]

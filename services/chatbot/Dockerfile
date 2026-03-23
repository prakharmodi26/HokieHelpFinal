FROM ghcr.io/prakharmodi26/hokiehelp-base:latest

LABEL org.opencontainers.image.source=https://github.com/prakharmodi26/HokieHelpFinal

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

# Base image already has torch, sentence-transformers, qdrant-client, openai, etc.
# This only installs the package itself — takes seconds.
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

CMD ["python", "-m", "chatbot.main"]

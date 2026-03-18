FROM python:3.11-slim

LABEL org.opencontainers.image.source=https://github.com/prakharmodi26/HokieHelpFinal

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "chatbot.main"]

FROM ghcr.io/astral-sh/uv:0.11.16 AS uv
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.source=https://github.com/prakharmodi26/HokieHelpFinal
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ src/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER 1000:1000
EXPOSE 8000
CMD ["python", "-m", "chatbot.main"]

FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install crawler (brings crawl4ai, minio, httpx)
COPY crawler/pyproject.toml crawler/
COPY crawler/src/ crawler/src/
RUN pip install --no-cache-dir ./crawler && python -m playwright install chromium --with-deps

# Install chunker
COPY chunker/pyproject.toml chunker/
COPY chunker/src/ chunker/src/
RUN pip install --no-cache-dir ./chunker

# Install admin
COPY admin/pyproject.toml admin/
COPY admin/src/ admin/src/
RUN pip install --no-cache-dir ./admin

EXPOSE 8080
CMD ["uvicorn", "admin.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

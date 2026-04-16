FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget gnupg \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with writable HOME (crawl4ai writes ~/.crawl4ai).
# Shared playwright browser cache readable by UID 1000.
RUN useradd -u 1000 -m -d /home/admin -s /usr/sbin/nologin admin \
    && mkdir -p /opt/ms-playwright \
    && chown -R 1000:1000 /opt/ms-playwright /home/admin
ENV HOME=/home/admin \
    PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright \
    CRAWL4_AI_BASE_DIRECTORY=/home/admin

# Install crawler (brings crawl4ai, minio, httpx) + playwright chromium.
# Install as root so --with-deps can apt-install system packages,
# then chown caches to UID 1000 so runtime user can read.
COPY crawler/pyproject.toml crawler/
COPY crawler/src/ crawler/src/
RUN pip install --no-cache-dir ./crawler \
    && python -m playwright install chromium --with-deps \
    && chown -R 1000:1000 /opt/ms-playwright /home/admin

# Install chunker
COPY chunker/pyproject.toml chunker/
COPY chunker/src/ chunker/src/
RUN pip install --no-cache-dir ./chunker

# Install admin
COPY admin/pyproject.toml admin/
COPY admin/src/ admin/src/
RUN pip install --no-cache-dir ./admin

USER 1000
EXPOSE 8080
CMD ["uvicorn", "admin.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

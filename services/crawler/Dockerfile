FROM python:3.11-slim

# Crawl4AI needs a browser for rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files and install
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

# Crawl4AI post-install: download browser
RUN crawl4ai-setup

# Run the crawler
CMD ["python", "-m", "crawler.main"]

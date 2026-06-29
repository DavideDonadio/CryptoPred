FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install system deps needed for undetected-chromedriver
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python deps
RUN uv sync --no-dev

# --- Runtime stage ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    cron \
    xvfb \
    xauth \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Chrome environment variables
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV DISPLAY=:99

# Copy application code
COPY main.py ./
COPY fetch_reddit.py ./
COPY live_data_builder.py ./
COPY pipeline.sh ./
COPY entrypoint.sh ./

# Copy model files only (runtime data comes from volume mount)
COPY data/models/ ./data/models/
RUN mkdir -p /app/data/clean_reddit

# Make scripts executable
RUN chmod +x pipeline.sh entrypoint.sh

# Install crontab
COPY crontab.txt /etc/cron.d/cryptopred
RUN chmod 0644 /etc/cron.d/cryptopred && \
    crontab /etc/cron.d/cryptopred && \
    touch /var/log/cron.log

# Create log directory
RUN mkdir -p /app/logs

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
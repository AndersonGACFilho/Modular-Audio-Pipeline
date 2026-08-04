FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Silero VAD is loaded by torch.hub. Warm its cache while building the image so
# a production job never needs to download executable code from GitHub.
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, onnx=False, trust_repo=True)"

COPY app ./app
COPY config.json ./
COPY docker/entrypoint.sh /usr/local/bin/audio-pipeline
RUN chmod +x /usr/local/bin/audio-pipeline

ENTRYPOINT ["/usr/local/bin/audio-pipeline"]

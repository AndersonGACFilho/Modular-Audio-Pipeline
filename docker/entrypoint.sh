#!/bin/sh
set -eu

mkdir -p /data/incoming

case "${AUDIO_PIPELINE_MODE:-rabbit}" in
  rabbit)
    exec python -m audio_pipeline.entrypoints.rabbitmq_worker
    ;;
  cli)
    exec python -m audio_pipeline.entrypoints.cli \
      -c "${AUDIO_PIPELINE_CLI_CONFIG:-/app/config.json}" \
      --media-dir "${AUDIO_PIPELINE_CLI_MEDIA_DIR:-/data}"
    ;;
  *)
    echo "AUDIO_PIPELINE_MODE must be 'cli' or 'rabbit', got: ${AUDIO_PIPELINE_MODE}" >&2
    exit 64
    ;;
esac

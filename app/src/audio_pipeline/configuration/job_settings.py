"""Environment-backed settings for local jobs, MongoDB, and RabbitMQ."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobSettings:
    data_root: Path
    mongodb_uri: str
    mongodb_database: str
    mongodb_collection: str
    rabbitmq_url: str
    rabbitmq_queue: str
    rabbitmq_dead_letter_queue: str
    lease_seconds: int

    @classmethod
    def from_environment(cls) -> "JobSettings":
        return cls(
            data_root=Path(os.getenv("AUDIO_PIPELINE_DATA_ROOT", "data")).resolve(),
            mongodb_uri=os.getenv("AUDIO_PIPELINE_MONGODB_URI", "mongodb://localhost:27017"),
            mongodb_database=os.getenv("AUDIO_PIPELINE_MONGODB_DATABASE", "audio_pipeline"),
            mongodb_collection=os.getenv("AUDIO_PIPELINE_MONGODB_COLLECTION", "audio_jobs"),
            rabbitmq_url=os.getenv("AUDIO_PIPELINE_RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F"),
            rabbitmq_queue=os.getenv("AUDIO_PIPELINE_RABBITMQ_QUEUE", "audio_jobs"),
            rabbitmq_dead_letter_queue=os.getenv("AUDIO_PIPELINE_RABBITMQ_DEAD_LETTER_QUEUE", "audio_jobs.dead"),
            lease_seconds=int(os.getenv("AUDIO_PIPELINE_JOB_LEASE_SECONDS", "3600")),
        )

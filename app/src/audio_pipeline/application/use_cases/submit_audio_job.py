"""Register an uploaded local file and publish its processing job."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ...domain.models import AudioJob, MediaAsset
from ..ports import JobPublisher, JobRepository


class SubmitAudioJob:
    def __init__(self, repository: JobRepository, queue: JobPublisher) -> None:
        self._repository = repository
        self._queue = queue

    def execute(self, source_path: str | Path, options: dict | None = None) -> AudioJob:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Uploaded media file does not exist: {source}")

        digest = hashlib.sha256()
        with source.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        job = AudioJob(
            job_id=str(uuid4()),
            source=MediaAsset(path=str(source.resolve()), original_name=source.name, size_bytes=source.stat().st_size, sha256=digest.hexdigest()),
            options=options or {},
            created_at=datetime.now().astimezone(),
        )
        self._repository.create(job)
        self._queue.publish(job.job_id)
        return job

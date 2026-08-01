"""Execute one persisted job through the existing audio pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from socket import gethostname

from ..ports import AudioProcessor, JobRepository, JobStorage
from ...domain.models import JobError, JobResult


class ProcessAudioJob:
    def __init__(self, repository: JobRepository, storage: JobStorage, processor: AudioProcessor, lease_seconds: int) -> None:
        self._repository = repository
        self._storage = storage
        self._processor = processor
        self._lease_seconds = lease_seconds

    def execute(self, job_id: str) -> None:
        lease_until = datetime.now(timezone.utc) + timedelta(seconds=self._lease_seconds)
        job = self._repository.claim(job_id, gethostname(), lease_until)
        if job is None:
            return

        processing_directory = self._storage.prepare_processing(job.job_id)
        results_directory = self._storage.results_directory(job.job_id, job.created_at)
        try:
            result = self._processor.process(job, processing_directory, results_directory)
            if not result.success or not result.output_file:
                raise RuntimeError(result.error or "Audio pipeline completed without an output file.")
            job.mark_completed(JobResult(output_path=result.output_file, segment_count=len(result.segments), metadata=result.metadata))
            self._repository.save(job)
        except Exception as error:
            job.mark_failed(JobError(message=str(error), error_type=type(error).__name__))
            self._repository.save(job)
            raise
        finally:
            self._storage.cleanup_processing(job.job_id)

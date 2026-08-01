"""Execute one persisted job through the existing audio pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from socket import gethostname

from ...application.pipeline import AudioPipeline
from ...config import PipelineConfig
from ...domain.models import JobError, JobResult
from ...domain.ports import JobRepository, JobStorage


class ProcessAudioJob:
    def __init__(self, repository: JobRepository, storage: JobStorage, lease_seconds: int) -> None:
        self._repository = repository
        self._storage = storage
        self._lease_seconds = lease_seconds

    def execute(self, job_id: str) -> None:
        lease_until = datetime.now(timezone.utc) + timedelta(seconds=self._lease_seconds)
        job = self._repository.claim(job_id, gethostname(), lease_until)
        if job is None:
            return

        processing_directory = self._storage.prepare_processing(job.job_id)
        results_directory = self._storage.results_directory(job.job_id, job.created_at)
        config = PipelineConfig(
            media_dir=str(Path(job.source.path).parent),
            temp_dir=str(processing_directory / "temp"),
            results_dir=str(results_directory),
            checkpoint_dir=str(processing_directory / "checkpoints"),
        )
        pipeline = AudioPipeline(config)
        try:
            result = pipeline.run(input_file=job.source.path)
            if not result.success or not result.output_file:
                raise RuntimeError(result.error or "Audio pipeline completed without an output file.")
            self._repository.mark_completed(job.job_id, JobResult(output_path=result.output_file, segment_count=len(result.segments), metadata=result.metadata))
        except Exception as error:
            self._repository.mark_failed(job.job_id, JobError(message=str(error), error_type=type(error).__name__))
            raise
        finally:
            pipeline.cleanup()
            self._storage.cleanup_processing(job.job_id)

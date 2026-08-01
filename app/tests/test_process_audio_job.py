from datetime import datetime, timezone
from pathlib import Path

from audio_pipeline.application.pipeline import PipelineResult
from audio_pipeline.application.use_cases import ProcessAudioJob
from audio_pipeline.domain.models import AudioJob, MediaAsset


class InMemoryRepository:
    def __init__(self, job):
        self.job = job
        self.saved = None

    def claim(self, *_args):
        self.job.mark_processing()
        return self.job

    def save(self, job):
        self.saved = job


class Workspace:
    def __init__(self, root):
        self.root = root
        self.cleaned = False

    def prepare_processing(self, _job_id):
        return self.root / "processing"

    def results_directory(self, _job_id, _created_at):
        return self.root / "results"

    def cleanup_processing(self, _job_id):
        self.cleaned = True


class SuccessfulProcessor:
    def process(self, job, _processing_directory, _results_directory):
        return PipelineResult(True, job.source.path, "result.json", [{"text": "done"}], metadata={"backend": "test"})


def test_process_audio_job_transitions_the_aggregate_before_persisting(tmp_path):
    job = AudioJob(
        job_id="job-1",
        source=MediaAsset(path=str(tmp_path / "source.wav"), original_name="source.wav", size_bytes=1),
        created_at=datetime.now(timezone.utc),
    )
    repository = InMemoryRepository(job)
    workspace = Workspace(tmp_path)

    ProcessAudioJob(repository, workspace, SuccessfulProcessor(), lease_seconds=60).execute(job.job_id)

    assert repository.saved is job
    assert job.status.value == "completed"
    assert job.result.output_path == "result.json"
    assert workspace.cleaned is True

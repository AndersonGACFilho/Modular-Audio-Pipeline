"""Adapter that executes the pipeline on a job workspace."""

from pathlib import Path

from ..application.pipeline import PipelineResult
from ..config import PipelineConfig
from ..domain.models import AudioJob
from .pipeline_factory import create_audio_pipeline


class AudioPipelineProcessor:
    def process(self, job: AudioJob, processing_directory: Path, results_directory: Path) -> PipelineResult:
        config = PipelineConfig(
            media_dir=str(Path(job.source.path).parent),
            temp_dir=str(processing_directory / "temp"),
            results_dir=str(results_directory),
            checkpoint_dir=str(processing_directory / "checkpoints"),
        )
        config.transcription.language = job.options.transcription.language
        config.transcription.locale = job.options.transcription.locale
        config.transcription.initial_prompt = job.options.transcription.initial_prompt
        config.transcription.hotwords = list(job.options.transcription.hotwords)
        config.transcription.condition_on_previous_text = job.options.transcription.condition_on_previous_text
        pipeline = create_audio_pipeline(config, analysis_options=job.options.analysis)
        try:
            return pipeline.run(input_file=job.source.path)
        finally:
            pipeline.cleanup()

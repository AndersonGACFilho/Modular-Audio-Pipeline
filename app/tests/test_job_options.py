from audio_pipeline.application.pipeline import PipelineResult
from audio_pipeline.bootstrap import audio_pipeline_processor
from audio_pipeline.bootstrap.audio_pipeline_processor import AudioPipelineProcessor
from audio_pipeline.domain.models import AnalysisOptions, AudioJob, AudioJobOptions, MediaAsset, TranscriptionOptions


def test_job_options_are_typed_and_preserve_asr_and_analysis_boundaries():
    options = AudioJobOptions.from_dict({
        "transcription": {"language": "pt", "hotwords": ["SIGOR"], "condition_on_previous_text": False},
        "analysis": {"profile_id": "technical_daily", "prompt": "Extract blockers."},
    })

    assert options.transcription.initial_prompt is None
    assert options.transcription.hotwords == ("SIGOR",)
    assert options.analysis.profile_id == "technical_daily"


def test_audio_processor_applies_the_job_options_to_the_pipeline_config(tmp_path, monkeypatch):
    job = AudioJob(
        job_id="job-1",
        source=MediaAsset(path=str(tmp_path / "source.wav"), original_name="source.wav", size_bytes=1),
        options=AudioJobOptions(
            transcription=TranscriptionOptions(language="pt", locale="pt-BR", hotwords=("SIGOR",), condition_on_previous_text=False),
            analysis=AnalysisOptions(profile_id="technical_daily", prompt="Extract blockers."),
        ),
    )
    captured = {}

    class FakePipeline:
        def run(self, input_file):
            return PipelineResult(True, input_file, "result.json", [])

        def cleanup(self):
            pass

    def create_pipeline(config, analysis_options):
        captured["config"] = config
        captured["analysis"] = analysis_options
        return FakePipeline()

    monkeypatch.setattr(audio_pipeline_processor, "create_audio_pipeline", create_pipeline)
    result = AudioPipelineProcessor().process(job, tmp_path / "processing", tmp_path / "results")

    assert result.success is True
    assert captured["config"].transcription.hotwords == ["SIGOR"]
    assert captured["config"].transcription.condition_on_previous_text is False
    assert captured["analysis"].profile_id == "technical_daily"

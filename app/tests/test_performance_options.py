from audio_pipeline.config import PipelineConfig
from audio_pipeline.application.pipeline import AudioPipeline
from audio_pipeline.config.profiles import ProfileRouter, ProfileRouting
from audio_pipeline.domain.naming import (
    contextual_output_stem,
    rename_derived_artifact,
    rename_source_media,
)
from audio_pipeline.documentation import archival_segments, documentation_text
from audio_pipeline.processing.postprocessing.hybrid import HybridLLMPostProcessor
from audio_pipeline.processing.speech.transcriber import FasterWhisperTranscriber


def test_config_exposes_performance_controls():
    config = PipelineConfig()

    assert config.transcription.batch_size >= 1
    assert config.transcription.word_timestamps is False
    assert config.llm.request_timeout_s > 0
    assert config.llm.chunk_size_chars >= 500


def test_long_llm_analysis_is_chunked_and_consolidated():
    processor = object.__new__(HybridLLMPostProcessor)
    processor.chunk_size_chars = 10
    processor.backend = "ollama"
    processor.profile_router = type(
        "Router",
        (),
        {"route": lambda self, text, source_path, classifier: ProfileRouting(
            profile="generic_meeting", confidence=1.0, reasoning="test"
        )},
    )()
    processor._process_chunk = lambda text: {
        "summary": text,
        "topics": ["topic"],
        "action_items": [],
        "sentiment": "Neutral",
    }
    processor._consolidate_chunks = lambda analyses: {
        "summary": "combined",
        "topics": ["topic"],
        "action_items": [],
        "sentiment": "Neutral",
    }

    result = processor.process("one two three four five")

    assert result["summary"] == "combined"
    assert result["formatting"]["profile"] == "generic_meeting"


def test_profile_fallback_uses_generic_profile_names_not_companies():
    routing = ProfileRouter().fallback(
        "files/results/Entrevistas/Kokku/recording.mp4"
    )

    assert routing.profile == "interview"
    assert routing.confidence == 0.65


def test_recruitment_content_overrides_a_misleading_mentoring_folder():
    transcript = (
        "I am the recruiter for this open position. Let's discuss compensation "
        "and the next steps in the interview process."
    )

    routing = ProfileRouter().route(
        transcript,
        "files/Mentorias/career_mentoring.mp4",
        classifier=lambda _prompt, _schema: {
            "profile": "career_mentoring",
            "confidence": 0.95,
            "reasoning": "Incorrectly followed the source folder",
        },
    )

    assert routing.profile == "interview"
    assert routing.confidence == 0.9


def test_contextual_output_name_uses_profile_and_detected_organization():
    stem = contextual_output_stem(
        "2026-06-08 14-30-46.mp4",
        {"profile": "interview", "organization": "Kokku Games"},
    )

    assert stem == "2026-06-08_14-30-46_interview_kokku-games"


def test_contextual_output_name_falls_back_without_routing_metadata():
    assert contextual_output_stem("recording.mp4", None) == "recording"


def test_source_media_rename_uses_contextual_stem(tmp_path):
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"media")

    renamed = rename_source_media(str(source), "2026-06-08_interview_kokku-games")

    assert renamed.endswith("2026-06-08_interview_kokku-games.mp4")
    assert not source.exists()


def test_generated_artifact_rename_preserves_processing_suffix(tmp_path):
    artifact = tmp_path / "recording_16000Hz_denoised_norm_loudnorm.wav"
    artifact.write_bytes(b"audio")

    renamed = rename_derived_artifact(
        str(artifact), "recording", "2026-06-08_interview_kokku-games"
    )

    assert renamed.endswith("2026-06-08_interview_kokku-games_16000Hz_denoised_norm_loudnorm.wav")
    assert not artifact.exists()


def test_archival_documentation_only_normalizes_whitespace():
    segments = [{"speaker": "SPEAKER_00", "start": 1.0, "text": "  Hello\n world  "}]

    archived = archival_segments(segments)

    assert archived[0]["text"] == "Hello world"
    assert segments[0]["text"] == "  Hello\n world  "
    assert documentation_text(archived) == "[00:00:01.000] SPEAKER_00: Hello world"


def test_batched_transcriber_uses_configured_options():
    calls = []

    class FakeInference:
        def transcribe(self, path, **kwargs):
            calls.append((path, kwargs))
            return [], object()

    transcriber = object.__new__(FasterWhisperTranscriber)
    transcriber._model = object()
    transcriber._inference = FakeInference()
    transcriber.beam_size = 5
    transcriber.language = "en"
    transcriber.task = "transcribe"
    transcriber.temperature = 0.0
    transcriber.prompt = "context"
    transcriber.internal_vad = False
    transcriber.word_timestamps = False
    transcriber.batch_size = 4

    transcriber._transcribe_with_model("audio.wav")

    assert calls == [
        (
            "audio.wav",
            {
                "beam_size": 5,
                "language": "en",
                "task": "transcribe",
                "temperature": 0.0,
                "initial_prompt": "context",
                "vad_filter": True,
                "word_timestamps": False,
                "batch_size": 4,
            },
        )
    ]


def test_pipeline_metrics_record_stage_and_total_duration():
    metrics = {"stage_durations_s": {}, "media": {}, "segments": {}}

    result = AudioPipeline._measure_stage(
        object(), metrics, "sample_stage", lambda: "completed"
    )
    completed = AudioPipeline._finalize_metrics(metrics, 0.0)

    assert result == "completed"
    assert completed["stage_durations_s"]["sample_stage"] >= 0
    assert completed["total_duration_s"] > 0

import logging
import urllib.error
import urllib.request

from audio_pipeline.config import PipelineConfig
from audio_pipeline.application.pipeline import AudioPipeline
from audio_pipeline.config.profiles import ProfileRouter, ProfileRouting
from audio_pipeline.domain.naming import contextual_output_stem
from audio_pipeline.infrastructure.storage.artifacts import rename_derived_artifact, rename_source_media
from audio_pipeline.documentation import archival_segments, documentation_text
from audio_pipeline.infrastructure.ai.hybrid import HybridLLMPostProcessor, SpeakerNameSuggestions
from audio_pipeline.infrastructure.speech.transcriber import FasterWhisperTranscriber


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


def test_speaker_name_suggestions_keep_only_known_diarization_labels():
    processor = object.__new__(HybridLLMPostProcessor)
    processor.backend = "ollama"
    processor._request_structured = lambda _prompt, _schema, max_length: {
        "suggestions": [
            {"speaker": "SPEAKER_00", "suggested_name": "Ana", "confidence": 0.9, "evidence": ["Eu sou a Ana."]},
            {"speaker": "SPEAKER_99", "suggested_name": "Bia", "confidence": 0.9, "evidence": ["irrelevant"]},
        ]
    }

    result = processor.suggest_speaker_names([
        {"speaker": "SPEAKER_00", "text": "Eu sou a Ana."},
        {"speaker": "SPEAKER_01", "text": "Prazer."},
    ])

    assert result == {"suggestions": [
        {"speaker": "SPEAKER_00", "suggested_name": "Ana", "confidence": 0.9, "evidence": ["Eu sou a Ana."]}
    ]}


def test_llm_activity_logs_start_and_completion(caplog):
    processor = object.__new__(HybridLLMPostProcessor)
    processor.backend = "ollama"
    processor.ollama_model = "qwen3.5:9b"

    with caplog.at_level(logging.INFO):
        with processor._llm_activity("analysis request"):
            pass

    assert "LLM started analysis request (ollama: qwen3.5:9b)" in caplog.text
    assert "LLM completed analysis request" in caplog.text


def test_local_fallback_uses_greedy_generation_and_cache():
    calls = []

    class Tokenizer:
        def __call__(self, value, **_kwargs):
            return {"input_ids": list(value)}

    processor = object.__new__(HybridLLMPostProcessor)
    processor.pipe = lambda prompt, **kwargs: calls.append((prompt, kwargs)) or [{"generated_text": prompt + "{}"}]
    processor.tokenizer = Tokenizer()
    processor.device = "cpu"
    processor.local_max_new_tokens = 192
    processor._extract_json = lambda value: {"response": value}

    result = processor._process_local("", user_prompt="source", max_new_tokens=64)

    assert result == {"response": "{}"}
    assert calls == [("source", {
        "max_new_tokens": 64,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
    })]


def test_ollama_probe_logs_why_a_fallback_is_used(monkeypatch, caplog):
    processor = object.__new__(HybridLLMPostProcessor)
    processor.ollama_host = "http://localhost:11434"

    def unavailable(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    with caplog.at_level(logging.WARNING):
        assert processor._detect_ollama() is None

    assert "Ollama unavailable at http://localhost:11434" in caplog.text
    assert "connection refused" in caplog.text


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
    transcriber.hotwords = ["SIGOR", "MTR"]
    transcriber.condition_on_previous_text = False
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
                "hotwords": "SIGOR, MTR",
                "condition_on_previous_text": False,
                "vad_filter": True,
                "word_timestamps": False,
                "batch_size": 4,
            },
        )
    ]


def test_pipeline_metrics_record_stage_and_total_duration():
    metrics = {"stage_durations_s": {}, "media": {}, "segments": {}}

    pipeline = object.__new__(AudioPipeline)
    pipeline.progress_callback = None
    result = pipeline._measure_stage(metrics, "sample_stage", lambda: "completed")
    completed = AudioPipeline._finalize_metrics(metrics, 0.0)

    assert result == "completed"
    assert completed["stage_durations_s"]["sample_stage"] >= 0
    assert completed["total_duration_s"] > 0

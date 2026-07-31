from audio_pipeline.config import PipelineConfig
from audio_pipeline.post_processing_hybrid import HybridLLMPostProcessor
from audio_pipeline.pipeline import AudioPipeline
from audio_pipeline.transcriber import FasterWhisperTranscriber


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
                "vad_filter": False,
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

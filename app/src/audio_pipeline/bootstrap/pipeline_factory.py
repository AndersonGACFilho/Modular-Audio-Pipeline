"""Composition root for the concrete audio-processing pipeline."""

from ..application.pipeline import AudioPipeline
from ..config import PipelineConfig
from ..domain.models import AnalysisOptions
from ..infrastructure.media.handler import MediaHandler
from ..infrastructure.media.preprocessor import AudioPreprocessor
from ..infrastructure.media.separator import NoOpVocalSeparator, VocalSeparator
from ..infrastructure.speech.diarizer import NoOpDiarizer, SpeakerDiarizer
from ..infrastructure.speech.redundancy import NoOpRedundancyRemover, RedundancyRemover
from ..infrastructure.speech.segment_merger import SegmentMerger
from ..infrastructure.speech.transcriber import FasterWhisperTranscriber, WhisperTranscriber
from ..infrastructure.speech.vad import NoOpVADFilter, SileroVADFilter, VADFilter
from ..infrastructure.storage.artifacts import LocalArtifactRenamer
from ..utils import CheckpointManager


def create_audio_pipeline(config: PipelineConfig, analysis_options: AnalysisOptions | None = None) -> AudioPipeline:
    """Build the production pipeline from a validated configuration."""
    config.validate()
    checkpoints = CheckpointManager(config.checkpoint_dir) if config.checkpoint_enabled else None
    separator = VocalSeparator.from_config(config, checkpoints) if config.vocal_separation.enabled else NoOpVocalSeparator()
    if not config.vad.enabled:
        vad = NoOpVADFilter()
    elif config.vad.provider == "silero":
        vad = SileroVADFilter(threshold=config.vad.threshold, sampling_rate=config.audio.sample_rate)
    else:
        vad = VADFilter.from_config(config)
    transcriber = FasterWhisperTranscriber.from_config(config) if config.transcription.backend == "faster-whisper" else WhisperTranscriber.from_config(config)
    diarizer = SpeakerDiarizer.from_config(config) if config.diarization.enabled else NoOpDiarizer()
    redundancy = RedundancyRemover.from_config(config) if config.redundancy.enabled else NoOpRedundancyRemover()
    merger = SegmentMerger(max_gap_s=config.segment_merging.max_gap_s)
    def create_llm_processor():
        from ..infrastructure.ai.hybrid import HybridLLMPostProcessor
        return HybridLLMPostProcessor(
            model=config.llm.openai_model, ollama_host=config.llm.ollama_host,
            ollama_model=config.llm.ollama_model, use_ollama=config.llm.use_ollama,
            use_openai=config.llm.use_openai, ollama_num_ctx=config.llm.ollama_num_ctx,
            ollama_keep_alive=config.llm.ollama_keep_alive, request_timeout_s=config.llm.request_timeout_s,
            chunk_size_chars=config.llm.chunk_size_chars, chunk_max_length=config.llm.chunk_max_length,
            disable_thinking=config.llm.disable_thinking, local_model=config.llm.local_model,
            device=config.llm.device, max_length=config.llm.max_length,
            temperature=config.llm.temperature, lazy_load=True,
            profile_id=analysis_options.profile_id if analysis_options else None,
            profile_prompt=analysis_options.prompt if analysis_options else None,
            output_language=analysis_options.output_language if analysis_options else "pt-BR",
        )
    return AudioPipeline(config, MediaHandler.from_config(config), AudioPreprocessor.from_config(config), separator, vad, transcriber, diarizer, redundancy, merger, checkpoints, LocalArtifactRenamer(), create_llm_processor)

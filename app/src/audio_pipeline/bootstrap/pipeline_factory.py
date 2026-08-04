"""Composition root for the concrete audio-processing pipeline."""

import logging
import os
from typing import Callable

from ..application.pipeline import AudioPipeline
from ..config import PipelineConfig
from ..config.gpu_profiles import apply_gpu_workload_profile
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

logger = logging.getLogger(__name__)


def create_audio_pipeline(
    config: PipelineConfig,
    analysis_options: AnalysisOptions | None = None,
    progress_callback: Callable[[str], None] | None = None,
    file_callback: Callable[[str], None] | None = None,
) -> AudioPipeline:
    """Build the production pipeline from a validated configuration."""
    if profile_name := os.getenv("AUDIO_PIPELINE_GPU_PROFILE"):
        profile = apply_gpu_workload_profile(config, profile_name)
        logger.info(
            "Applied GPU workload profile '%s' (transcription=%d, diarization=%d/%d)",
            profile_name,
            profile.transcription_batch_size,
            profile.diarization_segmentation_batch_size,
            profile.diarization_embedding_batch_size,
        )
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
            local_max_new_tokens=config.llm.local_max_new_tokens,
            local_attention_implementation=config.llm.local_attention_implementation,
            temperature=config.llm.temperature, lazy_load=True,
            profile_id=analysis_options.profile_id if analysis_options else None,
            profile_prompt=analysis_options.prompt if analysis_options else None,
            output_language=analysis_options.output_language if analysis_options else "pt-BR",
            progress_callback=progress_callback,
        )
    return AudioPipeline(config, MediaHandler.from_config(config), AudioPreprocessor.from_config(config), separator, vad, transcriber, diarizer, redundancy, merger, checkpoints, LocalArtifactRenamer(), create_llm_processor, progress_callback, file_callback)

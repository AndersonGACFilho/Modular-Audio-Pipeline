"""
audio_pipeline.pipeline

Orchestrator for the audio processing and transcription pipeline.

This module coordinates all pipeline steps: media discovery/conversion,
preprocessing (denoise, normalization, silence removal), optional vocal
separation, VAD, transcription, diarization, redundancy removal and final
output serialization.

The public API is the AudioPipeline class which accepts components via
dependency injection for testing and customization.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .protocols import (
    MediaHandlerProtocol,
    PreprocessorProtocol,
    VocalSeparatorProtocol,
    VADProtocol,
    TranscriberProtocol,
    DiarizerProtocol,
    RedundancyRemoverProtocol,
    DiarizationSegment,
    TimestampMapping
)
from .media_handler import MediaHandler
from .preprocessor import AudioPreprocessor
from .segment_merger import SegmentMerger
from .separator import VocalSeparator, NoOpVocalSeparator
from .vad import VADFilter, SileroVADFilter, NoOpVADFilter
from .transcriber import WhisperTranscriber, FasterWhisperTranscriber
from .diarizer import SpeakerDiarizer, NoOpDiarizer
from .redundancy import RedundancyRemover, NoOpRedundancyRemover
from .config import PipelineConfig, get_default_config
from .exceptions import AudioPipelineError, MediaNotFoundError
from .utils import CheckpointManager, ensure_directory

logger = logging.getLogger(__name__)

# Optional LLM post-processor (imported lazily).
# Declared here to satisfy static analysis
HybridLLMPostProcessor = None


@dataclass
class PipelineResult:
    """Result returned by AudioPipeline.run().

    Args:
        success:
            Whether the pipeline completed successfully.
        input_file:
            Path to the input media file processed.
        output_file:
            Path to the output transcription JSON file.
        segments:
            List of transcription segments with timing and speaker info.
        error:
            Optional error message if the pipeline failed.
        metadata:
            Additional metadata about the processing run.
        llm_analysis:
            Optional LLM analysis results if LLM post-processing was used.
    """
    success: bool
    input_file: str
    output_file: Optional[str]
    segments: List[Dict[str, Any]]
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    llm_analysis: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AudioPipeline:
    """Coordinates the full audio processing pipeline.

    The pipeline composes modular components and supports dependency
    injection for testing or custom implementations. The main entry point
    is run(input_file: Optional[str]) -> PipelineResult.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        media_handler: Optional[MediaHandlerProtocol] = None,
        preprocessor: Optional[PreprocessorProtocol] = None,
        separator: Optional[VocalSeparatorProtocol] = None,
        vad: Optional[VADProtocol] = None,
        transcriber: Optional[TranscriberProtocol] = None,
        diarizer: Optional[DiarizerProtocol] = None,
        redundancy_remover: Optional[RedundancyRemoverProtocol] = None
    ):
        """Create AudioPipeline.

        Parameters
        ----------
        config:
            PipelineConfig instance. If None, defaults are used.
        media_handler, preprocessor, separator, vad, transcriber, diarizer,
        redundancy_remover:
            Optional custom components implementing the corresponding
            protocols. If not provided, default implementations are created
            based on the configuration.
        """
        self.config = config or get_default_config()
        self.config.validate()

        # Setup directories
        self.media_dir = ensure_directory(self.config.media_dir)
        self.temp_dir = ensure_directory(self.config.temp_dir)
        self.results_dir = ensure_directory(self.config.results_dir)

        # Setup checkpoint manager
        self.checkpoint_manager = None
        if self.config.checkpoint_enabled:
            self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)

        # Initialize components
        self.media = media_handler or MediaHandler.from_config(self.config)
        self.preprocessor = preprocessor or AudioPreprocessor.from_config(self.config)

        # Separator
        if separator:
            self.separator = separator
        elif self.config.vocal_separation.enabled:
            self.separator = VocalSeparator.from_config(self.config, self.checkpoint_manager)
        else:
            self.separator = NoOpVocalSeparator()

        # VAD
        if vad:
            self.vad = vad
        elif self.config.vad.enabled:
            if self.config.vad.provider == "silero":
                logger.info("Using Silero VAD (optimized)")
                self.vad = SileroVADFilter(
                    threshold=self.config.vad.threshold,
                    sampling_rate=self.config.audio.sample_rate
                )
            else:
                logger.info("Using WebRTC VAD (legacy)")
                self.vad = VADFilter.from_config(self.config)
        else:
            self.vad = NoOpVADFilter()

        # Transcriber
        if transcriber:
            self.transcriber = transcriber
        elif self.config.transcription.backend == "faster-whisper":
            logger.info("Using FasterWhisper (optimized)")
            self.transcriber = FasterWhisperTranscriber.from_config(self.config)
        else:
            logger.info("Using standard Whisper")
            self.transcriber = WhisperTranscriber.from_config(self.config)

        # Diarizer
        if diarizer:
            self.diarizer = diarizer
        elif self.config.diarization.enabled:
            self.diarizer = SpeakerDiarizer.from_config(self.config)
        else:
            self.diarizer = NoOpDiarizer()

        # Redundancy remover
        if redundancy_remover:
            self.redundancy = redundancy_remover
        elif self.config.redundancy.enabled:
            self.redundancy = RedundancyRemover.from_config(self.config)
        else:
            self.redundancy = NoOpRedundancyRemover()

        # LLM Post-Processor — intentionally NOT loaded here.
        # It is initialized lazily inside run(), AFTER transcription and
        # diarization models have been unloaded, to avoid VRAM exhaustion
        # on GPUs with ≤ 8 GB of memory.
        self.llm_processor = None

        # Timestamp mappings
        self._timestamp_mappings: List[TimestampMapping] = []

    def _map_timestamp_to_original(
        self,
        processed_time: float,
        mappings: List[TimestampMapping]
    ) -> float:
        """Map timestamp from processed audio back to original.

        Parameters
        ----------
        processed_time:
            Time in seconds in the processed audio timeline.
        mappings:
            List of TimestampMapping objects produced during preprocessing.

        Returns
        -------
        float
            Corresponding time in the original audio timeline if mapping
            exists; otherwise returns processed_time unchanged.
        """
        if not mappings:
            return processed_time

        for mapping in mappings:
            if mapping.processed_start <= processed_time <= mapping.processed_end:
                ratio = (processed_time - mapping.processed_start) / \
                        (mapping.processed_end - mapping.processed_start + 1e-10)
                original_time = mapping.original_start + \
                               ratio * (mapping.original_end - mapping.original_start)
                return original_time

        return processed_time

    def _release_audio_models(self) -> None:
        """Release ASR and diarization models before LLM processing."""
        for component_name, component in (("transcriber", self.transcriber), ("diarizer", self.diarizer)):
            unload = getattr(component, "unload_model", None)
            if unload:
                logger.info(f"Unloading {component_name} to free VRAM for LLM...")
                unload()
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Unable to clear CUDA cache after audio model unload", exc_info=True)

    def _compose_timestamp_mappings(
        self, outer: List[TimestampMapping], inner: List[TimestampMapping]
    ) -> List[TimestampMapping]:
        """Compose processed→intermediate and intermediate→original mappings."""
        composed: List[TimestampMapping] = []
        for inner_mapping in inner:
            inner_duration = inner_mapping.original_end - inner_mapping.original_start
            if inner_duration <= 0:
                continue
            for outer_mapping in outer:
                start = max(inner_mapping.original_start, outer_mapping.processed_start)
                end = min(inner_mapping.original_end, outer_mapping.processed_end)
                if end <= start:
                    continue
                processed_duration = inner_mapping.processed_end - inner_mapping.processed_start
                processed_start = inner_mapping.processed_start + (start - inner_mapping.original_start) / inner_duration * processed_duration
                processed_end = inner_mapping.processed_start + (end - inner_mapping.original_start) / inner_duration * processed_duration
                outer_duration = outer_mapping.processed_end - outer_mapping.processed_start
                original_start = outer_mapping.original_start + (start - outer_mapping.processed_start) / outer_duration * (outer_mapping.original_end - outer_mapping.original_start)
                original_end = outer_mapping.original_start + (end - outer_mapping.processed_start) / outer_duration * (outer_mapping.original_end - outer_mapping.original_start)
                composed.append(TimestampMapping(processed_start, processed_end, original_start, original_end))
        return composed

    def _align_transcription_with_speakers(
        self,
        transcription_segments: List[Dict],
        diarization_segments: List[DiarizationSegment]
    ) -> List[Dict]:
        """Align transcription segments with diarization speaker labels.

        Parameters
        ----------
        transcription_segments:
            List of dicts produced by the transcriber with 'start' and 'end'.
        diarization_segments:
            List of DiarizationSegment instances returned by the diarizer.

        Returns
        -------
        List[Dict]
            List of aligned segments containing speaker labels.
        """
        aligned = []

        for seg in transcription_segments:
            start = seg["start"]
            end = seg["end"]
            text = seg.get("text", "").strip()

            if not text:
                continue

            speaker = "Unknown"
            max_overlap = 0

            for diar_seg in diarization_segments:
                overlap_start = max(start, diar_seg.start)
                overlap_end = min(end, diar_seg.end)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    speaker = diar_seg.speaker

            aligned.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text
            })

        return aligned

    def run(self, input_file: Optional[str] = None) -> PipelineResult:
        """Execute the full audio processing pipeline.

        Parameters
        ----------
        input_file:
            Optional path to a single media file to process. If None the
            pipeline will process the first discovered file in the configured
            media directory.

        Returns
        -------
        PipelineResult
            Result object containing success flag, output path, segments and
            optional error information.
        """
        try:
            # Step 1: Find media file
            if input_file:
                media_file, is_video = self.media.find_specific_file(input_file)
            else:
                media_file, is_video = self.media.find_media_file()

            base = Path(media_file).stem
            logger.info(f"Processing: {media_file}")

            # Step 2: Convert to WAV
            ext = Path(media_file).suffix.lower()
            if is_video or ext != '.wav':
                wav = self.media.convert_to_wav(media_file)
            else:
                wav = media_file

            # Step 3: Preprocess
            all_mappings: List[TimestampMapping] = []

            # Noise reduction
            if self.config.noise_reduction.enabled:
                logger.info("Reducing noise...")
                denoised = self.preprocessor.reduce_stationary_noise(wav)
            else:
                denoised = wav

            # Vocal separation
            if self.config.vocal_separation.enabled or self.config.vocal_separation.auto_detect:
                logger.info("Checking if vocal separation needed...")
                vocals = self.separator.extract_vocals(denoised)
            else:
                vocals = denoised

            # Normalization
            logger.info("Normalizing audio...")
            norm = self.preprocessor.normalize_audio(vocals)
            loudnorm = self.preprocessor.normalize_loudness(norm)

            # Silence removal
            if self.config.preserve_timestamps:
                logger.info("Removing silence (preserving timestamps)...")
                silence_removed, silence_mappings = self.preprocessor.remove_silence(
                    loudnorm, preserve_timestamps=True
                )
                all_mappings.extend(silence_mappings)
            else:
                silence_removed, _ = self.preprocessor.remove_silence(loudnorm)

            # Step 4: VAD
            if self.config.vad.enabled:
                logger.info(f"Applying VAD ({self.config.vad.provider})...")
                voiced_wav, vad_mappings = self.vad.filter_voice(
                    silence_removed, self.results_dir
                )
                if self.config.preserve_timestamps:
                    all_mappings = self._compose_timestamp_mappings(all_mappings, vad_mappings)
            else:
                voiced_wav = silence_removed

            # Step 5: Transcribe
            logger.info(f"Transcribing ({self.config.transcription.backend})...")
            transcription_result = self.transcriber.transcribe(voiced_wav)
            raw_segments = transcription_result.get("segments", [])
            logger.info(f"✓ Transcribed {len(raw_segments)} segments")

            # Step 6: Diarize
            if self.config.diarization.enabled:
                logger.info("Diarizing speakers...")
                diarization_segments = self.diarizer.diarize(
                    voiced_wav,
                    min_speakers=self.config.diarization.min_speakers,
                    max_speakers=self.config.diarization.max_speakers
                )
            else:
                diarization_segments = []

            # Step 7: Align
            logger.info("Aligning transcription with speakers...")
            aligned_segments = self._align_transcription_with_speakers(
                raw_segments, diarization_segments
            )

            # Step 8: Map timestamps
            if self.config.preserve_timestamps and all_mappings:
                logger.info("Mapping timestamps to original audio...")
                for seg in aligned_segments:
                    seg["original_start"] = self._map_timestamp_to_original(
                        seg["start"], all_mappings
                    )
                    seg["original_end"] = self._map_timestamp_to_original(
                        seg["end"], all_mappings
                    )

            # Step 9: Remove redundancies
            logger.info("Removing redundant segments...")
            final_segments = self.redundancy.remove(aligned_segments)
            logger.info(f"✓ Final: {len(final_segments)} segments")

            # Step 10: Merge short segments if needed
            if self.config.segment_merging.enabled:
                logger.info("Merging short segments...")
                merger = SegmentMerger(
                    max_gap_s=self.config.segment_merging.max_gap_s
                )
                final_segments = merger.merge(final_segments)

            # Step 11: LLM Post-Processing
            # Unload transcription and diarization models first to free VRAM
            # before loading the LLM (critical on GPUs with ≤ 8 GB VRAM).
            llm_analysis = None
            if self.config.llm.enabled:
                if self.llm_processor and hasattr(self.llm_processor, "unload_model"):
                    self.llm_processor.unload_model()
                self.llm_processor = None
                if hasattr(self.transcriber, 'unload_model'):
                    logger.info("Unloading transcriber to free VRAM for LLM...")
                    self.transcriber.unload_model()
                if hasattr(self.diarizer, 'unload_model'):
                    logger.info("Unloading diarizer to free VRAM for LLM...")
                    self.diarizer.unload_model()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        import gc; gc.collect()

                    from .post_processing_hybrid import HybridLLMPostProcessor
                    self.llm_processor = HybridLLMPostProcessor(
                        model=self.config.llm.openai_model,
                        ollama_host=self.config.llm.ollama_host,
                        ollama_model=self.config.llm.ollama_model,
                        use_ollama=self.config.llm.use_ollama,
                        use_openai=self.config.llm.use_openai,
                        ollama_num_ctx=self.config.llm.ollama_num_ctx,
                        ollama_keep_alive=self.config.llm.ollama_keep_alive,
                        local_model=self.config.llm.local_model,
                        device=self.config.llm.device,
                        max_length=self.config.llm.max_length,
                        temperature=self.config.llm.temperature,
                        lazy_load=True,
                    )
                    info = self.llm_processor.get_backend_info()
                    logger.info(f"✓ LLM initialized: {info['backend']} ({info['model']})")
                except Exception as e:
                    logger.warning(f"LLM post-processing disabled: {e}")
                    self.llm_processor = None

            if self.llm_processor:
                try:
                    logger.info("Analyzing with LLM...")
                    full_text = " ".join([s["text"] for s in final_segments])
                    llm_analysis = self.llm_processor.process(full_text)

                    if "error" not in llm_analysis:
                        logger.info("✓ LLM analysis complete")
                        logger.info(f"  Summary: {llm_analysis['summary'][:80]}...")
                        logger.info(f"  Topics: {len(llm_analysis['topics'])}")
                        logger.info(f"  Actions: {len(llm_analysis['action_items'])}")
                    else:
                        logger.warning(f"LLM analysis failed: {llm_analysis['error']}")

                except Exception as e:
                    logger.warning(f"LLM processing failed: {e}")
                    llm_analysis = {"error": str(e)}

            # Step 11: Save results
            output_data = {
                "metadata": {
                    "source_file": str(media_file),
                    "config": {
                        "model": self.config.transcription.model,
                        "language": self.config.transcription.language,
                        "vad_provider": self.config.vad.provider,
                        "transcription_backend": self.config.transcription.backend,
                    }
                },
                "segments": final_segments
            }

            if llm_analysis and "error" not in llm_analysis:
                output_data["llm_analysis"] = llm_analysis

            out_path = os.path.join(self.results_dir, f"{base}_transcription.json")
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as existing_file:
                    existing_source = json.load(existing_file).get("metadata", {}).get("source_file")
                if existing_source != str(media_file):
                    import hashlib
                    source_hash = hashlib.sha256(str(Path(media_file).resolve()).encode()).hexdigest()[:8]
                    out_path = os.path.join(self.results_dir, f"{base}_{source_hash}_transcription.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✓ Saved transcription: {out_path}")

            return PipelineResult(
                success=True,
                input_file=str(media_file),
                output_file=out_path,
                segments=final_segments,
                llm_analysis=llm_analysis,
                metadata={
                    "model": self.config.transcription.model,
                    "backend": self.config.transcription.backend,
                    "vad": self.config.vad.provider,
                    "llm_enabled": self.config.llm.enabled
                }
            )

        except MediaNotFoundError as e:
            logger.error(f"Media not found: {e}")
            return PipelineResult(
                success=False,
                input_file=str(input_file) if input_file else "",
                output_file=None,
                segments=[],
                error=str(e)
            )

        except AudioPipelineError as e:
            logger.error(f"Pipeline error: {e}")
            return PipelineResult(
                success=False,
                input_file=str(input_file) if input_file else "",
                output_file=None,
                segments=[],
                error=str(e)
            )

        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return PipelineResult(
                success=False,
                input_file=str(input_file) if input_file else "",
                output_file=None,
                segments=[],
                error=f"Unexpected error: {e}"
            )

    def run_transcription_only(self, input_wav: str) -> PipelineResult:
        """Run transcription only on a pre-processed WAV file."""
        try:
            result = self.transcriber.transcribe(input_wav)
            segments = result.get("segments", [])

            return PipelineResult(
                success=True,
                input_file=input_wav,
                output_file=None,
                segments=segments
            )
        except Exception as e:
            return PipelineResult(
                success=False,
                input_file=input_wav,
                output_file=None,
                segments=[],
                error=str(e)
            )

    def cleanup(self, remove_temp: bool = True, clear_checkpoints: bool = False) -> None:
        """Cleanup temporary files and unload models."""
        import shutil

        logger.info("Cleaning up...")

        # Unload models
        if hasattr(self.transcriber, 'unload_model'):
            self.transcriber.unload_model()
        if hasattr(self.diarizer, 'unload_model'):
            self.diarizer.unload_model()

        # Clear checkpoint cache only when explicitly requested. Checkpoints
        # live outside the temporary directory so they can support resuming.
        if clear_checkpoints and self.checkpoint_manager:
            self.checkpoint_manager.clear()

        # Remove temp directory
        if remove_temp and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"✓ Cleaned up temp directory: {self.temp_dir}")

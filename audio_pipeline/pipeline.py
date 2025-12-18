"""
Audio Pipeline Orchestrator - FULLY INTEGRATED VERSION

Coordinates all steps of the audio processing and transcription pipeline.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

from .protocols import (
    MediaHandlerProtocol,
    PreprocessorProtocol,
    VocalSeparatorProtocol,
    VADProtocol,
    TranscriberProtocol,
    DiarizerProtocol,
    RedundancyRemoverProtocol,
    TranscriptionSegment,
    DiarizationSegment,
    TimestampMapping
)
from .media_handler import MediaHandler
from .preprocessor import AudioPreprocessor
from .separator import VocalSeparator, NoOpVocalSeparator
from .vad import VADFilter, SileroVADFilter, NoOpVADFilter
from .transcriber import WhisperTranscriber, FasterWhisperTranscriber
from .diarizer import SpeakerDiarizer, NoOpDiarizer
from .redundancy import RedundancyRemover, NoOpRedundancyRemover
from .config import PipelineConfig, get_default_config
from .exceptions import AudioPipelineError, MediaNotFoundError
from .utils import CheckpointManager, ensure_directory

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
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
    """
    Coordinates all steps of the audio processing and transcription pipeline.
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
        self.config = config or get_default_config()
        self.config.validate()

        # Setup directories
        self.media_dir = ensure_directory(self.config.media_dir)
        self.temp_dir = ensure_directory(self.config.temp_dir)
        self.results_dir = ensure_directory(self.config.results_dir)

        # Setup checkpoint manager
        self.checkpoint_manager = None
        if self.config.checkpoint_enabled:
            self.checkpoint_manager = CheckpointManager(self.temp_dir)

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

        # LLM Post-Processor - CORRECTED INITIALIZATION
        self.llm_processor = None
        if self.config.llm.enabled:
            try:
                from .post_processing_hybrid import HybridLLMPostProcessor

                self.llm_processor = HybridLLMPostProcessor(
                    model=self.config.llm.openai_model,  # CORRETO: openai_model
                    local_model=self.config.llm.local_model,  # CORRETO: local_model
                    device=self.config.llm.device,  # CORRETO: device
                    max_length=self.config.llm.max_length,  # CORRETO: max_length
                    temperature=self.config.llm.temperature,  # CORRETO: temperature
                    force_local=not self.config.llm.use_openai  # CORRETO: inverte use_openai
                )

                # Log backend info
                info = self.llm_processor.get_backend_info()
                logger.info(f"✓ LLM initialized: {info['backend']} ({info['model']})")

            except ImportError as e:
                logger.warning(f"LLM post-processing disabled: {e}")
                logger.warning("Install with: pip install transformers torch openai instructor")
                self.llm_processor = None
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                self.llm_processor = None

        # Timestamp mappings
        self._timestamp_mappings: List[TimestampMapping] = []

    def _map_timestamp_to_original(
        self,
        processed_time: float,
        mappings: List[TimestampMapping]
    ) -> float:
        """Map timestamp from processed audio back to original."""
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

    def _align_transcription_with_speakers(
        self,
        transcription_segments: List[Dict],
        diarization_segments: List[DiarizationSegment]
    ) -> List[Dict]:
        """Align transcription segments with speaker labels."""
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
        """Execute the full audio processing pipeline."""
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
                    all_mappings.extend(vad_mappings)
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

            # Step 10: LLM Post-Processing
            llm_analysis = None
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

    def cleanup(self) -> None:
        """Cleanup temporary files and unload models."""
        import shutil

        logger.info("Cleaning up...")

        # Unload models
        if hasattr(self.transcriber, 'unload_model'):
            self.transcriber.unload_model()
        if hasattr(self.diarizer, 'unload_model'):
            self.diarizer.unload_model()

        # Clear checkpoint cache
        if self.checkpoint_manager:
            self.checkpoint_manager.clear()

        # Remove temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"✓ Cleaned up temp directory: {self.temp_dir}")
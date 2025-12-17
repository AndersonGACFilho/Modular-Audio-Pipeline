"""
Audio Pipeline Orchestrator.

Coordinates all steps of the audio processing and transcription pipeline.
Supports dependency injection, checkpoint/resume, and timestamp preservation.
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
from .vad import VADFilter, NoOpVADFilter
from .transcriber import WhisperTranscriber
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
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AudioPipeline:
    """
    Coordinates all steps of the audio processing and transcription pipeline.
    
    Features:
    - Dependency injection for all components
    - Checkpoint/resume support for long files
    - Timestamp preservation for mapping back to original
    - Configurable pipeline steps
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
        """
        Initialize AudioPipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
            media_handler: Media file handler (created from config if None)
            preprocessor: Audio preprocessor (created from config if None)
            separator: Vocal separator (created from config if None)
            vad: Voice activity detector (created from config if None)
            transcriber: Speech transcriber (created from config if None)
            diarizer: Speaker diarizer (created from config if None)
            redundancy_remover: Redundancy filter (created from config if None)
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
            self.checkpoint_manager = CheckpointManager(self.temp_dir)
        
        # Initialize components (dependency injection or create from config)
        self.media = media_handler or MediaHandler.from_config(self.config)
        self.preprocessor = preprocessor or AudioPreprocessor.from_config(self.config)
        
        # Separator - use NoOp if disabled
        if separator:
            self.separator = separator
        elif self.config.vocal_separation.enabled:
            self.separator = VocalSeparator.from_config(self.config, self.checkpoint_manager)
        else:
            self.separator = NoOpVocalSeparator()
        
        # VAD - use NoOp if disabled
        if vad:
            self.vad = vad
        elif self.config.vad.enabled:
            self.vad = VADFilter.from_config(self.config)
        else:
            self.vad = NoOpVADFilter()
        
        # Transcriber
        self.transcriber = transcriber or WhisperTranscriber.from_config(self.config)
        
        # Diarizer - use NoOp if disabled
        if diarizer:
            self.diarizer = diarizer
        elif self.config.diarization.enabled:
            self.diarizer = SpeakerDiarizer.from_config(self.config)
        else:
            self.diarizer = NoOpDiarizer()
        
        # Redundancy remover - use NoOp if disabled
        if redundancy_remover:
            self.redundancy = redundancy_remover
        elif self.config.redundancy.enabled:
            self.redundancy = RedundancyRemover.from_config(self.config)
        else:
            self.redundancy = NoOpRedundancyRemover()
        
        # Timestamp mappings for tracing back to original
        self._timestamp_mappings: List[TimestampMapping] = []
    
    def _map_timestamp_to_original(
        self,
        processed_time: float,
        mappings: List[TimestampMapping]
    ) -> float:
        """
        Map a timestamp from processed audio back to original audio.
        
        Args:
            processed_time: Time in processed audio
            mappings: List of timestamp mappings
            
        Returns:
            Corresponding time in original audio
        """
        if not mappings:
            return processed_time
        
        for mapping in mappings:
            if mapping.processed_start <= processed_time <= mapping.processed_end:
                # Linear interpolation within segment
                ratio = (processed_time - mapping.processed_start) / \
                        (mapping.processed_end - mapping.processed_start + 1e-10)
                original_time = mapping.original_start + \
                               ratio * (mapping.original_end - mapping.original_start)
                return original_time
        
        # If not found in mappings, return as-is
        return processed_time
    
    def _align_transcription_with_speakers(
        self,
        transcription_segments: List[Dict],
        diarization_segments: List[DiarizationSegment]
    ) -> List[Dict]:
        """
        Align transcription segments with speaker labels from diarization.
        
        Uses overlap duration to assign the most likely speaker to each segment.
        """
        aligned = []
        
        for seg in transcription_segments:
            start = seg["start"]
            end = seg["end"]
            text = seg.get("text", "").strip()
            
            if not text:
                continue
            
            # Find overlapping diarization segments
            speaker = "Unknown"
            max_overlap = 0
            
            for diar_seg in diarization_segments:
                # Calculate overlap
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
        """
        Execute the full audio processing pipeline.
        
        Args:
            input_file: Specific file to process (or None to auto-discover)
            
        Returns:
            PipelineResult with success status, output file, and segments
        """
        try:
            # Step 1: Find media file
            if input_file:
                media_file, is_video = self.media.find_specific_file(input_file)
            else:
                media_file, is_video = self.media.find_media_file()
            
            base = Path(media_file).stem
            logger.info(f"Processing: {media_file}")
            
            # Step 2: Convert to WAV if needed
            ext = Path(media_file).suffix.lower()
            if is_video or ext != '.wav':
                wav = self.media.convert_to_wav(media_file)
            else:
                wav = media_file
            
            # Step 3: Preprocess
            all_mappings: List[TimestampMapping] = []
            
            # Noise reduction
            if self.config.noise_reduction.enabled:
                denoised = self.preprocessor.reduce_stationary_noise(wav)
            else:
                denoised = wav
            
            # Vocal separation (auto-detects if needed when enabled)
            if self.config.vocal_separation.enabled or self.config.vocal_separation.auto_detect:
                vocals = self.separator.extract_vocals(denoised)
            else:
                vocals = denoised
            
            # Normalization
            norm = self.preprocessor.normalize_audio(vocals)
            loudnorm = self.preprocessor.normalize_loudness(norm)
            
            # Silence removal (with timestamp preservation)
            if self.config.preserve_timestamps:
                silence_removed, silence_mappings = self.preprocessor.remove_silence(
                    loudnorm, preserve_timestamps=True
                )
                all_mappings.extend(silence_mappings)
            else:
                silence_removed, _ = self.preprocessor.remove_silence(loudnorm)
            
            # Step 4: VAD (with timestamp preservation)
            if self.config.vad.enabled:
                voiced_wav, vad_mappings = self.vad.filter_voice(
                    silence_removed, self.results_dir
                )
                if self.config.preserve_timestamps:
                    all_mappings.extend(vad_mappings)
            else:
                voiced_wav = silence_removed
            
            # Step 5: Transcribe
            transcription_result = self.transcriber.transcribe(voiced_wav)
            raw_segments = transcription_result.get("segments", [])
            
            # Step 6: Diarize
            if self.config.diarization.enabled:
                diarization_segments = self.diarizer.diarize(
                    voiced_wav,
                    min_speakers=self.config.diarization.min_speakers,
                    max_speakers=self.config.diarization.max_speakers
                )
            else:
                diarization_segments = []
            
            # Step 7: Align transcription with speakers
            aligned_segments = self._align_transcription_with_speakers(
                raw_segments, diarization_segments
            )
            
            # Step 8: Map timestamps back to original (if enabled)
            if self.config.preserve_timestamps and all_mappings:
                for seg in aligned_segments:
                    seg["original_start"] = self._map_timestamp_to_original(
                        seg["start"], all_mappings
                    )
                    seg["original_end"] = self._map_timestamp_to_original(
                        seg["end"], all_mappings
                    )
            
            # Step 9: Remove redundancies
            final_segments = self.redundancy.remove(aligned_segments)
            
            # Step 10: Save results
            output_data = {
                "metadata": {
                    "source_file": str(media_file),
                    "config": {
                        "model": self.config.transcription.model,
                        "language": self.config.transcription.language,
                    }
                },
                "segments": final_segments
            }
            
            out_path = os.path.join(self.results_dir, f"{base}_transcription.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved transcription: {out_path}")
            
            return PipelineResult(
                success=True,
                input_file=str(media_file),
                output_file=out_path,
                segments=final_segments,
                metadata={"model": self.config.transcription.model}
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
        """
        Run transcription only on a pre-processed WAV file.
        
        Useful for testing or when preprocessing was done separately.
        """
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
        
        # Unload models to free memory
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
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")

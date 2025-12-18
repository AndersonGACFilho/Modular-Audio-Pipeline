"""
audio_pipeline.diarizer

Speaker diarization utilities for the audio pipeline.

Provides SpeakerDiarizer and NoOpDiarizer implementations that conform to
DiarizerProtocol. Docstrings use pydoc style to support Sphinx/pydoc.
"""

import os
import logging
from typing import List, Optional

import torch

from .protocols import DiarizerProtocol, DiarizationSegment
from .exceptions import DiarizationError, ModelLoadError
from .config import PipelineConfig, RetryConfig
from .utils import retry_with_backoff

logger = logging.getLogger(__name__)

__all__ = ["SpeakerDiarizer", "NoOpDiarizer"]


class SpeakerDiarizer(DiarizerProtocol):
    """
    Performs speaker diarization using pyannote.audio.
    
    Features:
    - Lazy model loading
    - GPU support with automatic device selection
    - Configurable speaker count constraints
    """
    
    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-3.1",
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        lazy_load: bool = True
    ):
        """
        Initialize SpeakerDiarizer.
        
        Args:
            model: Pyannote model name/path
            hf_token: Hugging Face authentication token
            device: Device to use ('cuda', 'cpu', or None for auto)
            lazy_load: If True, load model on first diarization
        """
        self.model_name = model
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        # Auto-select device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self._pipeline = None
        self._lazy_load = lazy_load
        # If loading fails due to incompatible runtime (e.g., NumPy 2.0), fall back to NoOp
        self._use_noop = False

        logger.info(f"Diarizer initialized (device: {self.device}, lazy_load: {lazy_load})")
        
        if not lazy_load:
            self.load_model()
    
    @classmethod
    def from_config(cls, config: PipelineConfig) -> "SpeakerDiarizer":
        """Create diarizer from pipeline configuration."""
        return cls(
            model=config.diarization.model,
            lazy_load=config.lazy_load_models
        )
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._pipeline is not None
    
    def  load_model(self) -> None:
        """
        Load the diarization pipeline.
        
        Raises:
            ModelLoadError: If loading fails
        """
        if self._pipeline is not None:
            logger.debug("Diarization pipeline already loaded")
            return
        
        if not self.hf_token:
            raise ModelLoadError(
                "Hugging Face token required for pyannote.audio",
                details="Set HF_TOKEN environment variable or pass hf_token parameter"
            )
        
        try:
            from pyannote.audio import Pipeline
            
            logger.info(f"Loading diarization pipeline: {self.model_name}...")
            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token
            ).to(self.device)

            if hasattr(self._pipeline, "segmentation_batch_size"):
                self._pipeline.segmentation_batch_size = 32

            if hasattr(self._pipeline, "embedding_batch_size"):
                self._pipeline.embedding_batch_size = 32
            
            logger.info(f"Diarization pipeline loaded on {self.device}")
            
        except ImportError:
            raise ModelLoadError(
                "pyannote.audio not installed",
                details="Install with: pip install pyannote.audio"
            )
        except Exception as e:
            msg = str(e)
            # Common failure on Windows with NumPy 2.0 where pyannote references np.NaN
            if 'np.NaN' in msg or 'NumPy 2.0' in msg or 'np.nan' in msg and 'was removed' in msg:
                logger.warning(
                    "Diarization loading failed due to NumPy compatibility issue: %s. "
                    "Attempting to monkeypatch numpy.NaN and retry once.", msg
                )
                try:
                    import numpy as np
                    # Provide backwards-compatible alias if missing
                    if not hasattr(np, 'NaN'):
                        setattr(np, 'NaN', np.nan)

                    from pyannote.audio import Pipeline
                    self._pipeline = Pipeline.from_pretrained(
                        self.model_name,
                        use_auth_token=self.hf_token
                    ).to(self.device)
                    logger.info("Diarization pipeline loaded after numpy monkeypatch")
                    return
                except Exception as e2:
                    logger.warning(f"Retry after numpy monkeypatch failed: {e2}")

            # If we reach here, cannot load pyannote - fall back to NoOp diarizer at runtime
            logger.error(f"Failed to load diarization model: {self.model_name}\nDetails: {msg}")
            logger.warning("Falling back to NoOp diarizer (single-speaker) so pipeline can continue.")
            self._use_noop = True
            self._pipeline = None


    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Diarization pipeline unloaded")
    
    @retry_with_backoff(
        config=RetryConfig(max_attempts=2, initial_delay_s=2.0),
        exceptions=(RuntimeError,)
    )
    def diarize(
        self,
        audio_path: str,
        min_speakers: int = 1,
        max_speakers: int = 5
    ) -> List[DiarizationSegment]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum expected number of speakers
            max_speakers: Maximum expected number of speakers
            
        Returns:
            List of DiarizationSegment with speaker, start, end times
            
        Raises:
            DiarizationError: If diarization fails
        """
        # If we previously determined we must use NoOp, delegate to NoOpDiarizer
        if getattr(self, '_use_noop', False):
            logger.warning("Using NoOpDiarizer fallback for diarization")
            return NoOpDiarizer().diarize(audio_path, min_speakers, max_speakers)

        if self._pipeline is None:
            self.load_model()
        
        logger.info(f"Diarizing: {audio_path} (speakers: {min_speakers}-{max_speakers})")
        
        try:
            diar_result = self._pipeline(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # Convert to our segment format
            # pyannote.audio returns different objects depending on version
            segments = self._extract_segments(diar_result)
            
            # Log summary
            speakers = set(s.speaker for s in segments)
            logger.info(f"Diarization complete: {len(segments)} segments, {len(speakers)} speakers")
            
            return segments
            
        except Exception as e:
            raise DiarizationError(
                f"Diarization failed for: {audio_path}",
                details=str(e)
            )
    
    def _extract_segments(self, diar_result) -> List[DiarizationSegment]:
        """
        Extract segments from pyannote diarization result.
        
        Handles both old and new pyannote.audio API versions.
        """
        segments = []
        
        # Try different iteration methods based on pyannote version
        try:
            # pyannote.audio 3.1+ with Annotation object
            if hasattr(diar_result, 'itertracks'):
                for turn, track, speaker in diar_result.itertracks(yield_label=True):
                    segments.append(DiarizationSegment(
                        speaker=speaker,
                        start=turn.start,
                        end=turn.end,
                        track=track
                    ))
            # Newer DiarizeOutput that's directly iterable
            elif hasattr(diar_result, '__iter__'):
                for item in diar_result:
                    if len(item) == 3:
                        turn, track, speaker = item
                        segments.append(DiarizationSegment(
                            speaker=speaker,
                            start=turn.start if hasattr(turn, 'start') else turn[0],
                            end=turn.end if hasattr(turn, 'end') else turn[1],
                            track=str(track)
                        ))
            else:
                raise DiarizationError(
                    "Unknown diarization result format",
                    details=f"Type: {type(diar_result)}"
                )
        except Exception as e:
            raise DiarizationError("Failed to parse diarization results", details=str(e))
        
        return segments
    
    def diarize_with_embedding(
        self,
        audio_path: str,
        min_speakers: int = 1,
        max_speakers: int = 5
    ) -> tuple:
        """
        Perform diarization and return speaker embeddings.
        
        Useful for speaker identification across multiple files.
        
        Returns:
            Tuple of (segments, embeddings_dict)
        """
        # This would require more complex implementation
        # For now, just return regular diarization
        segments = self.diarize(audio_path, min_speakers, max_speakers)
        return segments, {}


class NoOpDiarizer(DiarizerProtocol):
    """
    No-operation diarizer that assigns all speech to a single speaker.
    
    Used when diarization is disabled.
    """
    
    def __init__(self, default_speaker: str = "SPEAKER_00"):
        self.default_speaker = default_speaker
    
    def is_loaded(self) -> bool:
        return True
    
    def load_model(self) -> None:
        pass
    
    def diarize(
        self,
        audio_path: str,
        min_speakers: int = 1,
        max_speakers: int = 5
    ) -> List[DiarizationSegment]:
        """Return single segment covering entire audio."""
        import wave
        
        with wave.open(audio_path, 'rb') as wf:
            duration = wf.getnframes() / wf.getframerate()
        
        logger.debug(f"NoOp diarizer: assigning all to {self.default_speaker}")
        
        return [DiarizationSegment(
            speaker=self.default_speaker,
            start=0.0,
            end=duration,
            track="0"
        )]
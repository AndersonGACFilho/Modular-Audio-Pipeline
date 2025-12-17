"""
Protocol definitions for the Audio Pipeline.

Defines interfaces that allow swapping implementations without modifying
the pipeline orchestrator. Follows Dependency Inversion Principle.
"""

from typing import Protocol, List, Dict, Tuple, Optional, Any, runtime_checkable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscriptionSegment:
    """Represents a transcription segment with timing and speaker info."""
    text: str
    start: float
    end: float
    speaker: str = "Unknown"
    confidence: float = 1.0
    original_start: Optional[float] = None  # Original timestamp before processing
    original_end: Optional[float] = None


@dataclass
class DiarizationSegment:
    """Represents a speaker diarization segment."""
    speaker: str
    start: float
    end: float
    track: str = ""


@dataclass 
class TimestampMapping:
    """Maps processed timestamps back to original audio timestamps."""
    processed_start: float
    processed_end: float
    original_start: float
    original_end: float


@dataclass
class ProcessingResult:
    """Result of audio processing with timestamp mappings."""
    audio_path: str
    timestamp_mappings: List[TimestampMapping]
    

@runtime_checkable
class MediaHandlerProtocol(Protocol):
    """Protocol for media file discovery and conversion."""
    
    def find_media_file(self) -> Tuple[str, bool]:
        """Find media file in directory. Returns (path, is_video)."""
        ...
    
    def convert_to_wav(self, input_path: str) -> str:
        """Convert media to WAV format."""
        ...
    
    def validate_file(self, file_path: str) -> bool:
        """Validate that file exists and is readable."""
        ...


@runtime_checkable
class PreprocessorProtocol(Protocol):
    """Protocol for audio preprocessing operations."""
    
    def reduce_stationary_noise(
        self, 
        input_wav: str, 
        noise_sample_path: Optional[str] = None
    ) -> str:
        """Reduce stationary noise from audio."""
        ...
    
    def normalize_audio(self, input_wav: str) -> str:
        """Normalize audio levels."""
        ...
    
    def normalize_loudness(self, input_wav: str, target_lufs: float = -16.0) -> str:
        """Normalize loudness to target LUFS."""
        ...
    
    def remove_silence(self, input_wav: str) -> Tuple[str, List[TimestampMapping]]:
        """Remove silence while preserving timestamp mappings."""
        ...


@runtime_checkable
class VocalSeparatorProtocol(Protocol):
    """Protocol for vocal/music separation."""
    
    def extract_vocals(self, input_wav: str) -> str:
        """Extract vocals from audio."""
        ...
    
    def is_separation_needed(self, input_wav: str) -> bool:
        """Detect if audio has significant non-vocal content."""
        ...


@runtime_checkable
class VADProtocol(Protocol):
    """Protocol for Voice Activity Detection."""
    
    def filter_voice(self, input_wav: str, output_dir: str) -> Tuple[str, List[TimestampMapping]]:
        """Filter audio to keep only voiced segments with timestamp mappings."""
        ...
    
    def detect_speech_segments(self, input_wav: str) -> List[Tuple[float, float]]:
        """Detect speech segments without modifying audio."""
        ...


@runtime_checkable
class TranscriberProtocol(Protocol):
    """Protocol for speech-to-text transcription."""
    
    def transcribe(self, input_wav: str) -> Dict[str, Any]:
        """Transcribe audio to text with segments."""
        ...
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...
    
    def load_model(self) -> None:
        """Load the transcription model."""
        ...


@runtime_checkable
class DiarizerProtocol(Protocol):
    """Protocol for speaker diarization."""
    
    def diarize(
        self, 
        audio_path: str, 
        min_speakers: int = 2, 
        max_speakers: int = 5
    ) -> List[DiarizationSegment]:
        """Identify speaker turns in audio."""
        ...
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...
    
    def load_model(self) -> None:
        """Load the diarization model."""
        ...


@runtime_checkable
class RedundancyRemoverProtocol(Protocol):
    """Protocol for removing redundant transcription segments."""
    
    def remove(self, segments: List[Dict]) -> List[Dict]:
        """Remove redundant/duplicate segments."""
        ...
    
    def is_similar(self, a: str, b: str) -> bool:
        """Check if two texts are similar."""
        ...

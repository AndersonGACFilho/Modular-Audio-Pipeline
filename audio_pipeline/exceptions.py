"""
Custom exceptions for the Audio Pipeline.

Provides specific exception types for better error handling and debugging.
"""

from typing import Optional


class AudioPipelineError(Exception):
    """Base exception for all pipeline errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class MediaNotFoundError(AudioPipelineError):
    """Raised when no valid media file is found."""
    pass


class MediaConversionError(AudioPipelineError):
    """Raised when media conversion fails."""
    pass


class AudioProcessingError(AudioPipelineError):
    """Raised when audio preprocessing fails."""
    pass


class VocalSeparationError(AudioPipelineError):
    """Raised when vocal separation fails."""
    pass


class TranscriptionError(AudioPipelineError):
    """Raised when transcription fails."""
    pass


class DiarizationError(AudioPipelineError):
    """Raised when speaker diarization fails."""
    pass


class VADError(AudioPipelineError):
    """Raised when Voice Activity Detection fails."""
    pass


class ConfigurationError(AudioPipelineError):
    """Raised when configuration is invalid."""
    pass


class ModelLoadError(AudioPipelineError):
    """Raised when a model fails to load."""
    pass


class FileValidationError(AudioPipelineError):
    """Raised when file validation fails."""
    pass

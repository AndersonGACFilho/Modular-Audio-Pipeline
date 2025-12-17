"""
Audio Pipeline - A modular audio processing and transcription pipeline.

Features:
- Media conversion and validation
- Noise reduction with auto-detection
- Vocal separation (optional, for audio with music)
- Voice Activity Detection (VAD)
- Speech-to-text transcription with Whisper
- Speaker diarization with pyannote.audio
- Redundancy filtering
- Timestamp preservation

Example usage:
    from audio_pipeline import AudioPipeline, PipelineConfig

    config = PipelineConfig(media_dir="./recordings")
    pipeline = AudioPipeline(config)
    result = pipeline.run()

    if result.success:
        print(f"Transcription saved to: {result.output_file}")
"""

from .config import (
    PipelineConfig,
    AudioConfig,
    VADConfig,
    NoiseReductionConfig,
    VocalSeparationConfig,
    TranscriptionConfig,
    DiarizationConfig,
    RedundancyConfig,
    RetryConfig,
    DEFAULT_PROMPTS,
    get_default_config,
)

from .pipeline import AudioPipeline, PipelineResult

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
    TimestampMapping,
)

from .exceptions import (
    AudioPipelineError,
    MediaNotFoundError,
    MediaConversionError,
    AudioProcessingError,
    VocalSeparationError,
    TranscriptionError,
    DiarizationError,
    VADError,
    ConfigurationError,
    ModelLoadError,
    FileValidationError,
)

from .media_handler import MediaHandler
from .preprocessor import AudioPreprocessor
from .separator import VocalSeparator, NoOpVocalSeparator
from .vad import VADFilter, NoOpVADFilter
from .transcriber import WhisperTranscriber, FasterWhisperTranscriber
from .diarizer import SpeakerDiarizer, NoOpDiarizer
from .redundancy import RedundancyRemover, NoOpRedundancyRemover

from .utils import (
    retry_with_backoff,
    validate_file,
    CheckpointManager,
    get_file_hash,
    ensure_directory,
    get_audio_duration,
    format_timestamp,
    parse_timestamp,
)

__version__ = "2.0.0"
__author__ = "Anderson GAC Filho"

__all__ = [
    # Main classes
    "AudioPipeline",
    "PipelineResult",
    
    # Configuration
    "PipelineConfig",
    "AudioConfig",
    "VADConfig",
    "NoiseReductionConfig",
    "VocalSeparationConfig",
    "TranscriptionConfig",
    "DiarizationConfig",
    "RedundancyConfig",
    "RetryConfig",
    "DEFAULT_PROMPTS",
    "get_default_config",
    
    # Protocols
    "MediaHandlerProtocol",
    "PreprocessorProtocol",
    "VocalSeparatorProtocol",
    "VADProtocol",
    "TranscriberProtocol",
    "DiarizerProtocol",
    "RedundancyRemoverProtocol",
    
    # Data classes
    "TranscriptionSegment",
    "DiarizationSegment",
    "TimestampMapping",
    
    # Exceptions
    "AudioPipelineError",
    "MediaNotFoundError",
    "MediaConversionError",
    "AudioProcessingError",
    "VocalSeparationError",
    "TranscriptionError",
    "DiarizationError",
    "VADError",
    "ConfigurationError",
    "ModelLoadError",
    "FileValidationError",
    
    # Implementations
    "MediaHandler",
    "AudioPreprocessor",
    "VocalSeparator",
    "NoOpVocalSeparator",
    "VADFilter",
    "NoOpVADFilter",
    "WhisperTranscriber",
    "FasterWhisperTranscriber",
    "SpeakerDiarizer",
    "NoOpDiarizer",
    "RedundancyRemover",
    "NoOpRedundancyRemover",
    
    # Utilities
    "retry_with_backoff",
    "validate_file",
    "CheckpointManager",
    "get_file_hash",
    "ensure_directory",
    "get_audio_duration",
    "format_timestamp",
    "parse_timestamp",
]

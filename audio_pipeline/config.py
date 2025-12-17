"""
Configuration management for the Audio Pipeline.

Provides typed configuration with validation and environment variable support.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import os
import json
import logging

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    mode: int = 1  # 0-3, higher = more aggressive
    frame_duration_ms: int = 30
    padding_duration_ms: int = 500
    start_threshold: float = 0.5
    stop_threshold: float = 0.9
    enabled: bool = True


@dataclass  
class NoiseReductionConfig:
    """Noise reduction configuration."""
    enabled: bool = True
    auto_detect_noise: bool = True  # Auto-detect noise segments vs fixed profile
    noise_sample_duration_s: float = 0.5
    noise_sample_path: Optional[str] = None  # Optional external noise sample


@dataclass
class VocalSeparationConfig:
    """Vocal separation configuration."""
    enabled: bool = False  # Disabled by default - only enable for music
    model: str = "htdemucs"
    chunk_minutes: float = 5.0
    auto_detect: bool = True  # Auto-detect if separation is needed


@dataclass
class TranscriptionConfig:
    """Transcription configuration."""
    model: str = "large-v3"
    language: str = "pt"
    task: str = "transcribe"
    temperature: float = 0.0
    beam_size: int = 5
    prompt: str = ""
    response_format: str = "verbose_json"


@dataclass
class DiarizationConfig:
    """Speaker diarization configuration."""
    enabled: bool = True
    min_speakers: int = 1
    max_speakers: int = 5
    model: str = "pyannote/speaker-diarization-3.1"


@dataclass
class RedundancyConfig:
    """Redundancy removal configuration."""
    enabled: bool = True
    similarity_threshold: float = 0.85


@dataclass
class RetryConfig:
    """Retry configuration for external calls."""
    max_attempts: int = 3
    initial_delay_s: float = 1.0
    exponential_backoff: bool = True
    max_delay_s: float = 30.0


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    
    # Paths
    media_dir: str = "./files"
    temp_dir: Optional[str] = None  # Auto-generated if not set
    results_dir: Optional[str] = None  # Auto-generated if not set
    
    # Sub-configurations
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    noise_reduction: NoiseReductionConfig = field(default_factory=NoiseReductionConfig)
    vocal_separation: VocalSeparationConfig = field(default_factory=VocalSeparationConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    redundancy: RedundancyConfig = field(default_factory=RedundancyConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    
    # Processing options
    preserve_timestamps: bool = True  # Keep mapping to original timestamps
    subprocess_timeout_s: int = 600  # 10 minutes default timeout
    lazy_load_models: bool = True  # Load models only when needed
    checkpoint_enabled: bool = True  # Enable checkpoint/resume for long processes
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        self.media_dir = str(Path(self.media_dir).resolve())
        
        if self.temp_dir is None:
            self.temp_dir = str(Path(self.media_dir) / "temp")
        else:
            self.temp_dir = str(Path(self.temp_dir).resolve())
            
        if self.results_dir is None:
            self.results_dir = str(Path(self.media_dir) / "results")
        else:
            self.results_dir = str(Path(self.results_dir).resolve())
    
    def validate(self) -> None:
        """Validate configuration values."""
        errors = []
        
        # Audio validation
        if self.audio.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            errors.append(f"Invalid sample rate: {self.audio.sample_rate}")
        
        # VAD validation
        if not 0 <= self.vad.mode <= 3:
            errors.append(f"VAD mode must be 0-3, got: {self.vad.mode}")
        if self.vad.frame_duration_ms not in [10, 20, 30]:
            errors.append(f"VAD frame duration must be 10, 20, or 30ms")
        if not 0 <= self.vad.start_threshold <= 1:
            errors.append(f"VAD start threshold must be 0-1")
        if not 0 <= self.vad.stop_threshold <= 1:
            errors.append(f"VAD stop threshold must be 0-1")
            
        # Transcription validation
        valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"]
        if self.transcription.model not in valid_models:
            logger.warning(f"Unknown Whisper model: {self.transcription.model}")
            
        # Diarization validation
        if self.diarization.min_speakers > self.diarization.max_speakers:
            errors.append("min_speakers cannot be greater than max_speakers")
            
        # Redundancy validation
        if not 0 <= self.redundancy.similarity_threshold <= 1:
            errors.append("Similarity threshold must be 0-1")
        
        if errors:
            raise ConfigurationError(
                "Configuration validation failed",
                details="\n".join(errors)
            )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Simple fields
        for key in ["media_dir", "temp_dir", "results_dir", "preserve_timestamps", 
                    "subprocess_timeout_s", "lazy_load_models", "checkpoint_enabled"]:
            if key in data:
                setattr(config, key, data[key])
        
        # Nested configs
        if "audio" in data:
            config.audio = AudioConfig(**data["audio"])
        if "vad" in data:
            config.vad = VADConfig(**data["vad"])
        if "noise_reduction" in data:
            config.noise_reduction = NoiseReductionConfig(**data["noise_reduction"])
        if "vocal_separation" in data:
            config.vocal_separation = VocalSeparationConfig(**data["vocal_separation"])
        if "transcription" in data:
            config.transcription = TranscriptionConfig(**data["transcription"])
        if "diarization" in data:
            config.diarization = DiarizationConfig(**data["diarization"])
        if "redundancy" in data:
            config.redundancy = RedundancyConfig(**data["redundancy"])
        if "retry" in data:
            config.retry = RetryConfig(**data["retry"])
            
        config.__post_init__()
        return config
    
    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables
        if media_dir := os.getenv("AUDIO_PIPELINE_MEDIA_DIR"):
            config.media_dir = media_dir
        if model := os.getenv("AUDIO_PIPELINE_MODEL"):
            config.transcription.model = model
        if language := os.getenv("AUDIO_PIPELINE_LANGUAGE"):
            config.transcription.language = language
        if prompt := os.getenv("AUDIO_PIPELINE_PROMPT"):
            config.transcription.prompt = prompt
            
        config.__post_init__()
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# Default prompts for different use cases
DEFAULT_PROMPTS = {
    "pt_instructions": (
        "Esta gravação é meu chefe me dando instruções sobre tarefas de trabalho. "
        "Transcreva o áudio em português de forma precisa, mantendo a pontuação correta "
        "e indicando claramente quaisquer pausas ou hesitações. "
        "Formate a transcrição em parágrafos para melhor legibilidade."
    ),
    "pt_meeting": (
        "Esta é uma reunião de trabalho em português. "
        "Transcreva com precisão todas as falas, identificando diferentes interlocutores. "
        "Mantenha a pontuação correta e indique pausas com reticências."
    ),
    "pt_interview": (
        "Esta é uma entrevista em português. "
        "Transcreva com precisão as perguntas e respostas. "
        "Mantenha o tom e estilo de fala original."
    ),
    "en_general": (
        "Transcribe this audio accurately in English. "
        "Maintain proper punctuation and indicate pauses or hesitations. "
        "Format the transcription in paragraphs for readability."
    ),
    "en_technical": (
        "This is a technical discussion. "
        "Transcribe accurately, paying attention to technical terms and acronyms. "
        "Maintain proper punctuation."
    ),
}


def get_default_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    config = PipelineConfig()
    config.transcription.prompt = DEFAULT_PROMPTS["pt_instructions"]
    return config

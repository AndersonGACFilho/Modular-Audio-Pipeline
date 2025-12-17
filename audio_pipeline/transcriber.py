"""
Transcriber for the Audio Pipeline.

Wraps OpenAI Whisper for speech-to-text transcription.
Supports lazy model loading and configurable prompts.
"""

import logging
from typing import Dict, Any, Optional

from .protocols import TranscriberProtocol
from .exceptions import TranscriptionError, ModelLoadError
from .config import PipelineConfig, TranscriptionConfig
from .utils import retry_with_backoff, RetryConfig

logger = logging.getLogger(__name__)


class WhisperTranscriber(TranscriberProtocol):
    """
    Wraps OpenAI Whisper for transcription with lazy loading.
    
    Features:
    - Lazy model loading (load only when first transcription is requested)
    - Configurable prompts for different use cases
    - Retry support for robustness
    """
    
    # Available Whisper models and their approximate VRAM requirements
    MODEL_INFO = {
        "tiny": {"vram_gb": 1, "params": "39M"},
        "base": {"vram_gb": 1, "params": "74M"},
        "small": {"vram_gb": 2, "params": "244M"},
        "medium": {"vram_gb": 5, "params": "769M"},
        "large": {"vram_gb": 10, "params": "1550M"},
        "large-v2": {"vram_gb": 10, "params": "1550M"},
        "large-v3": {"vram_gb": 10, "params": "1550M"},
        "large-v3-turbo": {"vram_gb": 6, "params": "809M"},
    }
    
    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        language: str = "pt",
        prompt: str = "",
        task: str = "transcribe",
        temperature: float = 0.0,
        beam_size: int = 5,
        lazy_load: bool = True
    ):
        """
        Initialize WhisperTranscriber.
        
        Args:
            model_name: Whisper model name
            language: Language code for transcription
            prompt: Initial prompt to guide transcription
            task: Task type ('transcribe' or 'translate')
            temperature: Sampling temperature (0 = deterministic)
            beam_size: Beam size for beam search
            lazy_load: If True, load model on first transcription
        """
        self.model_name = model_name
        self.language = language
        self.prompt = prompt
        self.task = task
        self.temperature = temperature
        self.beam_size = beam_size
        
        self._model = None
        self._lazy_load = lazy_load
        
        if model_name not in self.MODEL_INFO:
            logger.warning(f"Unknown model: {model_name}. Proceeding anyway.")
        else:
            info = self.MODEL_INFO[model_name]
            logger.info(f"Whisper model: {model_name} ({info['params']} params, ~{info['vram_gb']}GB VRAM)")
        
        if not lazy_load:
            self.load_model()
    
    @classmethod
    def from_config(cls, config: PipelineConfig) -> "WhisperTranscriber":
        """Create transcriber from pipeline configuration."""
        return cls(
            model_name=config.transcription.model,
            language=config.transcription.language,
            prompt=config.transcription.prompt,
            task=config.transcription.task,
            temperature=config.transcription.temperature,
            beam_size=config.transcription.beam_size,
            lazy_load=config.lazy_load_models
        )
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
    
    def load_model(self) -> None:
        """
        Load the Whisper model.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        if self._model is not None:
            logger.debug("Model already loaded")
            return
        
        try:
            import whisper
            
            logger.info(f"Loading Whisper model: {self.model_name}...")
            self._model = whisper.load_model(self.model_name)
            logger.info(f"Whisper model loaded successfully")
            
        except ImportError:
            raise ModelLoadError(
                "OpenAI Whisper not installed",
                details="Install with: pip install openai-whisper"
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load Whisper model: {self.model_name}",
                details=str(e)
            )
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
            # Try to free GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info("Whisper model unloaded")
    
    @retry_with_backoff(
        config=RetryConfig(max_attempts=2, initial_delay_s=2.0),
        exceptions=(RuntimeError,)
    )
    def transcribe(self, input_wav: str) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            input_wav: Path to input WAV file
            
        Returns:
            Whisper result dict with 'text' and 'segments' keys
            
        Raises:
            TranscriptionError: If transcription fails
        """
        # Ensure model is loaded
        if self._model is None:
            self.load_model()
        
        logger.info(f"Transcribing: {input_wav}")
        
        try:
            result = self._model.transcribe(
                audio=input_wav,
                language=self.language,
                task=self.task,
                verbose=False,  # Reduce console spam
                temperature=self.temperature,
                beam_size=self.beam_size,
                initial_prompt=self.prompt if self.prompt else None,
                word_timestamps=True,  # Enable word-level timestamps
            )
            
            # Log summary
            num_segments = len(result.get('segments', []))
            text_length = len(result.get('text', ''))
            logger.info(f"Transcription complete: {num_segments} segments, {text_length} chars")
            
            return result
            
        except Exception as e:
            raise TranscriptionError(
                f"Transcription failed for: {input_wav}",
                details=str(e)
            )
    
    def transcribe_with_options(
        self,
        input_wav: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe with custom options, overriding defaults.
        
        Args:
            input_wav: Path to input WAV file
            **kwargs: Override transcription options
            
        Returns:
            Whisper result dict
        """
        if self._model is None:
            self.load_model()
        
        options = {
            'language': self.language,
            'task': self.task,
            'verbose': False,
            'temperature': self.temperature,
            'beam_size': self.beam_size,
            'initial_prompt': self.prompt if self.prompt else None,
        }
        options.update(kwargs)
        
        try:
            return self._model.transcribe(audio=input_wav, **options)
        except Exception as e:
            raise TranscriptionError(f"Transcription failed", details=str(e))


class FasterWhisperTranscriber(TranscriberProtocol):
    """
    Alternative transcriber using faster-whisper for better performance.
    
    Requires: pip install faster-whisper
    """
    
    def __init__(
        self,
        model_name: str = "large-v3",
        language: str = "pt",
        prompt: str = "",
        device: str = "auto",
        compute_type: str = "float16",
        lazy_load: bool = True
    ):
        self.model_name = model_name
        self.language = language
        self.prompt = prompt
        self.device = device
        self.compute_type = compute_type
        
        self._model = None
        self._lazy_load = lazy_load
        
        if not lazy_load:
            self.load_model()
    
    def is_loaded(self) -> bool:
        return self._model is not None
    
    def load_model(self) -> None:
        if self._model is not None:
            return
        
        try:
            from faster_whisper import WhisperModel
            
            logger.info(f"Loading faster-whisper model: {self.model_name}...")
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("faster-whisper model loaded")
            
        except ImportError:
            raise ModelLoadError(
                "faster-whisper not installed",
                details="Install with: pip install faster-whisper"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load model", details=str(e))
    
    def transcribe(self, input_wav: str) -> Dict[str, Any]:
        if self._model is None:
            self.load_model()
        
        try:
            segments, info = self._model.transcribe(
                input_wav,
                language=self.language,
                initial_prompt=self.prompt if self.prompt else None,
                beam_size=5,
                vad_filter=True,
            )
            
            # Convert to Whisper-compatible format
            result_segments = []
            full_text = []
            
            for seg in segments:
                result_segments.append({
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text,
                })
                full_text.append(seg.text)
            
            return {
                'text': ' '.join(full_text),
                'segments': result_segments,
                'language': info.language,
            }
            
        except Exception as e:
            raise TranscriptionError(f"Transcription failed", details=str(e))

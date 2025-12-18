"""
audio_pipeline.transcriber

High-level transcribers used by the audio processing pipeline.

This module provides two implementations that conform to the TranscriberProtocol:

- WhisperTranscriber: a wrapper around OpenAI's `whisper` package with lazy
  loading and retry support.
- FasterWhisperTranscriber: an optimized implementation backed by
  `faster-whisper` / CTranslate2 with GPU/CPU fallbacks.

All public classes and functions are documented using conventional pydoc
style so they can be consumed by pydoc or Sphinx autodoc.

Example usage:

    from audio_pipeline.transcriber import WhisperTranscriber

    transcriber = WhisperTranscriber(model_name="small", lazy_load=True)
    result = transcriber.transcribe("/path/to/audio.wav")
    print(result['text'])
"""

from typing import Dict, Any
import logging

import torch

from .protocols import TranscriberProtocol
from .exceptions import TranscriptionError, ModelLoadError
from .config import PipelineConfig, RetryConfig
from .utils import retry_with_backoff

logger = logging.getLogger(__name__)

__all__ = ["WhisperTranscriber", "FasterWhisperTranscriber"]

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

class WhisperTranscriber(TranscriberProtocol):
    """Wrapper around OpenAI's ``whisper`` package.

    This transcriber supports lazy model loading, configurable prompts, and a
    simple retry policy for transient errors. Methods follow pydoc conventions
    so tooling such as pydoc or Sphinx can extract structured documentation.

    Parameters
    ----------
    model_name : str
        Name of the whisper model to load (e.g. ``small``, ``large-v3-turbo``).
    language : str
        Language code for transcription (ISO 639-1 style, e.g. ``pt`` or ``en``).
    prompt : str
        Optional initial prompt to guide transcription.
    task : str
        Task type, one of ``transcribe`` or ``translate``.
    temperature : float
        Sampling temperature; 0.0 produces deterministic output.
    beam_size : int
        Beam search width used by the backend (if supported).
    lazy_load : bool
        Whether to delay model loading until the first transcription call.

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
    ) -> None:
        """Create a WhisperTranscriber instance.

        See class docstring for parameter descriptions.
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
        """Instantiate from a PipelineConfig object.

        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration object.

        Returns
        -------
        WhisperTranscriber
            Configured transcriber instance.
        """
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
        """Return True if the underlying model has been loaded.

        Returns
        -------
        bool
            Model loaded state.
        """
        return self._model is not None
    
    def load_model(self) -> None:
        """Load the Whisper model into memory.

        Raises
        ------
        ModelLoadError
            If the whisper package is not installed or model loading fails.
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
        """Unload the Whisper model and attempt to free GPU memory.

        This method attempts to delete the model object and call CUDA cache
        cleanup if torch is available and CUDA is used.
        """
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
        """Transcribe a WAV audio file to text.

        Parameters
        ----------
        input_wav : str
            Path to the input WAV file.

        Returns
        -------
        dict
            Backend-specific transcription result. Typically contains ``text``
            and ``segments`` keys.

        Raises
        ------
        TranscriptionError
            If the transcription fails.
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
        """Override default transcription options for a single call.

        Parameters
        ----------
        input_wav : str
            Path to the input WAV file.
        **kwargs
            Options passed directly to the backend transcribe method.

        Returns
        -------
        dict
            Backend-specific transcription result.
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
    """Faster inference transcriber using the ``faster-whisper`` implementation.

    This class attempts to load a CTranslate2-backed model and falls back to
    CPU/int8 when GPU initialization fails. If both CTranslate2 and CPU
    fallbacks fail, it will attempt to fallback to the standard OpenAI
    ``whisper`` implementation as a last resort.
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        language: str = "pt",
        lazy_load: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self._model = None

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not found. Reverting to CPU (int8).")
            self.device = "cpu"
            self.compute_type = "int8"

        if not lazy_load:
            self.load_model()

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "FasterWhisperTranscriber":
        """Create instance from PipelineConfig.

        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration object.

        Returns
        -------
        FasterWhisperTranscriber
            Configured transcriber instance.
        """
        return cls(
            model_name=config.transcription.model,
            device=config.transcription.device,
            compute_type=config.transcription.compute_type,
            language=config.transcription.language,
            lazy_load=config.lazy_load_models
        )

    def load_model(self) -> None:
        """Load the faster-whisper model.

        Raises
        ------
        ModelLoadError
            If faster-whisper is not installed or model loading fails on all
            fallback attempts.
        """
        if self._model is not None: return

        if WhisperModel is None:
            raise ModelLoadError("faster-whisper library not installed.")

        try:
            logger.info(f"Loading faster-whisper: {self.model_name} ({self.device})")
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
        except Exception as e:
            # Try to fallback to CPU if GPU load failed (common on misconfigured Windows/CUDA)
            logger.warning(
                f"Failed to load faster-whisper on device {self.device}: {e}. "
                "Attempting to fallback to CPU (int8) for inference."
            )

            # Attempt CPU fallback
            try:
                self.device = "cpu"
                self.compute_type = "int8"
                logger.info(f"Retrying faster-whisper load on CPU with compute_type={self.compute_type}")
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type
                )
                logger.info("Loaded faster-whisper on CPU successfully")
            except Exception as e2:
                # If CPU fallback fails, raise original error context
                raise ModelLoadError(f"Failed to load model on GPU and CPU fallback also failed: {e2}")

    def is_loaded(self) -> bool:
        """Return True when the model has been loaded.

        Returns
        -------
        bool
            True if loaded.
        """
        return self._model is not None

    def transcribe(self, input_wav: str) -> Dict[str, Any]:
        """Transcribe an audio file using the faster-whisper backend.

        Parameters
        ----------
        input_wav : str
            Path to the input audio file.

        Returns
        -------
        dict
            A dictionary with keys ``text``, ``segments``, and other metadata.

        Raises
        ------
        TranscriptionError
            If transcription fails after attempting fallbacks.
        """
        if self._model is None: self.load_model()

        try:
            logger.info(f"Transcribing (Optimized): {input_wav}")
            segments_gen, info = self._model.transcribe(
                input_wav,
                beam_size=self.beam_size,
                language=self.language,
                vad_filter=True, # Built-in Silero VAD
                word_timestamps=True
            )

            segments = list(segments_gen)
            processed_segments = []
            full_text = []

            for seg in segments:
                processed_segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "confidence": getattr(seg, 'avg_logprob', 0.0)
                })
                full_text.append(seg.text.strip())

            return {
                "text": " ".join(full_text),
                "segments": processed_segments,
                "language": info.language,
                "duration": info.duration
            }

        except Exception as e:
            # If CUDA-related DLL missing (common on Windows), attempt CPU fallback once
            msg = str(e).lower()
            if 'cublas' in msg or 'cublas64' in msg or 'cuda' in msg or 'dll' in msg:
                logger.warning(
                    f"Faster-whisper inference failed (likely CUDA/runtime issue): {e}. "
                    "Attempting CPU fallback and retrying inference."
                )
                try:
                    # Unload current model and force CPU/int8
                    try:
                        self.unload_model()
                    except Exception:
                        pass

                    self.device = 'cpu'
                    self.compute_type = 'int8'
                    self.load_model()

                    # Retry transcription once on CPU
                    logger.info("Retrying transcription on CPU (int8)...")
                    segments_gen, info = self._model.transcribe(
                        input_wav,
                        beam_size=self.beam_size,
                        language=self.language,
                        vad_filter=True,
                        word_timestamps=True
                    )

                    segments = list(segments_gen)
                    processed_segments = []
                    full_text = []

                    for seg in segments:
                        processed_segments.append({
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text.strip(),
                            "confidence": getattr(seg, 'avg_logprob', 0.0)
                        })
                        full_text.append(seg.text.strip())

                    return {
                        "text": " ".join(full_text),
                        "segments": processed_segments,
                        "language": info.language,
                        "duration": info.duration
                    }

                except Exception as e2:
                    # If CPU fallback still fails (common when CTranslate2/faster-whisper was installed with CUDA
                    # support but CUDA runtime is missing), attempt to fallback to the standard OpenAI Whisper
                    # implementation which runs on CPU without relying on cublas DLL.
                    logger.warning(
                        f"CPU fallback for faster-whisper failed: {e2}. "
                        "Attempting to fallback to OpenAI Whisper (CPU) as a last resort."
                    )

                    try:
                        # Choose a Whisper model compatible with openai-whisper (tiny, base, small, medium, large)
                        target_model = self.model_name if self.model_name in WhisperTranscriber.MODEL_INFO else "small"
                        # If the current model name isn't supported by OpenAI Whisper, default to a small CPU-friendly model

                        # Remove any "-v2", "-v3", "-v3-turbo" suffixes for compatibility
                        for suffix in ["-v2", "-v3", "-v3-turbo"]:
                            if target_model.endswith(suffix):
                                target_model = target_model.replace(suffix, "")

                        if target_model not in ["tiny", "base", "small", "medium", "large"]:
                            target_model = "small"

                        logger.info(f"Falling back to Whisper (CPU) model: {target_model}")
                        whisper_fallback = WhisperTranscriber(
                            model_name=target_model,
                            language=self.language,
                            lazy_load=False
                        )

                        # Perform transcription with the slower but more compatible backend
                        return whisper_fallback.transcribe(input_wav)
                    except Exception as e3:
                        raise TranscriptionError(
                            f"Transcription failed after CPU fallback and Whisper fallback: {e3}"
                        )

            # Not a CUDA/runtime issue or fallback didn't help
            raise TranscriptionError(f"Transcription failed: {e}")

    def unload_model(self) -> None:
         """Unload faster-whisper model to free memory."""
         if hasattr(self, '_model') and self._model is not None:
             try:
                 # Attempt to delete model and free GPU memory
                 del self._model
             except Exception:
                 pass
             self._model = None
             try:
                 import torch
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache()
             except Exception:
                 pass
             logger.info("FasterWhisper model unloaded")


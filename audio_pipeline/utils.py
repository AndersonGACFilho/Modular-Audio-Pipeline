"""
Utility functions for the Audio Pipeline.

Provides retry logic, file validation, checkpoint management, and common operations.
"""

import os
import time
import json
import hashlib
import functools
import logging
from pathlib import Path
from typing import TypeVar, Callable, Optional, Any, Dict, List
from dataclasses import dataclass, asdict

from .exceptions import FileValidationError, AudioPipelineError
from .config import RetryConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        config: Retry configuration
        exceptions: Tuple of exceptions to catch
        on_retry: Optional callback on retry (exception, attempt_number)
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = config.initial_delay_s
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
                        raise
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    logger.warning(
                        f"Attempt {attempt}/{config.max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    time.sleep(delay)
                    
                    if config.exponential_backoff:
                        delay = min(delay * 2, config.max_delay_s)
            
            raise last_exception  # Should never reach here
        return wrapper
    return decorator


def validate_file(
    file_path: str,
    must_exist: bool = True,
    allowed_extensions: Optional[List[str]] = None,
    min_size_bytes: int = 0,
    max_size_bytes: Optional[int] = None
) -> bool:
    """
    Validate a file path and its properties.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        allowed_extensions: List of allowed extensions (e.g., ['.wav', '.mp3'])
        min_size_bytes: Minimum file size
        max_size_bytes: Maximum file size
        
    Returns:
        True if valid
        
    Raises:
        FileValidationError: If validation fails
    """
    path = Path(file_path)
    
    if must_exist:
        if not path.exists():
            raise FileValidationError(f"File does not exist: {file_path}")
        if not path.is_file():
            raise FileValidationError(f"Path is not a file: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise FileValidationError(f"File is not readable: {file_path}")
    
    if allowed_extensions:
        ext = path.suffix.lower()
        if ext not in [e.lower() for e in allowed_extensions]:
            raise FileValidationError(
                f"Invalid file extension: {ext}",
                details=f"Allowed extensions: {allowed_extensions}"
            )
    
    if must_exist and path.exists():
        size = path.stat().st_size
        
        if size < min_size_bytes:
            raise FileValidationError(
                f"File too small: {size} bytes",
                details=f"Minimum size: {min_size_bytes} bytes"
            )
        
        if max_size_bytes and size > max_size_bytes:
            raise FileValidationError(
                f"File too large: {size} bytes",
                details=f"Maximum size: {max_size_bytes} bytes"
            )
    
    return True


def get_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """Calculate hash of a file for caching/checkpointing."""
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


@dataclass
class Checkpoint:
    """Represents a processing checkpoint for resume capability."""
    step_name: str
    input_file: str
    output_file: str
    input_hash: str
    timestamp: float
    metadata: Dict[str, Any]


class CheckpointManager:
    """Manages checkpoints for resumable processing."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoints.json"
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._load()
    
    def _load(self) -> None:
        """Load checkpoints from disk."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                    for key, val in data.items():
                        self._checkpoints[key] = Checkpoint(**val)
            except Exception as e:
                logger.warning(f"Failed to load checkpoints: {e}")
                self._checkpoints = {}
    
    def _save(self) -> None:
        """Save checkpoints to disk."""
        data = {k: asdict(v) for k, v in self._checkpoints.items()}
        with open(self.checkpoint_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_checkpoint_key(self, step_name: str, input_file: str) -> str:
        """Generate a unique key for a checkpoint."""
        input_hash = get_file_hash(input_file)
        return f"{step_name}:{input_hash}"
    
    def has_valid_checkpoint(self, step_name: str, input_file: str) -> bool:
        """Check if a valid checkpoint exists for this step."""
        key = self.get_checkpoint_key(step_name, input_file)
        
        if key not in self._checkpoints:
            return False
        
        checkpoint = self._checkpoints[key]
        
        # Check if output file still exists
        if not Path(checkpoint.output_file).exists():
            return False
        
        # Verify input hasn't changed
        current_hash = get_file_hash(input_file)
        if current_hash != checkpoint.input_hash:
            return False
        
        return True
    
    def get_checkpoint(self, step_name: str, input_file: str) -> Optional[Checkpoint]:
        """Get checkpoint if valid."""
        if self.has_valid_checkpoint(step_name, input_file):
            key = self.get_checkpoint_key(step_name, input_file)
            return self._checkpoints[key]
        return None
    
    def save_checkpoint(
        self,
        step_name: str,
        input_file: str,
        output_file: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save a checkpoint."""
        key = self.get_checkpoint_key(step_name, input_file)
        
        self._checkpoints[key] = Checkpoint(
            step_name=step_name,
            input_file=input_file,
            output_file=output_file,
            input_hash=get_file_hash(input_file),
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self._save()
        logger.debug(f"Saved checkpoint for {step_name}")
    
    def clear(self) -> None:
        """Clear all checkpoints."""
        self._checkpoints = {}
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


def ensure_directory(path: str) -> str:
    """Ensure directory exists and return absolute path."""
    abs_path = str(Path(path).resolve())
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds using wave module."""
    import wave
    import contextlib
    
    with contextlib.closing(wave.open(file_path, 'rb')) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def parse_timestamp(timestamp: str) -> float:
    """Parse HH:MM:SS.mmm to seconds."""
    parts = timestamp.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    else:
        return float(timestamp)

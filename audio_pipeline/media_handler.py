"""
Media Handler for the Audio Pipeline.

Handles media file discovery, validation, and conversion to WAV format.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Set
import logging

from .protocols import MediaHandlerProtocol
from .exceptions import MediaNotFoundError, MediaConversionError, FileValidationError
from .utils import validate_file, retry_with_backoff
from .config import PipelineConfig, RetryConfig

logger = logging.getLogger(__name__)


class MediaHandler(MediaHandlerProtocol):
    """
    Discovers media files in a directory and converts them to mono WAV.
    
    Implements MediaHandlerProtocol for dependency injection.
    """
    
    AUDIO_EXTENSIONS: Set[str] = {'.mp3', '.m4a', '.wav', '.ogg', '.flac', '.aac', '.wma', '.opus'}
    VIDEO_EXTENSIONS: Set[str] = {'.mp4', '.avi', '.mov', '.wmv', '.mkv', '.webm', '.m4v'}
    
    def __init__(
        self,
        media_dir: str,
        temp_dir: str,
        sample_rate: int = 16000,
        timeout_s: int = 600
    ):
        """
        Initialize MediaHandler.
        
        Args:
            media_dir: Directory to search for media files
            temp_dir: Directory for temporary files
            sample_rate: Target sample rate for conversion
            timeout_s: Timeout for FFmpeg operations
        """
        self.media_dir = str(Path(media_dir).resolve())
        self.temp_dir = str(Path(temp_dir).resolve())
        self.sample_rate = sample_rate
        self.timeout_s = timeout_s
        
        # Validate media directory exists
        if not os.path.isdir(self.media_dir):
            raise FileValidationError(f"Media directory does not exist: {self.media_dir}")
    
    @classmethod
    def from_config(cls, config: PipelineConfig) -> "MediaHandler":
        """Create MediaHandler from pipeline configuration."""
        return cls(
            media_dir=config.media_dir,
            temp_dir=config.temp_dir,
            sample_rate=config.audio.sample_rate,
            timeout_s=config.subprocess_timeout_s
        )
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate that a file exists, is readable, and has valid extension.
        
        Args:
            file_path: Path to validate
            
        Returns:
            True if valid
            
        Raises:
            FileValidationError: If validation fails
        """
        all_extensions = self.AUDIO_EXTENSIONS | self.VIDEO_EXTENSIONS
        return validate_file(
            file_path,
            must_exist=True,
            allowed_extensions=list(all_extensions),
            min_size_bytes=100  # Minimum valid audio file size
        )
    
    def _prepare_temp_dir(self) -> None:
        """Prepare temporary directory, cleaning if exists."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def find_media_file(self) -> Tuple[str, bool]:
        """
        Find the first valid audio/video file in media_dir.
        
        Returns:
            Tuple of (file_path, is_video)
            
        Raises:
            MediaNotFoundError: If no valid media file found
        """
        self._prepare_temp_dir()
        
        # First, try to find audio files (preferred)
        for fname in sorted(os.listdir(self.media_dir)):
            full_path = os.path.join(self.media_dir, fname)
            
            if not os.path.isfile(full_path):
                continue
            
            ext = Path(fname).suffix.lower()
            
            if ext in self.AUDIO_EXTENSIONS:
                logger.info(f"Found audio file: {fname}")
                return full_path, False
        
        # Then try video files
        for fname in sorted(os.listdir(self.media_dir)):
            full_path = os.path.join(self.media_dir, fname)
            
            if not os.path.isfile(full_path):
                continue
            
            ext = Path(fname).suffix.lower()
            
            if ext in self.VIDEO_EXTENSIONS:
                logger.info(f"Found video file: {fname}")
                return full_path, True
        
        raise MediaNotFoundError(
            f"No valid media file found in {self.media_dir}",
            details=f"Supported audio: {self.AUDIO_EXTENSIONS}\nSupported video: {self.VIDEO_EXTENSIONS}"
        )
    
    def find_specific_file(self, filename: str) -> Tuple[str, bool]:
        """
        Find a specific file by name.
        
        Args:
            filename: Name of the file to find
            
        Returns:
            Tuple of (file_path, is_video)
            
        Raises:
            MediaNotFoundError: If file not found
        """
        full_path = os.path.join(self.media_dir, filename)
        
        if not os.path.isfile(full_path):
            raise MediaNotFoundError(f"File not found: {filename}")
        
        ext = Path(filename).suffix.lower()
        
        if ext in self.AUDIO_EXTENSIONS:
            return full_path, False
        elif ext in self.VIDEO_EXTENSIONS:
            return full_path, True
        else:
            raise MediaNotFoundError(
                f"Unsupported file format: {ext}",
                details=f"File: {filename}"
            )
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    @retry_with_backoff(
        config=RetryConfig(max_attempts=2, initial_delay_s=1.0),
        exceptions=(subprocess.SubprocessError,)
    )
    def convert_to_wav(self, input_path: str) -> str:
        """
        Convert input audio/video to mono WAV at target sample rate.
        
        Args:
            input_path: Path to input media file
            
        Returns:
            Path to converted WAV file
            
        Raises:
            MediaConversionError: If conversion fails
        """
        # Validate input file
        self.validate_file(input_path)
        
        # Check FFmpeg availability
        if not self._check_ffmpeg():
            raise MediaConversionError(
                "FFmpeg not found",
                details="Please install FFmpeg and ensure it's in your PATH"
            )
        
        base = Path(input_path).stem
        out_path = os.path.join(self.temp_dir, f"{base}_{self.sample_rate}Hz.wav")

        cmd = [
            'ffmpeg', '-y', '-i', input_path, '-vn',
            '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(self.sample_rate),
            out_path
        ]

        logger.info(f"Converting {Path(input_path).name} to WAV...")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.timeout_s
            )
            
            if result.returncode != 0:
                stderr = result.stderr.decode(errors='replace')
                raise MediaConversionError(
                    f"FFmpeg conversion failed",
                    details=stderr[-1000:] if len(stderr) > 1000 else stderr
                )
            
            # Validate output
            if not os.path.exists(out_path):
                raise MediaConversionError("Output file not created")
            
            if os.path.getsize(out_path) < 100:
                raise MediaConversionError("Output file is too small, conversion may have failed")
            
            logger.info(f"Converted to: {out_path}")
            return out_path
            
        except subprocess.TimeoutExpired:
            raise MediaConversionError(
                f"FFmpeg timed out after {self.timeout_s}s",
                details="Consider increasing timeout or checking the input file"
            )
    
    def get_media_info(self, input_path: str) -> dict:
        """
        Get media file information using FFprobe.
        
        Args:
            input_path: Path to media file
            
        Returns:
            Dictionary with duration, sample_rate, channels, codec info
        """
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            input_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout.decode())
                
                # Extract relevant info
                audio_stream = next(
                    (s for s in info.get('streams', []) if s.get('codec_type') == 'audio'),
                    {}
                )
                
                return {
                    'duration': float(info.get('format', {}).get('duration', 0)),
                    'sample_rate': int(audio_stream.get('sample_rate', 0)),
                    'channels': int(audio_stream.get('channels', 0)),
                    'codec': audio_stream.get('codec_name', 'unknown'),
                    'bit_rate': int(info.get('format', {}).get('bit_rate', 0)),
                }
        except Exception as e:
            logger.warning(f"Failed to get media info: {e}")
        
        return {}

"""
audio_pipeline.separator

Vocal separation utilities for the audio pipeline.

Provides VocalSeparator (Demucs) and NoOpVocalSeparator implementations with
chunked processing and optional checkpoint support. Docstrings follow pydoc
conventions for Sphinx/pydoc extraction.
"""

import os
import subprocess
import shutil
import glob
from pathlib import Path
from typing import Optional
import logging

import numpy as np
from pydub import AudioSegment

from .protocols import VocalSeparatorProtocol
from .exceptions import VocalSeparationError
from .config import PipelineConfig
from .utils import CheckpointManager

logger = logging.getLogger(__name__)

__all__ = ["VocalSeparator", "NoOpVocalSeparator"]


class VocalSeparator(VocalSeparatorProtocol):
    """
    Uses Demucs to isolate vocals from audio.
    
    Features:
    - Auto-detection of whether separation is needed
    - Chunk-based processing for memory efficiency
    - Checkpoint support for resuming interrupted processing
    """
    
    def __init__(
        self,
        sample_rate: int,
        temp_dir: str,
        model: str = "htdemucs",
        chunk_minutes: float = 5.0,
        timeout_s: int = 600,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """
        Initialize VocalSeparator.
        
        Args:
            sample_rate: Target sample rate
            temp_dir: Directory for temporary files
            model: Demucs model name
            chunk_minutes: Chunk size for processing
            timeout_s: Timeout for subprocess calls
            checkpoint_manager: Optional checkpoint manager for resume support
        """
        self.sample_rate = sample_rate
        self.temp_dir = temp_dir
        self.model = model
        self.chunk_minutes = chunk_minutes
        self.timeout_s = timeout_s
        self.checkpoint_manager = checkpoint_manager
        
        os.makedirs(temp_dir, exist_ok=True)
    
    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
        checkpoint_manager: Optional[CheckpointManager] = None
    ) -> "VocalSeparator":
        """Create separator from pipeline configuration."""
        return cls(
            sample_rate=config.audio.sample_rate,
            temp_dir=config.temp_dir,
            model=config.vocal_separation.model,
            chunk_minutes=config.vocal_separation.chunk_minutes,
            timeout_s=config.subprocess_timeout_s,
            checkpoint_manager=checkpoint_manager
        )
    
    def _analyze_audio_content(self, input_wav: str) -> dict:
        """
        Analyze audio to determine if it has significant non-vocal content.
        
        Uses spectral analysis to detect music characteristics.
        
        Returns:
            Dict with analysis results including 'has_music' boolean
        """
        try:
            seg = AudioSegment.from_wav(input_wav)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            
            # Simple heuristics for music detection
            # Music typically has:
            # 1. More consistent energy (less dynamic range in short windows)
            # 2. Periodic patterns (rhythm)
            # 3. More low-frequency content
            
            # Calculate short-term energy variance
            window_size = int(self.sample_rate * 0.05)  # 50ms windows
            num_windows = len(samples) // window_size
            
            if num_windows < 10:
                return {'has_music': False, 'confidence': 0.0, 'reason': 'Audio too short'}
            
            energies = []
            for i in range(num_windows):
                window = samples[i * window_size:(i + 1) * window_size]
                energies.append(np.sqrt(np.mean(window ** 2)))
            
            energies = np.array(energies)
            energy_std = np.std(energies)
            energy_mean = np.mean(energies)
            
            # Coefficient of variation - lower = more consistent (music-like)
            cv = energy_std / (energy_mean + 1e-10)
            
            # Music typically has CV < 0.5, speech has CV > 0.7
            has_music = cv < 0.6
            confidence = max(0, min(1, (0.8 - cv) / 0.4))
            
            result = {
                'has_music': has_music,
                'confidence': confidence,
                'energy_cv': cv,
                'reason': 'Low energy variance suggests background music' if has_music else 'High energy variance suggests speech only'
            }
            
            logger.info(f"Audio analysis: {result}")
            return result
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}, assuming no music")
            return {'has_music': False, 'confidence': 0.0, 'reason': f'Analysis failed: {e}'}
    
    def is_separation_needed(self, input_wav: str) -> bool:
        """
        Determine if vocal separation is needed for this audio.
        
        Args:
            input_wav: Path to input WAV file
            
        Returns:
            True if separation is recommended
        """
        analysis = self._analyze_audio_content(input_wav)
        return analysis.get('has_music', False) and analysis.get('confidence', 0) > 0.5
    
    def _check_demucs(self) -> bool:
        """Check if Demucs is available."""
        try:
            result = subprocess.run(
                ['python', '-m', 'demucs', '--help'],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _process_chunk(
        self,
        chunk_path: str,
        out_dir: str,
        chunk_index: int
    ) -> str:
        """
        Process a single audio chunk through Demucs.
        
        Args:
            chunk_path: Path to chunk WAV file
            out_dir: Output directory for stems
            chunk_index: Index of this chunk
            
        Returns:
            Path to extracted vocals file
        """
        cmd = [
            'python', '-m', 'demucs',
            '--two-stems', 'vocals',
            '-n', self.model,
            '-o', out_dir,
            chunk_path
        ]
        
        logger.info(f"Processing chunk {chunk_index}: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.timeout_s
            )
            
            if result.returncode != 0:
                stderr = result.stderr.decode('utf-8', errors='replace')
                raise VocalSeparationError(
                    f"Demucs failed on chunk {chunk_index}",
                    details=stderr[-1000:]
                )
            
            # Find output vocals file
            stem_dir = os.path.join(out_dir, self.model, Path(chunk_path).stem)
            vocals_path = os.path.join(stem_dir, 'vocals.wav')
            
            if not os.path.exists(vocals_path):
                # Try alternative path patterns
                pattern = os.path.join(out_dir, '**', 'vocals.wav')
                matches = glob.glob(pattern, recursive=True)
                
                if matches:
                    vocals_path = matches[-1]  # Take most recent
                else:
                    raise VocalSeparationError(
                        f"Vocals file not found for chunk {chunk_index}",
                        details=f"Expected at: {vocals_path}"
                    )
            
            return vocals_path
            
        except subprocess.TimeoutExpired:
            raise VocalSeparationError(
                f"Demucs timed out on chunk {chunk_index}",
                details=f"Timeout: {self.timeout_s}s"
            )
    
    def extract_vocals(self, input_wav: str, force: bool = False) -> str:
        """
        Extract vocals from audio using Demucs.
        
        Processes audio in chunks to manage memory usage.
        Supports checkpoint/resume for long files.
        
        Args:
            input_wav: Path to input WAV file
            force: Force separation even if not detected as needed
            
        Returns:
            Path to vocals-only WAV file
        """
        # Check if separation is needed
        if not force and not self.is_separation_needed(input_wav):
            logger.info("Vocal separation not needed, skipping")
            return input_wav
        
        # Check Demucs availability
        if not self._check_demucs():
            raise VocalSeparationError(
                "Demucs not found",
                details="Install with: pip install demucs"
            )
        
        # Check for existing checkpoint
        if self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.get_checkpoint("vocal_separation", input_wav)
            if checkpoint and os.path.exists(checkpoint.output_file):
                logger.info(f"Using cached vocals from checkpoint: {checkpoint.output_file}")
                return checkpoint.output_file
        
        # Load audio and prepare for chunked processing
        seg = AudioSegment.from_wav(input_wav)
        total_ms = len(seg)
        chunk_ms = int(self.chunk_minutes * 60_000)
        
        vocals_full = AudioSegment.silent(duration=0, frame_rate=self.sample_rate)
        out_dir = os.path.join(self.temp_dir, 'stems')
        os.makedirs(out_dir, exist_ok=True)
        
        # Process chunks
        chunk_index = 0
        for start in range(0, total_ms, chunk_ms):
            end = min(start + chunk_ms, total_ms)
            chunk = seg[start:end]
            
            chunk_path = os.path.join(self.temp_dir, f"chunk_{chunk_index}.wav")
            chunk.export(chunk_path, format='wav')
            
            try:
                vocals_path = self._process_chunk(chunk_path, out_dir, chunk_index)
                vocals_full += AudioSegment.from_wav(vocals_path)
                
                # Save intermediate checkpoint
                if self.checkpoint_manager and chunk_index > 0:
                    intermediate_path = os.path.join(
                        self.temp_dir,
                        f"{Path(input_wav).stem}_vocals_partial.wav"
                    )
                    vocals_full.export(intermediate_path, format='wav')
                    
            finally:
                # Cleanup chunk files
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            
            chunk_index += 1
            logger.info(f"Processed chunk {chunk_index}/{(total_ms + chunk_ms - 1) // chunk_ms}")
        
        # Cleanup stems directory
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        
        # Export final result
        out_path = os.path.join(self.temp_dir, f"{Path(input_wav).stem}_vocals.wav")
        vocals_full.export(out_path, format='wav')
        
        # Save checkpoint
        if self.checkpoint_manager:
            self.checkpoint_manager.save_checkpoint(
                step_name="vocal_separation",
                input_file=input_wav,
                output_file=out_path,
                metadata={"model": self.model, "chunks": chunk_index}
            )
        
        logger.info(f"Vocals extracted: {out_path}")
        return out_path


class NoOpVocalSeparator(VocalSeparatorProtocol):
    """
    No-operation separator that passes through audio unchanged.
    
    Used when vocal separation is disabled.
    """
    
    def extract_vocals(self, input_wav: str) -> str:
        """Return input unchanged."""
        logger.debug("NoOp vocal separator: passing through unchanged")
        return input_wav
    
    def is_separation_needed(self, input_wav: str) -> bool:
        """Always returns False."""
        return False

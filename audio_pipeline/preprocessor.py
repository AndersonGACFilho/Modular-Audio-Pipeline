"""
Audio Preprocessor for the Audio Pipeline.

Handles noise reduction, normalization, and silence removal with timestamp preservation.
"""

import os
import wave
import contextlib
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging

from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_nonsilent, detect_silence
import noisereduce as nr
import pyloudnorm as pyln

from .protocols import PreprocessorProtocol, TimestampMapping
from .exceptions import AudioProcessingError
from .config import PipelineConfig, NoiseReductionConfig

logger = logging.getLogger(__name__)


class AudioPreprocessor(PreprocessorProtocol):
    """
    Performs noise reduction, silence removal, and loudness normalization.
    
    Implements timestamp preservation to map processed audio back to original.
    """
    
    def __init__(
        self,
        sample_rate: int,
        temp_dir: str,
        noise_config: Optional[NoiseReductionConfig] = None
    ):
        """
        Initialize AudioPreprocessor.
        
        Args:
            sample_rate: Target sample rate
            temp_dir: Directory for temporary files
            noise_config: Noise reduction configuration
        """
        self.sample_rate = sample_rate
        self.temp_dir = temp_dir
        self.noise_config = noise_config or NoiseReductionConfig()
        
        os.makedirs(temp_dir, exist_ok=True)
    
    @classmethod
    def from_config(cls, config: PipelineConfig) -> "AudioPreprocessor":
        """Create preprocessor from pipeline configuration."""
        return cls(
            sample_rate=config.audio.sample_rate,
            temp_dir=config.temp_dir,
            noise_config=config.noise_reduction
        )
    
    def read_wave(self, path: str) -> Tuple[bytes, int]:
        """Read WAV file and return PCM data and sample rate."""
        try:
            with contextlib.closing(wave.open(path, 'rb')) as wf:
                sr = wf.getframerate()
                pcm = wf.readframes(wf.getnframes())
            return pcm, sr
        except Exception as e:
            raise AudioProcessingError(f"Failed to read WAV file: {path}", details=str(e))
    
    def write_wave(self, path: str, audio: bytes, sample_rate: int) -> None:
        """Write PCM data to WAV file."""
        try:
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
        except Exception as e:
            raise AudioProcessingError(f"Failed to write WAV file: {path}", details=str(e))
    
    def _detect_noise_segments(self, audio: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        """
        Detect segments that are likely noise (low energy, no speech).
        
        Uses energy-based detection combined with zero-crossing rate.
        
        Args:
            audio: Audio samples as float32 array
            sr: Sample rate
            
        Returns:
            List of (start_sample, end_sample) tuples for noise segments
        """
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(sr * 0.010)    # 10ms hop
        
        # Calculate frame-wise energy
        num_frames = (len(audio) - frame_length) // hop_length + 1
        energies = np.zeros(num_frames)
        zcrs = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio[start:end]
            
            # RMS energy
            energies[i] = np.sqrt(np.mean(frame ** 2))
            
            # Zero crossing rate
            zcrs[i] = np.sum(np.abs(np.diff(np.signbit(frame)))) / frame_length
        
        # Noise segments have low energy AND high zero-crossing rate (relative)
        energy_threshold = np.percentile(energies, 20)  # Bottom 20% energy
        zcr_threshold = np.percentile(zcrs, 50)  # Above median ZCR
        
        noise_frames = (energies < energy_threshold) & (zcrs > zcr_threshold * 0.5)
        
        # Find contiguous noise regions
        noise_segments = []
        in_noise = False
        start_frame = 0
        
        for i, is_noise in enumerate(noise_frames):
            if is_noise and not in_noise:
                start_frame = i
                in_noise = True
            elif not is_noise and in_noise:
                # Require minimum length (100ms)
                if (i - start_frame) * hop_length / sr >= 0.1:
                    start_sample = start_frame * hop_length
                    end_sample = i * hop_length
                    noise_segments.append((start_sample, end_sample))
                in_noise = False
        
        return noise_segments
    
    def reduce_stationary_noise(
        self,
        input_wav: str,
        noise_sample_path: Optional[str] = None
    ) -> str:
        """
        Reduce stationary noise from audio.
        
        Uses auto-detection or provided noise sample.
        
        Args:
            input_wav: Input WAV file path
            noise_sample_path: Optional path to noise sample file
            
        Returns:
            Path to denoised WAV file
        """
        if not self.noise_config.enabled:
            logger.info("Noise reduction disabled, skipping")
            return input_wav
        
        pcm, sr = self.read_wave(input_wav)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        
        # Get noise profile
        if noise_sample_path:
            # Use provided noise sample
            noise_pcm, noise_sr = self.read_wave(noise_sample_path)
            noise_clip = np.frombuffer(noise_pcm, dtype=np.int16).astype(np.float32)
            logger.info(f"Using provided noise sample: {noise_sample_path}")
        elif self.noise_config.auto_detect_noise:
            # Auto-detect noise segments
            noise_segments = self._detect_noise_segments(audio, sr)
            
            if noise_segments:
                # Use the longest noise segment
                longest = max(noise_segments, key=lambda x: x[1] - x[0])
                noise_clip = audio[longest[0]:longest[1]]
                logger.info(f"Auto-detected noise segment: {longest[0]/sr:.2f}s - {longest[1]/sr:.2f}s")
            else:
                # Fall back to first N seconds
                duration_samples = int(sr * self.noise_config.noise_sample_duration_s)
                noise_clip = audio[:duration_samples]
                logger.warning("No noise segments detected, using first 0.5s as noise profile")
        else:
            # Use first N seconds as noise profile
            duration_samples = int(sr * self.noise_config.noise_sample_duration_s)
            noise_clip = audio[:duration_samples]
            logger.info(f"Using first {self.noise_config.noise_sample_duration_s}s as noise profile")
        
        try:
            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                y_noise=noise_clip,
                prop_decrease=0.8,  # How much to reduce noise
                stationary=True
            )
        except Exception as e:
            raise AudioProcessingError("Noise reduction failed", details=str(e))
        
        out_path = os.path.join(self.temp_dir, f"{Path(input_wav).stem}_denoised.wav")
        self.write_wave(out_path, reduced.astype(np.int16).tobytes(), sr)
        
        logger.info(f"Noise reduced: {out_path}")
        return out_path
    
    def normalize_audio(self, input_wav: str) -> str:
        """
        Apply peak normalization and ensure mono at target sample rate.
        
        Args:
            input_wav: Input WAV file path
            
        Returns:
            Path to normalized WAV file
        """
        try:
            seg = AudioSegment.from_wav(input_wav)
            norm_seg = (
                normalize(seg)
                .set_frame_rate(self.sample_rate)
                .set_channels(1)
                .set_sample_width(2)  # 16-bit
            )
            
            out_path = os.path.join(self.temp_dir, f"{Path(input_wav).stem}_norm.wav")
            norm_seg.export(out_path, format='wav')
            
            logger.info(f"Audio normalized: {out_path}")
            return out_path
            
        except Exception as e:
            raise AudioProcessingError("Audio normalization failed", details=str(e))
    
    def normalize_loudness(self, input_wav: str, target_lufs: float = -16.0) -> str:
        """
        Apply LUFS loudness normalization.
        
        Args:
            input_wav: Input WAV file path
            target_lufs: Target loudness in LUFS (default -16)
            
        Returns:
            Path to loudness-normalized WAV file
        """
        try:
            seg = AudioSegment.from_wav(input_wav)
            samples = np.array(seg.get_array_of_samples(), dtype=np.int16).astype(np.float32) / 32768.0
            
            # Ensure mono for pyloudnorm
            if seg.channels > 1:
                samples = samples.reshape(-1, seg.channels).mean(axis=1)
            
            meter = pyln.Meter(self.sample_rate)
            loudness = meter.integrated_loudness(samples)
            
            # Handle silent or near-silent audio
            if not np.isfinite(loudness) or loudness < -70:
                logger.warning("Audio is too quiet for LUFS normalization, skipping")
                return input_wav
            
            normalized = pyln.normalize.loudness(samples, loudness, target_lufs)
            
            # Prevent clipping
            peak = np.abs(normalized).max()
            if peak > 1.0:
                normalized /= peak
                logger.debug(f"Applied limiter to prevent clipping (peak was {peak:.2f})")
            
            out_int16 = np.clip(normalized * 32768, -32768, 32767).astype(np.int16).tobytes()
            
            out_path = os.path.join(self.temp_dir, f"{Path(input_wav).stem}_loudnorm.wav")
            self.write_wave(out_path, out_int16, self.sample_rate)
            
            logger.info(f"Loudness normalized to {target_lufs} LUFS: {out_path}")
            return out_path
            
        except Exception as e:
            raise AudioProcessingError("Loudness normalization failed", details=str(e))
    
    def remove_silence(
        self,
        input_wav: str,
        min_silence_len: int = 250,
        silence_offset_db: float = 40.0,
        silence_margin: int = 100,
        preserve_timestamps: bool = True
    ) -> Tuple[str, List[TimestampMapping]]:
        """
        Remove silent segments from audio while preserving timestamp mappings.
        
        Args:
            input_wav: Input WAV file path
            min_silence_len: Minimum silence length in ms to remove
            silence_offset_db: dB below average to consider silence
            silence_margin: Margin in ms to keep around speech
            preserve_timestamps: Whether to return timestamp mappings
            
        Returns:
            Tuple of (output_path, list_of_timestamp_mappings)
        """
        try:
            seg = AudioSegment.from_wav(input_wav)
            silence_thresh = seg.dBFS - silence_offset_db
            
            nonsilent_ranges = detect_nonsilent(
                seg,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh
            )
            
            if not nonsilent_ranges:
                logger.warning("No non-silent segments found, returning original")
                return input_wav, []
            
            # Build output audio and timestamp mappings
            cleaned = AudioSegment.empty()
            timestamp_mappings: List[TimestampMapping] = []
            processed_position_ms = 0
            
            for start_ms, end_ms in nonsilent_ranges:
                # Apply margin
                s = max(0, start_ms - silence_margin)
                e = min(len(seg), end_ms + silence_margin)
                
                chunk = seg[s:e]
                chunk_duration_ms = len(chunk)
                
                # Record timestamp mapping
                if preserve_timestamps:
                    mapping = TimestampMapping(
                        processed_start=processed_position_ms / 1000.0,
                        processed_end=(processed_position_ms + chunk_duration_ms) / 1000.0,
                        original_start=s / 1000.0,
                        original_end=e / 1000.0
                    )
                    timestamp_mappings.append(mapping)
                
                # Add chunk (with crossfade for smooth transitions)
                if len(cleaned) == 0:
                    cleaned = chunk
                else:
                    # Small crossfade to avoid clicks
                    crossfade_ms = min(20, chunk_duration_ms // 4)
                    cleaned = cleaned.append(chunk, crossfade=crossfade_ms)
                    # Adjust for crossfade in position tracking
                    processed_position_ms -= crossfade_ms
                
                processed_position_ms += chunk_duration_ms
            
            out_path = os.path.join(self.temp_dir, f"{Path(input_wav).stem}_nosilence.wav")
            cleaned.export(out_path, format='wav')
            
            original_duration = len(seg) / 1000.0
            processed_duration = len(cleaned) / 1000.0
            removed_duration = original_duration - processed_duration
            
            logger.info(
                f"Silence removed: {out_path} "
                f"(removed {removed_duration:.1f}s, {removed_duration/original_duration*100:.1f}%)"
            )
            
            return out_path, timestamp_mappings
            
        except Exception as e:
            raise AudioProcessingError("Silence removal failed", details=str(e))
    
    def detect_silence_segments(
        self,
        input_wav: str,
        min_silence_len: int = 500,
        silence_offset_db: float = 40.0
    ) -> List[Tuple[float, float]]:
        """
        Detect silence segments without modifying audio.
        
        Useful for analysis or choosing noise profile.
        
        Args:
            input_wav: Input WAV file path
            min_silence_len: Minimum silence length in ms
            silence_offset_db: dB below average to consider silence
            
        Returns:
            List of (start_seconds, end_seconds) tuples
        """
        seg = AudioSegment.from_wav(input_wav)
        silence_thresh = seg.dBFS - silence_offset_db
        
        silent_ranges = detect_silence(
            seg,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        return [(s / 1000.0, e / 1000.0) for s, e in silent_ranges]

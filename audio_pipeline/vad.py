"""
Voice Activity Detection (VAD) for the Audio Pipeline.

Filters audio to keep only voiced segments using WebRTC VAD.
Preserves timestamp mappings for alignment with original audio.
"""

import os
import wave
import contextlib
import collections
from pathlib import Path
from typing import Tuple, List, Optional
import logging

import webrtcvad

from .protocols import VADProtocol, TimestampMapping
from .exceptions import VADError
from .config import PipelineConfig, VADConfig

logger = logging.getLogger(__name__)


class VADFilter(VADProtocol):
    """
    Filters out non-speech using WebRTC VAD.
    
    Implements timestamp preservation for mapping back to original audio.
    """
    
    # WebRTC VAD only supports these sample rates
    SUPPORTED_SAMPLE_RATES = [8000, 16000, 32000, 48000]
    
    # WebRTC VAD only supports these frame durations
    SUPPORTED_FRAME_DURATIONS = [10, 20, 30]
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        padding_duration_ms: int = 500,
        start_threshold: float = 0.5,
        stop_threshold: float = 0.9,
        vad_mode: int = 1
    ):
        """
        Initialize VAD filter.
        
        Args:
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame duration in ms (must be 10, 20, or 30)
            padding_duration_ms: Padding duration for smoothing
            start_threshold: Ratio of voiced frames to trigger speech start
            stop_threshold: Ratio of unvoiced frames to trigger speech end
            vad_mode: Aggressiveness mode (0-3, higher = more aggressive)
        """
        # Validate parameters
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise VADError(
                f"Unsupported sample rate: {sample_rate}",
                details=f"WebRTC VAD supports: {self.SUPPORTED_SAMPLE_RATES}"
            )
        
        if frame_duration_ms not in self.SUPPORTED_FRAME_DURATIONS:
            raise VADError(
                f"Unsupported frame duration: {frame_duration_ms}ms",
                details=f"WebRTC VAD supports: {self.SUPPORTED_FRAME_DURATIONS}ms"
            )
        
        if not 0 <= vad_mode <= 3:
            raise VADError(f"VAD mode must be 0-3, got: {vad_mode}")
        
        self.sample_rate = sample_rate
        self.frame_ms = frame_duration_ms
        self.padding_ms = padding_duration_ms
        self.start_th = start_threshold
        self.stop_th = stop_threshold
        
        try:
            self.vad = webrtcvad.Vad(vad_mode)
        except Exception as e:
            raise VADError("Failed to initialize WebRTC VAD", details=str(e))
    
    @classmethod
    def from_config(cls, config: PipelineConfig) -> "VADFilter":
        """Create VAD filter from pipeline configuration."""
        return cls(
            sample_rate=config.audio.sample_rate,
            frame_duration_ms=config.vad.frame_duration_ms,
            padding_duration_ms=config.vad.padding_duration_ms,
            start_threshold=config.vad.start_threshold,
            stop_threshold=config.vad.stop_threshold,
            vad_mode=config.vad.mode
        )
    
    def read_wave(self, path: str) -> Tuple[bytes, int]:
        """Read WAV file and return PCM data and sample rate."""
        try:
            with contextlib.closing(wave.open(path, 'rb')) as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                
                if channels != 1:
                    raise VADError(f"VAD requires mono audio, got {channels} channels")
                if sample_width != 2:
                    raise VADError(f"VAD requires 16-bit audio, got {sample_width * 8}-bit")
                
                pcm = wf.readframes(wf.getnframes())
            return pcm, sr
        except wave.Error as e:
            raise VADError(f"Failed to read WAV file: {path}", details=str(e))
    
    def write_wave(self, path: str, audio: bytes, sample_rate: int) -> None:
        """Write PCM data to WAV file."""
        try:
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
        except Exception as e:
            raise VADError(f"Failed to write WAV file: {path}", details=str(e))
    
    def _frame_generator(self, pcm: bytes, sample_rate: int) -> List[Tuple[bytes, float, float]]:
        """
        Generate frames from PCM data with timestamps.
        
        Yields:
            Tuples of (frame_bytes, start_time, end_time)
        """
        # Frame size in bytes (16-bit = 2 bytes per sample)
        frame_len = int(sample_rate * (self.frame_ms / 1000) * 2)
        
        frames = []
        offset = 0
        frame_index = 0
        
        while offset + frame_len <= len(pcm):
            frame = pcm[offset:offset + frame_len]
            start_time = frame_index * self.frame_ms / 1000.0
            end_time = (frame_index + 1) * self.frame_ms / 1000.0
            frames.append((frame, start_time, end_time))
            offset += frame_len
            frame_index += 1
        
        return frames
    
    def detect_speech_segments(self, input_wav: str) -> List[Tuple[float, float]]:
        """
        Detect speech segments without modifying audio.
        
        Args:
            input_wav: Path to input WAV file
            
        Returns:
            List of (start_seconds, end_seconds) tuples for speech segments
        """
        pcm, sr = self.read_wave(input_wav)
        frames = self._frame_generator(pcm, sr)
        
        ring_size = int(self.padding_ms / self.frame_ms)
        ring = collections.deque(maxlen=ring_size)
        
        speech_segments = []
        triggered = False
        segment_start = 0.0
        
        for frame, start_time, end_time in frames:
            try:
                is_speech = self.vad.is_speech(frame, sr)
            except Exception:
                is_speech = False
            
            if not triggered:
                ring.append((frame, is_speech, start_time, end_time))
                voiced_count = sum(1 for _, speech, _, _ in ring if speech)
                
                if voiced_count > self.start_th * ring.maxlen:
                    # Speech started
                    triggered = True
                    segment_start = ring[0][2] if ring else start_time
                    ring.clear()
            else:
                ring.append((frame, is_speech, start_time, end_time))
                unvoiced_count = sum(1 for _, speech, _, _ in ring if not speech)
                
                if unvoiced_count > self.stop_th * ring.maxlen:
                    # Speech ended
                    triggered = False
                    segment_end = ring[0][3] if ring else end_time
                    speech_segments.append((segment_start, segment_end))
                    ring.clear()
        
        # Handle case where speech continues to end
        if triggered:
            speech_segments.append((segment_start, frames[-1][3] if frames else 0))
        
        return speech_segments
    
    def filter_voice(
        self,
        input_wav: str,
        output_dir: str,
        preserve_timestamps: bool = True
    ) -> Tuple[str, List[TimestampMapping]]:
        """
        Filter audio to keep only voiced segments.
        
        Args:
            input_wav: Path to input WAV file
            output_dir: Directory for output file
            preserve_timestamps: Whether to return timestamp mappings
            
        Returns:
            Tuple of (output_path, list_of_timestamp_mappings)
        """
        pcm, sr = self.read_wave(input_wav)
        frames = self._frame_generator(pcm, sr)
        
        if not frames:
            raise VADError("No frames generated from audio")
        
        ring_size = int(self.padding_ms / self.frame_ms)
        ring = collections.deque(maxlen=ring_size)
        
        voiced_segments: List[Tuple[List[bytes], float, float]] = []
        current_segment_frames: List[bytes] = []
        current_segment_start = 0.0
        triggered = False
        
        for frame, start_time, end_time in frames:
            try:
                is_speech = self.vad.is_speech(frame, sr)
            except Exception:
                is_speech = False
            
            if not triggered:
                ring.append((frame, is_speech, start_time, end_time))
                voiced_count = sum(1 for _, speech, _, _ in ring if speech)
                
                if voiced_count > self.start_th * ring.maxlen:
                    # Speech started - include buffered frames
                    triggered = True
                    current_segment_start = ring[0][2] if ring else start_time
                    current_segment_frames = [f for f, _, _, _ in ring]
                    ring.clear()
            else:
                current_segment_frames.append(frame)
                ring.append((frame, is_speech, start_time, end_time))
                unvoiced_count = sum(1 for _, speech, _, _ in ring if not speech)
                
                if unvoiced_count > self.stop_th * ring.maxlen:
                    # Speech ended
                    triggered = False
                    segment_end = ring[0][3] if ring else end_time
                    voiced_segments.append((
                        current_segment_frames.copy(),
                        current_segment_start,
                        segment_end
                    ))
                    current_segment_frames = []
                    ring.clear()
        
        # Handle trailing speech
        if triggered and current_segment_frames:
            voiced_segments.append((
                current_segment_frames,
                current_segment_start,
                frames[-1][2]
            ))
        
        if not voiced_segments:
            logger.warning("No voiced segments detected, returning original audio")
            return input_wav, []
        
        # Build output audio and timestamp mappings
        all_voiced_frames = []
        timestamp_mappings: List[TimestampMapping] = []
        processed_position = 0.0
        
        for segment_frames, orig_start, orig_end in voiced_segments:
            segment_duration = len(segment_frames) * self.frame_ms / 1000.0
            
            if preserve_timestamps:
                mapping = TimestampMapping(
                    processed_start=processed_position,
                    processed_end=processed_position + segment_duration,
                    original_start=orig_start,
                    original_end=orig_end
                )
                timestamp_mappings.append(mapping)
            
            all_voiced_frames.extend(segment_frames)
            processed_position += segment_duration
        
        # Write output
        voiced_audio = b''.join(all_voiced_frames)
        out_path = os.path.join(output_dir, f"{Path(input_wav).stem}_voice.wav")
        self.write_wave(out_path, voiced_audio, sr)
        
        # Log statistics
        original_duration = len(frames) * self.frame_ms / 1000.0
        voiced_duration = processed_position
        removed_duration = original_duration - voiced_duration
        
        logger.info(
            f"VAD filtered: {out_path} "
            f"(kept {voiced_duration:.1f}s, removed {removed_duration:.1f}s, "
            f"{voiced_duration/original_duration*100:.1f}% voiced)"
        )
        
        return out_path, timestamp_mappings


class NoOpVADFilter(VADProtocol):
    """
    No-operation VAD that passes through audio unchanged.
    
    Used when VAD is disabled.
    """
    
    def filter_voice(
        self,
        input_wav: str,
        output_dir: str
    ) -> Tuple[str, List[TimestampMapping]]:
        """Return input unchanged with identity mapping."""
        logger.debug("NoOp VAD: passing through unchanged")
        
        # Create identity mapping
        import wave
        with wave.open(input_wav, 'rb') as wf:
            duration = wf.getnframes() / wf.getframerate()
        
        mapping = TimestampMapping(
            processed_start=0.0,
            processed_end=duration,
            original_start=0.0,
            original_end=duration
        )
        
        return input_wav, [mapping]
    
    def detect_speech_segments(self, input_wav: str) -> List[Tuple[float, float]]:
        """Return entire audio as one segment."""
        import wave
        with wave.open(input_wav, 'rb') as wf:
            duration = wf.getnframes() / wf.getframerate()
        return [(0.0, duration)]

import os
import wave
import contextlib
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_nonsilent
import noisereduce as nr
import pyloudnorm as pyln
import logging

from typing import Tuple

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """
    Performs noise reduction, silence removal, and loudness normalization.
    """

    def __init__(self, sample_rate: int, temp_dir: str):
        self.sample_rate = sample_rate
        self.temp_dir = temp_dir

    def read_wave(self, path: str) -> Tuple[bytes, int]:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            sr = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())
        return pcm, sr

    def write_wave(self, path: str, audio: bytes, sample_rate: int):
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

    def reduce_stationary_noise(self, input_wav: str) -> str:
        """
        Use the first 0.5s as noise profile to denoise.
        """
        pcm, sr = self.read_wave(input_wav)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        noise_clip = audio[: int(sr * 0.5)]
        reduced = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip)
        out_path = os.path.join(self.temp_dir, f"{Path(input_wav).stem}_denoised.wav")
        self.write_wave(out_path, reduced.astype(np.int16).tobytes(), sr)
        return out_path

    def normalize_audio(self, input_wav: str) -> str:
        """
        Basic peak normalization + ensure mono 16kHz.
        """
        seg = AudioSegment.from_wav(input_wav)
        norm_seg = normalize(seg).set_frame_rate(self.sample_rate).set_channels(1)
        out_path = os.path.join(self.temp_dir, f"{Path(input_wav).stem}_norm.wav")
        norm_seg.export(out_path, format='wav')
        return out_path

    def normalize_loudness(self, input_wav: str, target_lufs: float = -16.0) -> str:
        """
        Apply LUFS normalization via pyloudnorm.
        """
        seg = AudioSegment.from_wav(input_wav)
        samples = np.array(seg.get_array_of_samples(), dtype=np.int16).astype(np.float32) / 32768.0
        meter = pyln.Meter(self.sample_rate)
        loudness = meter.integrated_loudness(samples)
        normalized = pyln.normalize.loudness(samples, loudness, target_lufs)
        peak = np.abs(normalized).max()
        if peak > 1.0:
            normalized /= peak
        out_int16 = np.clip(normalized * 32768, -32768, 32767).astype(np.int16).tobytes()
        out_path = os.path.join(self.temp_dir, f"{Path(input_wav).stem}_loudnorm.wav")
        self.write_wave(out_path, out_int16, self.sample_rate)
        return out_path

    def remove_silence(
        self,
        input_wav: str,
        min_silence_len: int = 250,
        silence_offset_db: float = 40.0,
        silence_margin: int = 100
    ) -> str:
        """
        Remove silent segments from the WAV, keeping a small margin around speech.
        """
        seg = AudioSegment.from_wav(input_wav)
        silence_thresh = seg.dBFS - silence_offset_db
        nonsilent_ranges = detect_nonsilent(
            seg,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )

        cleaned = AudioSegment.empty()
        for start_ms, end_ms in nonsilent_ranges:
            s = max(0, start_ms - silence_margin)
            e = min(len(seg), end_ms + silence_margin)
            chunk = seg[s:e]
            if len(cleaned) == 0:
                cleaned = chunk
            else:
                cleaned = cleaned.append(chunk, crossfade=20)

        out_path = os.path.join(self.temp_dir, f"{Path(input_wav).stem}_nosilence.wav")
        cleaned.export(out_path, format='wav')
        logger.info("Silence removed: %s", out_path)
        return out_path

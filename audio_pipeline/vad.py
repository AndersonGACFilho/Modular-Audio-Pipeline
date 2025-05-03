import os
import wave
import contextlib
import collections
from pathlib import Path
import webrtcvad
import logging

from typing import Tuple

logger = logging.getLogger(__name__)

class VADFilter:
    """
    Filters out non-speech using WebRTC VAD.
    """
    def __init__(self, sample_rate: int, frame_duration_ms: int,
                 padding_duration_ms: int, start_threshold: float,
                 stop_threshold: float, vad_mode: int = 1):
        self.sample_rate = sample_rate
        self.frame_ms = frame_duration_ms
        self.padding_ms = padding_duration_ms
        self.start_th = start_threshold
        self.stop_th = stop_threshold
        self.vad = webrtcvad.Vad(vad_mode)

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

    def filter_voice(self, input_wav: str, output_dir: str) -> str:
        """
        Outputs a WAV containing only voiced frames.
        """
        pcm, sr = self.read_wave(input_wav)
        frame_len = int(sr * (self.frame_ms / 1000) * 2)
        frames, offset = [], 0
        while offset + frame_len <= len(pcm):
            frames.append(pcm[offset:offset+frame_len])
            offset += frame_len

        ring = collections.deque(maxlen=int(self.padding_ms / self.frame_ms))
        voiced_segments, triggered = [], False

        for chunk in frames:
            is_speech = self.vad.is_speech(chunk, sr)
            if not triggered:
                ring.append((chunk, is_speech))
                if sum(1 for c, speech in ring if speech) > self.start_th * ring.maxlen:
                    triggered = True
                    for c, _ in ring:
                        voiced_segments.append(c)
                    ring.clear()
            else:
                voiced_segments.append(chunk)
                ring.append((chunk, is_speech))
                if sum(1 for c, speech in ring if not speech) > self.stop_th * ring.maxlen:
                    triggered = False
                    ring.clear()

        out_path = os.path.join(output_dir, f"{Path(input_wav).stem}_voice.wav")
        self.write_wave(out_path, b''.join(voiced_segments), sr)
        return out_path

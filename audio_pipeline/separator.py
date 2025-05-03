import os
import shutil
from pathlib import Path
from pydub import AudioSegment
from spleeter.separator import Separator
import logging

logger = logging.getLogger(__name__)

class VocalSeparator:
    """
    Uses Spleeter 2-stem to isolate vocals in chunks.
    """
    def __init__(self, sample_rate: int, temp_dir: str):
        self.sample_rate = sample_rate
        self.temp_dir = temp_dir
        self.separator = Separator('spleeter:2stems', multiprocess=False)

    def extract_vocals(self, input_wav: str, chunk_minutes: float = 5.0) -> str:
        seg = AudioSegment.from_wav(input_wav)
        total_ms, chunk_ms = len(seg), int(chunk_minutes * 60_000)
        vocals_full = AudioSegment.silent(duration=0, frame_rate=self.sample_rate)

        for start in range(0, total_ms, chunk_ms):
            end = min(start + chunk_ms, total_ms)
            chunk = seg[start:end]
            chunk_path = os.path.join(self.temp_dir, f"chunk_{start//1000}.wav")
            chunk.export(chunk_path, format='wav')

            out_dir = os.path.join(self.temp_dir, 'stems')
            os.makedirs(out_dir, exist_ok=True)
            self.separator.separate_to_file(chunk_path, out_dir)

            stem_dir = os.path.join(out_dir, Path(chunk_path).stem)
            vocals_path = os.path.join(stem_dir, 'vocals.wav')
            vocals_full += AudioSegment.from_wav(vocals_path)

            os.remove(chunk_path)
            shutil.rmtree(stem_dir)

        out_path = os.path.join(self.temp_dir, f"{Path(input_wav).stem}_vocals.wav")
        vocals_full.export(out_path, format='wav')
        return out_path

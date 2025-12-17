import os
import subprocess
import shutil
import glob
from pydub import AudioSegment
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VocalSeparator:
    """
    Uses Demucs to isolate vocals from audio.
    """
    def __init__(self, sample_rate: int, temp_dir: str, model: str = "htdemucs"):
        self.sample_rate = sample_rate
        self.temp_dir = temp_dir
        self.model = model

    def extract_vocals(self, input_wav: str, chunk_minutes: float = 5.0) -> str:
        """
        Extract vocals from audio using Demucs.
        Processes audio in chunks to manage memory usage.
        """
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

            # Run Demucs via subprocess
            cmd = [
                'python', '-m', 'demucs',
                '--two-stems', 'vocals',
                '-n', self.model,
                '-o', out_dir,
                chunk_path
            ]
            logger.info("Running Demucs: %s", ' '.join(cmd))
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                # Decode with errors='replace' to handle encoding issues on Windows
                stderr_text = result.stderr.decode('utf-8', errors='replace')
                logger.error("Demucs error: %s", stderr_text)
                raise RuntimeError(f"Demucs separation failed: {stderr_text}")

            # Demucs outputs to: out_dir/model_name/chunk_name/vocals.wav
            stem_dir = os.path.join(out_dir, self.model, Path(chunk_path).stem)
            vocals_path = os.path.join(stem_dir, 'vocals.wav')

            if not os.path.exists(vocals_path):
                # Try alternative path patterns
                pattern = os.path.join(out_dir, '**', 'vocals.wav')
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    vocals_path = matches[0]
                else:
                    raise FileNotFoundError(f"Vocals file not found after Demucs separation: {vocals_path}")

            vocals_full += AudioSegment.from_wav(vocals_path)

            # Cleanup chunk files
            os.remove(chunk_path)
            if os.path.exists(stem_dir):
                shutil.rmtree(stem_dir)

        out_path = os.path.join(self.temp_dir, f"{Path(input_wav).stem}_vocals.wav")
        vocals_full.export(out_path, format='wav')
        logger.info("Vocals extracted: %s", out_path)
        return out_path
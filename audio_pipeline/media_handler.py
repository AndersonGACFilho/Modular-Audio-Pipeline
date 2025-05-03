from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MediaHandler:
    """
    Discovers media files in a directory and converts them to mono WAV.
    """
    def __init__(self, media_dir: str, temp_dir: str, sample_rate: int = 16000):
        self.media_dir = media_dir
        self.temp_dir = temp_dir
        self.sample_rate = sample_rate

    def find_media_file(self) -> tuple[str, bool] | None:
        """
        Look for the first audio/video file in media_dir.
        Returns (path, is_video) or None.
        """
        audio_exts = {'.mp3', '.m4a', '.wav', '.ogg', '.flac', '.aac'}
        video_exts = {'.mp4', '.avi', '.mov', '.wmv'}

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

        for fname in os.listdir(self.media_dir):
            full = os.path.join(self.media_dir, fname)
            if not os.path.isfile(full):
                continue
            ext = Path(fname).suffix.lower()
            if ext in audio_exts:
                return full, False
            if ext in video_exts:
                return full, True

        logger.error("No valid media file found in %s", self.media_dir)
        return None

    def convert_to_wav(self, input_path: str) -> str:
        """
        Convert input audio/video to mono WAV at sample_rate.
        """
        base = Path(input_path).stem
        out_path = os.path.join(self.temp_dir, f"{base}_{self.sample_rate}Hz.wav")
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(self.sample_rate),
            out_path
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            err = result.stderr.decode(errors='ignore')
            logger.error("FFmpeg conversion error: %s", err)
            raise RuntimeError(f"Failed to convert to WAV: {err}")
        return out_path

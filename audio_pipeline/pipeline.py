import os
import json
import logging
from pathlib import Path

from .media_handler import MediaHandler
from .preprocessor import AudioPreprocessor
from .redundancy import RedundancyRemover
from .separator import VocalSeparator
from .transcriber import WhisperTranscriber
from .diarizer import SpeakerDiarizer
from .vad import VADFilter
from .config import CONFIG

logger = logging.getLogger(__name__)

class AudioPipeline:
    """
    Coordinates all steps of the audio processing and transcription pipeline.
    """

    def __init__(self, config):
        self.config = config
        self.media_dir = config["media_dir"]
        self.temp_dir = os.path.join(self.media_dir, "temp")
        self.results_dir = os.path.join(self.media_dir, "results")
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.media    = MediaHandler(self.media_dir, self.temp_dir, config["sample_rate"])
        self.preproc  = AudioPreprocessor(config["sample_rate"], self.temp_dir)
        self.sep      = VocalSeparator(config["sample_rate"], self.temp_dir)
        self.vad      = VADFilter(
            sample_rate=config["sample_rate"],
            frame_duration_ms=config["frame_duration_ms"],
            padding_duration_ms=config["padding_duration_ms"],
            start_threshold=config["start_threshold"],
            stop_threshold=config["stop_threshold"],
            vad_mode=config["vad_mode"]
        )
        self.transc   = WhisperTranscriber(config["model"], config["language"], config["prompt"])
        self.diarizer = SpeakerDiarizer()
        self.redundancy = RedundancyRemover(config["redundancy_threshold"])

    def run(self):
        found = self.media.find_media_file()
        if not found:
            logger.error("No media file found, aborting.")
            return
        media_file, is_video = found
        base = Path(media_file).stem
        logger.info("Processing %s", media_file)

        # 1. Convert
        ext = Path(media_file).suffix.lower()
        wav = self.media.convert_to_wav(media_file) if is_video or ext != '.wav' else media_file

        # 2. Preprocess
        denoised  = self.preproc.reduce_stationary_noise(wav)
        vocals    = self.sep.extract_vocals(denoised)
        norm      = self.preproc.normalize_audio(vocals)
        loudnorm  = self.preproc.normalize_loudness(norm)
        silence = self.preproc.remove_silence(loudnorm)

        # 3. VAD
        voiced_wav = self.vad.filter_voice(silence, self.results_dir)

        # 4. Transcribe & Diarize
        result     = self.transc.transcribe(voiced_wav)
        diar_steps = self.diarizer.diarize(voiced_wav)

        # 5. Align segments ➔ speaker
        segments   = result.get("segments", [])
        final_out  = []
        for seg in segments:
            start, end, text = seg["start"], seg["end"], seg["text"].strip()
            speaker = "Unknown"
            overlaps = [
                (turn, spk)
                for (turn, _, spk) in diar_steps
                if not (turn.end <= start or turn.start >= end)
            ]
            if overlaps:
                counts = {}
                for (turn, spk) in overlaps:
                    dur = min(turn.end, end) - max(turn.start, start)
                    counts[spk] = counts.get(spk, 0) + dur
                speaker = max(counts, key=counts.get)
            final_out.append({"speaker": speaker, "start": start, "end": end, "text": text})

        final_filtered = self.redundancy.remove(final_out)
        # 6. Save JSON
        out_path = os.path.join(self.results_dir, f"{base}_transcription.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_filtered, f, ensure_ascii=False, indent=2)
        logger.info("Saved transcription ➔ %s", out_path)

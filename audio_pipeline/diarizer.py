from pyannote.audio import Pipeline
import torch
import logging

logger = logging.getLogger(__name__)

class SpeakerDiarizer:
    """
    Performs speaker diarization using pyannote.audio.
    """
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        ).to(device)

    def diarize(self, audio_path: str, min_speakers=2, max_speakers=5):
        logger.info("Diarizing %s", audio_path)
        diar = self.pipeline(audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
        return list(diar.itertracks(yield_label=True))

import whisper
import logging

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """
    Wraps OpenAI Whisper for transcription with a prompt.
    """
    def __init__(self, model_name: str, language: str, prompt: str):
        self.model = whisper.load_model(model_name)
        self.language = language
        self.prompt = prompt

    def transcribe(self, input_wav: str) -> dict:
        """
        Returns Whisper's result dict, including 'segments'.
        """
        logger.info("Transcribing %s", input_wav)
        return self.model.transcribe(
            audio=input_wav,
            language=self.language,
            task='transcribe',
            verbose=True,
            temperature=0.0,
            beam_size=5,
            prompt=self.prompt
        )

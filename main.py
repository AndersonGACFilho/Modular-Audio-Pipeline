import os
import argparse
import logging
from dotenv import load_dotenv
import openai
from huggingface_hub.hf_api import HfFolder

from audio_pipeline.config import CONFIG
from audio_pipeline.pipeline import AudioPipeline

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """
    Load and validate environment variables for APIs.
    """
    load_dotenv()

    hf = os.getenv("HF_TOKEN")
    if not hf:
        logger.error("Missing HF_TOKEN")
        raise EnvironmentError("Set HF_TOKEN")
    HfFolder.save_token(hf)

def parse_args():
    parser = argparse.ArgumentParser(description="Audio Processing Pipeline")
    parser.add_argument("--media-dir", default=CONFIG["media_dir"])
    parser.add_argument("--model", default=CONFIG["model"])
    parser.add_argument("--language", default=CONFIG["language"])
    opts = parser.parse_args()
    CONFIG.update({
        "media_dir": opts.media_dir,
        "model": opts.model,
        "language": opts.language
    })
    return CONFIG

def main():
    config = parse_args()
    setup_environment()
    AudioPipeline(config).run()

if __name__ == "__main__":
    main()

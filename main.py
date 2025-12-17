"""
Audio Processing Pipeline CLI.

Main entry point for running the audio transcription pipeline.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from huggingface_hub import login

from audio_pipeline.config import (
    PipelineConfig,
    DEFAULT_PROMPTS,
    get_default_config
)
from audio_pipeline.pipeline import AudioPipeline
from audio_pipeline.exceptions import AudioPipelineError, ConfigurationError

# Setup logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_environment() -> None:
    """
    Load and validate environment variables for APIs.
    
    Raises:
        EnvironmentError: If required tokens are missing
    """
    load_dotenv()

    import warnings
    warnings.filterwarnings("ignore", message=".*torchaudio._backend.*")
    warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")
    warnings.filterwarnings("ignore", message=".*has been moved to.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
    warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
    warnings.filterwarnings("ignore", message=".*weights_only=False.*")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning(
            "HF_TOKEN not set. Speaker diarization will not work. "
            "Set HF_TOKEN environment variable or add to .env file."
        )
    else:
        try:
            login(token=hf_token, add_to_git_credential=False)
            logger.info("Hugging Face authentication successful")
        except Exception as e:
            logger.warning(f"Hugging Face login failed: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Audio Processing & Transcription Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
              # Process audio in default directory with defaults
              python main.py
            
              # Process specific directory with custom model
              python main.py --media-dir ./recordings --model large-v3
            
              # Process single file with English transcription
              python main.py --input recording.mp3 --language en
            
              # Use configuration file
              python main.py --config config.json
            
              # Disable diarization for single speaker
              python main.py --no-diarization
            
              # Enable vocal separation for audio with music
              python main.py --separate-vocals
        """
    )
    
    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--media-dir", "-d",
        type=str,
        help="Directory containing media files (default: ./files)"
    )
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Specific input file to process"
    )
    input_group.add_argument(
        "--config", "-c",
        type=str,
        help="Path to JSON configuration file"
    )
    
    # Transcription options
    trans_group = parser.add_argument_group("Transcription Options")
    trans_group.add_argument(
        "--model", "-m",
        type=str,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"],
        help="Whisper model to use (default: large-v3-turbo)"
    )
    trans_group.add_argument(
        "--language", "-l",
        type=str,
        help="Language code for transcription (default: pt)"
    )
    trans_group.add_argument(
        "--prompt", "-p",
        type=str,
        help="Initial prompt to guide transcription"
    )
    trans_group.add_argument(
        "--prompt-preset",
        type=str,
        choices=list(DEFAULT_PROMPTS.keys()),
        help="Use a preset prompt"
    )
    
    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--separate-vocals",
        action="store_true",
        help="Enable vocal separation (useful for audio with music)"
    )
    proc_group.add_argument(
        "--auto-separate",
        action="store_true",
        help="Auto-detect if vocal separation is needed"
    )
    proc_group.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization"
    )
    proc_group.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Voice Activity Detection"
    )
    proc_group.add_argument(
        "--no-noise-reduction",
        action="store_true",
        help="Disable noise reduction"
    )
    proc_group.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum expected number of speakers (default: 1)"
    )
    proc_group.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum expected number of speakers (default: 5)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Directory for output files"
    )
    output_group.add_argument(
        "--preserve-timestamps",
        action="store_true",
        default=True,
        help="Preserve original timestamps (default: True)"
    )
    
    # Debug options
    debug_group = parser.add_argument_group("Debug Options")
    debug_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    debug_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    debug_group.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't cleanup temporary files after processing"
    )
    
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """
    Build pipeline configuration from arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        PipelineConfig instance
    """
    # Start with config file or defaults
    if args.config and os.path.exists(args.config):
        config = PipelineConfig.from_json(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.media_dir:
        config.media_dir = args.media_dir
    if args.output_dir:
        config.results_dir = args.output_dir
    
    # Transcription options
    if args.model:
        config.transcription.model = args.model
    if args.language:
        config.transcription.language = args.language
    if args.prompt:
        config.transcription.prompt = args.prompt
    elif args.prompt_preset:
        config.transcription.prompt = DEFAULT_PROMPTS[args.prompt_preset]
    
    # Processing options
    if args.separate_vocals:
        config.vocal_separation.enabled = True
    if args.auto_separate:
        config.vocal_separation.auto_detect = True
    if args.no_diarization:
        config.diarization.enabled = False
    if args.no_vad:
        config.vad.enabled = False
    if args.no_noise_reduction:
        config.noise_reduction.enabled = False
    if args.min_speakers:
        config.diarization.min_speakers = args.min_speakers
    if args.max_speakers:
        config.diarization.max_speakers = args.max_speakers
    
    config.preserve_timestamps = args.preserve_timestamps
    
    # Re-run post_init to update paths
    config.__post_init__()
    
    return config


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        # Setup environment
        setup_environment()
        
        # Build configuration
        config = build_config(args)
        
        logger.info(f"Media directory: {config.media_dir}")
        logger.info(f"Model: {config.transcription.model}")
        logger.info(f"Language: {config.transcription.language}")
        
        # Create and run pipeline
        pipeline = AudioPipeline(config)
        
        result = pipeline.run(input_file=args.input)
        
        if result.success:
            logger.info(f"✓ Processing complete!")
            logger.info(f"  Input: {result.input_file}")
            logger.info(f"  Output: {result.output_file}")
            logger.info(f"  Segments: {len(result.segments)}")
            
            # Cleanup if requested
            if not args.no_cleanup:
                pipeline.cleanup()
            
            return 0
        else:
            logger.error(f"✗ Processing failed: {result.error}")
            return 1
            
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except AudioPipelineError as e:
        logger.error(f"Pipeline error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

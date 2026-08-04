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

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEDIA_DIRECTORY = PROJECT_ROOT / "files" / "incoming"

if __package__ in {None, ""}:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch
from dotenv import load_dotenv
from huggingface_hub import login

from audio_pipeline.config import (
    PipelineConfig,
    DEFAULT_PROMPTS,
)
from audio_pipeline.application.pipeline import AudioPipeline
from audio_pipeline.bootstrap import create_audio_pipeline
from audio_pipeline.bootstrap.runtime_preflight import RuntimePreflight
from audio_pipeline.domain.exceptions import AudioPipelineError, ConfigurationError
from shared.observability import TerminalProgress, configure_logging


logger = logging.getLogger(__name__)


def setup_environment() -> None:
    """
    Load and validate environment variables for APIs.
    
    Raises:
        EnvironmentError: If required tokens are missing
    """
    load_dotenv()

    try:
        import numpy as _np
        for _alias in ("NaN", "NAN"):
            if not hasattr(_np, _alias):
                setattr(_np, _alias, _np.nan)
        if not hasattr(_np, 'nan'):
            setattr(_np, 'nan', float('nan'))
    except Exception:
        pass

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
        help=f"Directory containing media files (default: {DEFAULT_MEDIA_DIRECTORY})"
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
        config = PipelineConfig(media_dir=str(DEFAULT_MEDIA_DIRECTORY))

    apply_container_ollama_host_override(config)
    
    # Override with command line arguments
    if args.media_dir:
        config.media_dir = args.media_dir
        config.temp_dir = None
        config.results_dir = None
        config.checkpoint_dir = None
    if args.output_dir:
        config.results_dir = args.output_dir
    
    # Transcription options
    if args.model:
        config.transcription.model = args.model
    if args.language:
        config.transcription.language = args.language
    if args.prompt:
        config.transcription.initial_prompt = args.prompt
    elif args.prompt_preset:
        config.transcription.initial_prompt = DEFAULT_PROMPTS[args.prompt_preset]
    
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


def apply_container_ollama_host_override(
    config: PipelineConfig, *, is_container: Optional[bool] = None
) -> None:
    """Apply the Compose-only host bridge without breaking the local CLI."""
    if is_container is None:
        is_container = Path("/.dockerenv").exists()
    ollama_host = os.getenv("AUDIO_PIPELINE_OLLAMA_HOST")
    if is_container and ollama_host:
        config.llm.ollama_host = ollama_host
        logger.info("Using Docker Ollama host: %s", ollama_host)


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()
    pipeline: Optional[AudioPipeline] = None

    log_level = logging.DEBUG if args.debug else logging.INFO
    console = configure_logging(
        entrypoint=Path(__file__).stem,
        level=log_level,
        log_directory=PROJECT_ROOT / "logs",
    )
    progress = TerminalProgress(console)
    
    try:
        # Setup environment
        setup_environment()
        
        # Build configuration
        config = build_config(args)
        
        logger.info(f"Media directory: {config.media_dir}")
        logger.info(f"Model: {config.transcription.model}")
        logger.info(f"Language: {config.transcription.language}")
        RuntimePreflight().run(config)
        
        # Build the pipeline once, then process the local media queue in order.
        pipeline = create_audio_pipeline(
            config,
            progress_callback=progress.update_stage,
            file_callback=progress.set_file,
        )
        media_files = [args.input] if args.input else pipeline.media.list_media_files()
        progress.set_total_files(len(media_files))
        progress.start()

        all_success = True
        result = None
        for index, media_file in enumerate(media_files, start=1):
            progress.set_file(media_file, index)
            result = pipeline.run(input_file=media_file)
            if result.success:
                logger.info("Completed file %d/%d: %s", index, len(media_files), result.output_file)
            else:
                all_success = False
                logger.error("Failed file %d/%d (%s): %s", index, len(media_files), media_file, result.error)
        
        if all_success:
            logger.info("Queue completed: %d file(s) processed.", len(media_files))
            return 0
        logger.error("Queue completed with failures. Review the per-file errors above.")
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
    finally:
        progress.stop()
        if pipeline is not None and not args.no_cleanup:
            pipeline.cleanup()


if __name__ == "__main__":
    sys.exit(main())

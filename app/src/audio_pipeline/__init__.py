"""
Audio processing and transcription pipeline.
"""

from .config import PipelineConfig, get_default_config

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .application.pipeline import AudioPipeline, PipelineResult

__version__ = "2.0.0"
__author__ = "Anderson GAC Filho"

__all__ = [
    "AudioPipeline",
    "PipelineResult",
    "PipelineConfig",
    "get_default_config",
]


def __getattr__(name: str):
    if name in {"AudioPipeline", "PipelineResult"}:
        from .application.pipeline import AudioPipeline, PipelineResult

        return {
            "AudioPipeline": AudioPipeline,
            "PipelineResult": PipelineResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

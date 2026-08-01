"""Port used by the job use case to process one source asset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ...domain.models import AudioJob

if TYPE_CHECKING:
    from ..pipeline import PipelineResult


class AudioProcessor(Protocol):
    def process(self, job: AudioJob, processing_directory: Path, results_directory: Path) -> PipelineResult: ...

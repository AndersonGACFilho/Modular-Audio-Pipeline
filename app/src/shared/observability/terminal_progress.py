"""Rich progress UI for interactive terminal entrypoints only."""

from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


class TerminalProgress:
    """Keeps the current file, step, and substep visible below terminal logs."""

    _STAGES = (
        "Media discovery", "Media conversion", "Noise reduction", "Vocal separation",
        "Peak normalization", "Loudness normalization", "Silence removal",
        "Voice activity detection", "Transcription", "Diarization", "Speaker alignment",
        "Timestamp mapping", "Redundancy removal", "Segment merging", "Llm analysis",
        "Saving results",
    )

    def __init__(self, console: Console, total_files: int = 1) -> None:
        self._enabled = console.is_terminal
        self._total_files = total_files
        self._file_index = 1
        self._file_name = "Preparing pipeline"
        self._stage = "Starting"
        self._stage_number = 0
        self._task_id: int | None = None
        self._progress = Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TimeElapsedColumn(), console=console, transient=True,
        )

    def start(self) -> None:
        if self._enabled:
            self._progress.start()
            self._task_id = self._progress.add_task(self._description(), total=len(self._STAGES))

    def set_total_files(self, total_files: int) -> None:
        self._total_files = max(total_files, 1)

    def set_file(self, path: str, index: int = 1) -> None:
        if index != self._file_index:
            self._stage = "Starting"
            self._stage_number = 0
        self._file_index = index
        self._file_name = Path(path).name
        self._refresh()

    def update_stage(self, stage: str) -> None:
        self._stage = stage
        stage_name = stage.split(" - ", maxsplit=1)[0].casefold()
        for index, known_stage in enumerate(self._STAGES, start=1):
            if known_stage.casefold() == stage_name:
                self._stage_number = index
                break
        self._refresh()

    def stop(self) -> None:
        if self._enabled:
            self._progress.stop()

    def _description(self) -> str:
        return (
            f"Arquivo {self._file_index}/{self._total_files}: {self._file_name} - "
            f"{self._stage} ({self._stage_number}/{len(self._STAGES)})"
        )

    def _refresh(self) -> None:
        if self._enabled and self._task_id is not None:
            self._progress.update(
                self._task_id, completed=self._stage_number, description=self._description()
            )

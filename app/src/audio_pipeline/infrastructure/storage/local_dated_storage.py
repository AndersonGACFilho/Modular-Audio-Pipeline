"""Local, date-partitioned storage for large audio-processing jobs."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


class LocalDatedStorage:
    def __init__(self, data_root: str | Path) -> None:
        self._root = Path(data_root).resolve()

    def _dated_directory(self, area: str, job_id: str, created_at: datetime) -> Path:
        return self._root / area / created_at.strftime("%Y") / created_at.strftime("%m") / created_at.strftime("%d") / job_id

    def incoming_path(self, job_id: str, created_at: datetime, extension: str) -> Path:
        suffix = extension if extension.startswith(".") else f".{extension}"
        directory = self._dated_directory("incoming", job_id, created_at)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"source{suffix.lower()}"

    def prepare_processing(self, job_id: str) -> Path:
        directory = self._root / "processing" / job_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def results_directory(self, job_id: str, created_at: datetime) -> Path:
        directory = self._dated_directory("results", job_id, created_at)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def failed_directory(self, job_id: str, created_at: datetime) -> Path:
        directory = self._dated_directory("failed", job_id, created_at)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def cleanup_processing(self, job_id: str) -> None:
        directory = self._root / "processing" / job_id
        if directory.exists():
            shutil.rmtree(directory)

"""Domain model and lifecycle rules for an audio-processing job."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    UPLOADING = "uploading"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class MediaAsset:
    path: str
    original_name: str
    size_bytes: int
    sha256: str | None = None


@dataclass(frozen=True)
class JobResult:
    output_path: str
    segment_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobError:
    message: str
    error_type: str


@dataclass
class AudioJob:
    job_id: str
    source: MediaAsset
    status: JobStatus = JobStatus.QUEUED
    options: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    attempt_count: int = 0
    result: JobResult | None = None
    error: JobError | None = None

    def mark_processing(self) -> None:
        if self.status is not JobStatus.QUEUED:
            raise ValueError(f"Cannot process a job in {self.status.value!r} state.")
        self.status = JobStatus.PROCESSING
        self.started_at = utc_now()
        self.updated_at = self.started_at
        self.attempt_count += 1

    def mark_completed(self, result: JobResult) -> None:
        if self.status is not JobStatus.PROCESSING:
            raise ValueError(f"Cannot complete a job in {self.status.value!r} state.")
        self.status = JobStatus.COMPLETED
        self.result = result
        self.error = None
        self.completed_at = utc_now()
        self.updated_at = self.completed_at

    def mark_failed(self, error: JobError) -> None:
        if self.status is not JobStatus.PROCESSING:
            raise ValueError(f"Cannot fail a job in {self.status.value!r} state.")
        self.status = JobStatus.FAILED
        self.error = error
        self.completed_at = utc_now()
        self.updated_at = self.completed_at

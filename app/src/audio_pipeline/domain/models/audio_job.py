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


@dataclass(frozen=True)
class TranscriptionOptions:
    """Immutable ASR options captured when a job is submitted."""

    language: str = "pt"
    locale: str = "pt-BR"
    initial_prompt: str | None = None
    hotwords: tuple[str, ...] = ()
    condition_on_previous_text: bool = False


@dataclass(frozen=True)
class AnalysisOptions:
    """Immutable LLM-analysis options captured when a job is submitted."""

    profile_id: str | None = None
    prompt: str | None = None
    output_language: str = "pt-BR"


@dataclass(frozen=True)
class AudioJobOptions:
    transcription: TranscriptionOptions = field(default_factory=TranscriptionOptions)
    analysis: AnalysisOptions = field(default_factory=AnalysisOptions)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "AudioJobOptions":
        value = value or {}
        transcription = value.get("transcription", {})
        analysis = value.get("analysis", {})
        return cls(
            transcription=TranscriptionOptions(
                language=transcription.get("language", "pt"),
                locale=transcription.get("locale", "pt-BR"),
                initial_prompt=transcription.get("initial_prompt"),
                hotwords=tuple(transcription.get("hotwords", ())),
                condition_on_previous_text=transcription.get("condition_on_previous_text", False),
            ),
            analysis=AnalysisOptions(
                profile_id=analysis.get("profile_id"),
                prompt=analysis.get("prompt"),
                output_language=analysis.get("output_language", "pt-BR"),
            ),
        )


@dataclass
class AudioJob:
    job_id: str
    source: MediaAsset
    status: JobStatus = JobStatus.QUEUED
    options: AudioJobOptions = field(default_factory=AudioJobOptions)
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

"""MongoDB implementation of the audio-job repository port."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from ....domain.models import AudioJob, JobError, JobResult, JobStatus, MediaAsset


class MongoDBJobRepository:
    def __init__(self, uri: str, database: str, collection: str) -> None:
        try:
            from pymongo import MongoClient, ReturnDocument
        except ImportError as error:
            raise RuntimeError("MongoDB support requires 'pymongo'. Run 'uv sync'.") from error

        self._return_document = ReturnDocument
        self._collection = MongoClient(uri)[database][collection]
        self._collection.create_index([("status", 1), ("updated_at", 1)])
        self._collection.create_index("created_at")

    def create(self, job: AudioJob) -> None:
        self._collection.insert_one(self._to_document(job))

    def get(self, job_id: str) -> AudioJob | None:
        document = self._collection.find_one({"_id": job_id})
        return self._from_document(document) if document else None

    def claim(self, job_id: str, worker_id: str, lease_until: datetime) -> AudioJob | None:
        now = datetime.now(timezone.utc)
        document = self._collection.find_one_and_update(
            {"_id": job_id, "status": JobStatus.QUEUED.value},
            {"$set": {"status": JobStatus.PROCESSING.value, "worker_id": worker_id, "started_at": now, "updated_at": now, "lease_until": lease_until}, "$inc": {"attempt_count": 1}},
            return_document=self._return_document.AFTER,
        )
        return self._from_document(document) if document else None

    def mark_completed(self, job_id: str, result: JobResult) -> None:
        now = datetime.now(timezone.utc)
        self._collection.update_one({"_id": job_id, "status": JobStatus.PROCESSING.value}, {"$set": {"status": JobStatus.COMPLETED.value, "result": asdict(result), "error": None, "completed_at": now, "updated_at": now}, "$unset": {"lease_until": "", "worker_id": ""}})

    def mark_failed(self, job_id: str, error: JobError) -> None:
        now = datetime.now(timezone.utc)
        self._collection.update_one({"_id": job_id, "status": JobStatus.PROCESSING.value}, {"$set": {"status": JobStatus.FAILED.value, "error": asdict(error), "completed_at": now, "updated_at": now}, "$unset": {"lease_until": "", "worker_id": ""}})

    @staticmethod
    def _to_document(job: AudioJob) -> dict[str, Any]:
        document = asdict(job)
        document["_id"] = document.pop("job_id")
        document["status"] = job.status.value
        return document

    @staticmethod
    def _from_document(document: dict[str, Any]) -> AudioJob:
        return AudioJob(
            job_id=document["_id"],
            source=MediaAsset(**document["source"]),
            status=JobStatus(document["status"]),
            options=document.get("options", {}),
            created_at=document["created_at"],
            updated_at=document["updated_at"],
            started_at=document.get("started_at"),
            completed_at=document.get("completed_at"),
            attempt_count=document.get("attempt_count", 0),
            result=JobResult(**document["result"]) if document.get("result") else None,
            error=JobError(**document["error"]) if document.get("error") else None,
        )

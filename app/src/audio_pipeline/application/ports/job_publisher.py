"""Outbound port for scheduling an audio job."""

from typing import Protocol


class JobPublisher(Protocol):
    def publish(self, job_id: str) -> None: ...

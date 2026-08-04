"""Outbound port for scheduling an audio job."""

from typing import Protocol


class JobPublisher(Protocol):
    async def publish(self, job_id: str) -> None: ...

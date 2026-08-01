"""Message bus port. Messages contain job identifiers only."""

from collections.abc import Callable
from typing import Protocol


class JobQueue(Protocol):
    def publish(self, job_id: str) -> None: ...
    def consume(self, handler: Callable[[str], None]) -> None: ...

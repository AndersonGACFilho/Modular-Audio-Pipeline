"""RabbitMQ worker entry point; separate from the synchronous local CLI."""

from __future__ import annotations

import asyncio

from ..application.use_cases import ProcessAudioJob
from ..bootstrap import AudioPipelineProcessor
from ..configuration import JobSettings
from ..infrastructure.messaging.rabbitmq.job_queue import RabbitMQJobQueue
from ..infrastructure.persistence.mongodb.job_repository import MongoDBJobRepository
from ..infrastructure.storage.local_dated_storage import LocalDatedStorage
from shared.observability import configure_logging


async def run() -> int:
    settings = JobSettings.from_environment()
    configure_logging(entrypoint="rabbitmq_worker", log_directory=settings.data_root / "logs")
    repository = MongoDBJobRepository(settings.mongodb_uri, settings.mongodb_database, settings.mongodb_collection)
    await repository.ensure_indexes()
    queue = RabbitMQJobQueue(settings.rabbitmq_url, settings.rabbitmq_queue, settings.rabbitmq_dead_letter_queue)
    use_case = ProcessAudioJob(repository, LocalDatedStorage(settings.data_root), AudioPipelineProcessor(), settings.lease_seconds)
    try:
        await queue.consume(use_case.execute)
    finally:
        await repository.close()
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())

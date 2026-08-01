"""RabbitMQ worker entry point; separate from the synchronous local CLI."""

from __future__ import annotations

from ..application.use_cases import ProcessAudioJob
from ..configuration import JobSettings
from ..infrastructure.messaging.rabbitmq.job_queue import RabbitMQJobQueue
from ..infrastructure.persistence.mongodb.job_repository import MongoDBJobRepository
from ..infrastructure.storage.local_dated_storage import LocalDatedStorage
from shared.observability import configure_logging


def main() -> int:
    settings = JobSettings.from_environment()
    configure_logging(entrypoint="rabbitmq_worker", log_directory=settings.data_root / "logs")
    repository = MongoDBJobRepository(settings.mongodb_uri, settings.mongodb_database, settings.mongodb_collection)
    queue = RabbitMQJobQueue(settings.rabbitmq_url, settings.rabbitmq_queue, settings.rabbitmq_dead_letter_queue)
    use_case = ProcessAudioJob(repository, LocalDatedStorage(settings.data_root), settings.lease_seconds)
    queue.consume(use_case.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

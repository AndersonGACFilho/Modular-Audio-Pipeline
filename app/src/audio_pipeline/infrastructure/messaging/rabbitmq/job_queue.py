"""Synchronous RabbitMQ queue adapter for CPU/GPU-bound audio jobs."""

from __future__ import annotations

import json
from collections.abc import Callable


class RabbitMQJobQueue:
    def __init__(self, url: str, queue_name: str, dead_letter_queue: str) -> None:
        try:
            import pika
        except ImportError as error:
            raise RuntimeError("RabbitMQ support requires 'pika'. Run 'uv sync'.") from error

        self._pika = pika
        self._parameters = pika.URLParameters(url)
        self._queue_name = queue_name
        self._dead_letter_queue = dead_letter_queue

    def _channel(self):
        connection = self._pika.BlockingConnection(self._parameters)
        channel = connection.channel()
        channel.queue_declare(queue=self._dead_letter_queue, durable=True)
        channel.queue_declare(queue=self._queue_name, durable=True, arguments={"x-dead-letter-exchange": "", "x-dead-letter-routing-key": self._dead_letter_queue})
        return connection, channel

    def publish(self, job_id: str) -> None:
        connection, channel = self._channel()
        try:
            channel.basic_publish(exchange="", routing_key=self._queue_name, body=json.dumps({"job_id": job_id}), properties=self._pika.BasicProperties(delivery_mode=self._pika.DeliveryMode.Persistent, content_type="application/json"))
        finally:
            connection.close()

    def consume(self, handler: Callable[[str], None]) -> None:
        connection, channel = self._channel()
        channel.basic_qos(prefetch_count=1)

        def on_message(channel, method, _properties, body) -> None:
            try:
                job_id = json.loads(body)["job_id"]
                handler(job_id)
            except Exception:
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                return
            channel.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_consume(queue=self._queue_name, on_message_callback=on_message)
        try:
            channel.start_consuming()
        finally:
            connection.close()

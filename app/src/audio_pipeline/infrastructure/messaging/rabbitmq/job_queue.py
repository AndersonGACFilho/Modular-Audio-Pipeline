"""Async RabbitMQ adapter with durable messages and dead-letter routing."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable


class RabbitMQJobQueue:
    def __init__(self, url: str, queue_name: str, dead_letter_queue: str) -> None:
        try:
            import aio_pika
        except ImportError as error:
            raise RuntimeError("RabbitMQ support requires 'aio-pika'. Run 'uv sync'.") from error

        self._aio_pika = aio_pika
        self._url = url
        self._queue_name = queue_name
        self._dead_letter_queue = dead_letter_queue

    async def _declare_queues(self, channel):
        await channel.declare_queue(self._dead_letter_queue, durable=True)
        return await channel.declare_queue(
            self._queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": self._dead_letter_queue,
            },
        )

    async def publish(self, job_id: str) -> None:
        connection = await self._aio_pika.connect_robust(self._url)
        try:
            channel = await connection.channel(publisher_confirms=True)
            await self._declare_queues(channel)
            message = self._aio_pika.Message(
                json.dumps({"schema_version": 1, "job_id": job_id}).encode(),
                delivery_mode=self._aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json",
            )
            await channel.default_exchange.publish(message, routing_key=self._queue_name)
        finally:
            await connection.close()

    async def consume(self, handler: Callable[[str], Awaitable[None]]) -> None:
        connection = await self._aio_pika.connect_robust(self._url)
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        queue = await self._declare_queues(channel)

        async def on_message(message) -> None:
            try:
                payload = json.loads(message.body)
                await handler(payload["job_id"])
            except Exception:
                await message.reject(requeue=False)
            else:
                await message.ack()

        await queue.consume(on_message)
        try:
            await asyncio.Future()
        finally:
            await connection.close()

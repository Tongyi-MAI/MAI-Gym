from __future__ import annotations

import asyncio
import gzip
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class CompressionPool:
    def __init__(self, workers: int = 4, level: int = 1) -> None:
        self.workers = workers
        self.level = level
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="http-gzip")
        self.slots = asyncio.Semaphore(workers)
        self.active = 0
        self.waiting = 0
        self.completed = 0
        self.failed = 0

    async def compress(self, body: bytes) -> bytes:
        self.waiting += 1
        try:
            await self.slots.acquire()
        finally:
            self.waiting -= 1
        self.active += 1
        try:
            future = asyncio.get_running_loop().run_in_executor(
                self.executor,
                self._compress,
                body,
            )
        except BaseException:
            self.active -= 1
            self.slots.release()
            raise

        def finished(future: asyncio.Future) -> None:

            self.active -= 1
            self.slots.release()
            if future.cancelled() or future.exception() is not None:
                self.failed += 1
            else:
                self.completed += 1

        future.add_done_callback(finished)
        return await asyncio.shield(future)

    def _compress(self, body: bytes) -> bytes:
        return gzip.compress(body, compresslevel=self.level, mtime=0)

    def snapshot(self) -> dict:
        return {
            "workers": self.workers,
            "level": self.level,
            "active": self.active,
            "waiting": self.waiting,
            "completed": self.completed,
            "failed": self.failed,
        }

    def close(self) -> None:

        self.executor.shutdown(wait=False, cancel_futures=True)


def accepts_gzip(value: str) -> bool:
    encodings = {}
    for entry in value.lower().split(","):
        parts = entry.strip().split(";")
        quality = 1.0
        for part in parts[1:]:
            if part.strip().startswith("q="):
                try:
                    quality = float(part.strip()[2:])
                except ValueError:
                    quality = 0.0
        encodings[parts[0].strip()] = quality
    return encodings.get("gzip", encodings.get("*", 0.0)) > 0


class BoundedGZipMiddleware:
    def __init__(self, app: ASGIApp, pool: CompressionPool, minimum_size: int = 1024) -> None:
        self.app = app
        self.pool = pool
        self.minimum_size = minimum_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("path") == "/health"
            or scope.get("method") == "HEAD"
        ):
            await self.app(scope, receive, send)
            return
        request_headers = Headers(scope=scope)
        start: Message | None = None
        started = False

        async def send_response(message: Message) -> None:
            nonlocal start, started
            if message["type"] == "http.response.start":
                start = {**message, "headers": list(message.get("headers", []))}
                return
            if start is not None and not started:
                started = True
                headers = MutableHeaders(raw=start["headers"])
                body = message.get("body", b"")
                eligible = (
                    message["type"] == "http.response.body"
                    and not message.get("more_body", False)
                    and len(body) >= self.minimum_size
                    and "content-encoding" not in headers
                    and "content-range" not in headers
                    and "range" not in request_headers
                    and start["status"] not in (204, 206, 304)
                    and not headers.get("content-type", "").startswith("text/event-stream")
                    and "no-transform" not in headers.get("cache-control", "").lower()
                )
                if eligible:
                    headers.add_vary_header("Accept-Encoding")
                    if accepts_gzip(request_headers.get("accept-encoding", "")):
                        body = await self.pool.compress(body)
                        headers["Content-Encoding"] = "gzip"
                        headers["Content-Length"] = str(len(body))
                        message = {**message, "body": body}
                await send(start)
            await send(message)

        await self.app(scope, receive, send_response)


class EventLoopWatchdog:
    def __init__(self, interval: float = 0.5, warning_seconds: float = 1.0) -> None:
        self.interval = interval
        self.warning_seconds = warning_seconds
        self.samples = 0
        self.last_lag_seconds = 0.0
        self.max_lag_seconds = 0.0
        self.stalls = 0

    def snapshot(self) -> dict:
        return {
            "interval_seconds": self.interval,
            "samples": self.samples,
            "last_lag_seconds": self.last_lag_seconds,
            "max_lag_seconds": self.max_lag_seconds,
            "stalls": self.stalls,
        }

    async def run(self) -> None:
        last_warning = float("-inf")
        while True:
            expected = time.monotonic() + self.interval
            await asyncio.sleep(self.interval)
            now = time.monotonic()
            self.last_lag_seconds = max(0.0, now - expected)
            self.max_lag_seconds = max(self.max_lag_seconds, self.last_lag_seconds)
            self.samples += 1
            if self.last_lag_seconds >= self.warning_seconds:
                self.stalls += 1
                if now - last_warning >= 30:
                    logger.warning(
                        "[event-loop] scheduling delayed %.3fs (max=%.3fs stalls=%d)",
                        self.last_lag_seconds,
                        self.max_lag_seconds,
                        self.stalls,
                    )
                    last_warning = now

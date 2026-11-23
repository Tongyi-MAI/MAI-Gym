from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import os
import threading
from dataclasses import dataclass
from io import SEEK_CUR, SEEK_END, SEEK_SET
from typing import TYPE_CHECKING

import aiofiles
import aiofiles.os
from aiooss2 import adapter as _aiooss2_adapter

if "decode" in _aiooss2_adapter.AsyncPayload.__abstractmethods__:

    def _decode_stub(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        raise NotImplementedError("AsyncPayload.decode is not implemented")

    _aiooss2_adapter.AsyncPayload.decode = _decode_stub
    _aiooss2_adapter.AsyncPayload.__abstractmethods__ = frozenset(
        m for m in _aiooss2_adapter.AsyncPayload.__abstractmethods__ if m != "decode"
    )

from ossfs import AioOSSFileSystem, OSSFileSystem

if TYPE_CHECKING:
    from mai_gym.models.config import OSSConfig

logger = logging.getLogger(__name__)

OSS_SCHEME = "oss://"
_OSS_UPLOAD_CHUNK_BYTES = 256 * 1024


class _ChunkedBytesReader:
    def __init__(self, data: bytes | memoryview) -> None:
        self._data = memoryview(data)
        self._position = 0

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: slice) -> _ChunkedBytesReader:
        return _ChunkedBytesReader(self._data[key])

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = SEEK_SET) -> int:
        if whence == SEEK_SET:
            position = offset
        elif whence == SEEK_CUR:
            position = self._position + offset
        elif whence == SEEK_END:
            position = len(self._data) + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self._position >= len(self._data):
            return b""
        if size < 0 or size > _OSS_UPLOAD_CHUNK_BYTES:
            size = _OSS_UPLOAD_CHUNK_BYTES
        end = min(len(self._data), self._position + size)
        chunk = self._data[self._position : end].tobytes()
        self._position = end
        return chunk


_AsyncFSKey = tuple[asyncio.AbstractEventLoop, str, str, str, int]
_FS_BY_LOOP: dict[_AsyncFSKey, AioOSSFileSystem] = {}
_FS_BY_LOOP_LOCK = threading.Lock()


@dataclass(slots=True)
class _OSSGetGate:
    limit: int
    semaphore: asyncio.Semaphore
    throttle_logged: bool = False


_OSS_GET_GATE_BY_LOOP: dict[asyncio.AbstractEventLoop, _OSSGetGate] = {}
_OSS_GET_GATE_LOCK = threading.Lock()


_SYNC_FS: OSSFileSystem | None = None


def is_oss_path(path: str) -> bool:

    return path.startswith(OSS_SCHEME)


def _get_oss_get_gate() -> _OSSGetGate | None:

    raw_limit = os.environ.get("MAI_GYM_MAX_INFLIGHT_OSS_GET", "").strip()
    if not raw_limit:
        return None
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise ValueError(
            f"MAI_GYM_MAX_INFLIGHT_OSS_GET must be a positive integer; got {raw_limit!r}"
        ) from exc
    if limit < 1:
        raise ValueError(
            f"MAI_GYM_MAX_INFLIGHT_OSS_GET must be a positive integer; got {raw_limit!r}"
        )

    loop = asyncio.get_running_loop()
    with _OSS_GET_GATE_LOCK:
        gate = _OSS_GET_GATE_BY_LOOP.get(loop)
        if gate is None:
            gate = _OSSGetGate(limit=limit, semaphore=asyncio.Semaphore(limit))
            _OSS_GET_GATE_BY_LOOP[loop] = gate
            logger.info(
                "Created process-wide OSS GET gate for loop %d (cap=%d)",
                id(loop),
                limit,
            )
        elif gate.limit != limit:
            raise RuntimeError(
                "MAI_GYM_MAX_INFLIGHT_OSS_GET changed after the event-loop "
                f"gate was created ({gate.limit} -> {limit})"
            )
    return gate


def get_async_fs(cfg: OSSConfig) -> AioOSSFileSystem:

    loop = asyncio.get_running_loop()
    pool_size = int(getattr(cfg, "connection_pool_size", 20))
    cache_key = (
        loop,
        cfg.endpoint,
        cfg.access_key_id,
        cfg.access_key_secret,
        pool_size,
    )
    with _FS_BY_LOOP_LOCK:
        fs = _FS_BY_LOOP.get(cache_key)
    if fs is not None:
        return fs

    candidate = AioOSSFileSystem(
        endpoint=cfg.endpoint,
        key=cfg.access_key_id,
        secret=cfg.access_key_secret,
        psize=pool_size,
        skip_instance_cache=True,
        asynchronous=True,
    )

    candidate.asynchronous = True
    candidate._loop = None

    with _FS_BY_LOOP_LOCK:
        fs = _FS_BY_LOOP.setdefault(cache_key, candidate)

    if fs is candidate:
        logger.info(
            "Created AioOSSFileSystem singleton for loop %d (endpoint=%s, connection_pool_size=%d)",
            id(loop),
            cfg.endpoint,
            pool_size,
        )
    return fs


async def close_async_fs() -> None:

    loop = asyncio.get_running_loop()
    with _FS_BY_LOOP_LOCK:
        keys = [key for key in _FS_BY_LOOP if key[0] is loop]
        by_identity = {id(_FS_BY_LOOP[key]): _FS_BY_LOOP.pop(key) for key in keys}
        filesystems = list(by_identity.values())
    with _OSS_GET_GATE_LOCK:
        _OSS_GET_GATE_BY_LOOP.pop(loop, None)

    for fs in filesystems:
        session = getattr(fs, "_session", None)
        try:
            if session is not None and not session.closed:
                await session.close()
        except Exception:
            logger.warning(
                "Failed to close AioOSSFileSystem session for loop %d",
                id(loop),
                exc_info=True,
            )
        finally:
            fs._session = None


async def write_bytes(
    path: str,
    data: bytes,
    *,
    fs: AioOSSFileSystem | None = None,
) -> None:

    if is_oss_path(path):
        if fs is None:
            raise ValueError(f"OSS path {path!r} requires an AioOSSFileSystem (fs=)")
        await fs._pipe_file(path, _ChunkedBytesReader(data))
        return
    parent = os.path.dirname(path)
    if parent:
        await aiofiles.os.makedirs(parent, exist_ok=True)
    async with aiofiles.open(path, "wb") as f:
        await f.write(data)


async def write_text(
    path: str,
    text: str,
    *,
    fs: AioOSSFileSystem | None = None,
) -> None:

    await write_bytes(path, text.encode("utf-8"), fs=fs)


async def read_bytes(
    path: str,
    *,
    fs: AioOSSFileSystem | None = None,
    timeout_s: float | None = None,
) -> bytes:

    if is_oss_path(path):
        if fs is None:
            raise ValueError(f"OSS path {path!r} requires an AioOSSFileSystem (fs=)")
        gate = _get_oss_get_gate()
        if gate is not None and gate.semaphore.locked() and not gate.throttle_logged:
            gate.throttle_logged = True
            logger.info(
                "OSS GET in-flight gate engaged (cap=%d); queueing new reads",
                gate.limit,
            )
        cm = gate.semaphore if gate is not None else contextlib.nullcontext()
        async with cm:
            if timeout_s is None:
                return await fs._cat_file(path)
            async with asyncio.timeout(timeout_s):
                return await fs._cat_file(path)
    async with aiofiles.open(path, "rb") as f:
        return await f.read()


async def fetch_image_b64(
    uri: str,
    *,
    fs: AioOSSFileSystem | None = None,
    timeout_s: float | None = None,
) -> str:

    return base64.b64encode(await read_bytes(uri, fs=fs, timeout_s=timeout_s)).decode("ascii")


async def makedirs(
    path: str,
    *,
    fs: AioOSSFileSystem | None = None,
    exist_ok: bool = True,
) -> None:

    if is_oss_path(path):
        return
    await aiofiles.os.makedirs(path, exist_ok=exist_ok)


async def exists(
    path: str,
    *,
    fs: AioOSSFileSystem | None = None,
) -> bool:

    if is_oss_path(path):
        if fs is None:
            raise ValueError(f"OSS path {path!r} requires an AioOSSFileSystem (fs=)")
        return await fs._exists(path)
    return await aiofiles.os.path.exists(path)


async def delete(
    path: str,
    *,
    fs: AioOSSFileSystem | None = None,
) -> None:

    if is_oss_path(path):
        if fs is None:
            raise ValueError(f"OSS path {path!r} requires an AioOSSFileSystem (fs=)")
        with contextlib.suppress(FileNotFoundError):
            await fs._rm(path)
        return
    with contextlib.suppress(FileNotFoundError):
        await aiofiles.os.remove(path)


def get_sync_fs(cfg: OSSConfig, *, connection_pool_size: int | None = None) -> OSSFileSystem:

    global _SYNC_FS
    if _SYNC_FS is None:
        if connection_pool_size is not None:
            import oss2.defaults

            oss2.defaults.connection_pool_size = connection_pool_size
        _SYNC_FS = OSSFileSystem(
            endpoint=cfg.endpoint,
            key=cfg.access_key_id,
            secret=cfg.access_key_secret,
        )
        logger.info(
            "Created sync OSSFileSystem singleton (endpoint=%s, pool=%s)",
            cfg.endpoint,
            connection_pool_size or "default",
        )
    return _SYNC_FS


def sync_write_bytes(
    path: str,
    data: bytes,
    *,
    fs: OSSFileSystem | None = None,
) -> None:
    if is_oss_path(path):
        if fs is None:
            raise ValueError(f"OSS path {path!r} requires an OSSFileSystem (fs=)")
        fs.pipe_file(path, data)
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def sync_write_text(
    path: str,
    text: str,
    *,
    fs: OSSFileSystem | None = None,
) -> None:
    sync_write_bytes(path, text.encode("utf-8"), fs=fs)


def sync_makedirs(
    path: str,
    *,
    fs: OSSFileSystem | None = None,
    exist_ok: bool = True,
) -> None:
    if is_oss_path(path):
        return
    os.makedirs(path, exist_ok=exist_ok)


def join(base: str, *parts: str) -> str:

    out = base.rstrip("/")
    for p in parts:
        out = f"{out}/{p.strip('/')}"
    return out

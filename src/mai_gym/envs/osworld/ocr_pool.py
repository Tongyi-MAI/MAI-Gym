from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_installed = False
_install_lock = threading.Lock()
_readers: dict[tuple[str, ...], Any] = {}
_readers_lock = threading.Lock()


def _concurrency() -> int:
    try:
        return max(1, int(os.environ.get("MAI_GYM_OCR_CONCURRENCY", "4")))
    except ValueError:
        return 4


_ocr_sem = threading.BoundedSemaphore(_concurrency())


def install_ocr_pool() -> None:

    global _installed
    if _installed:
        return
    with _install_lock:
        if _installed:
            return
        try:
            import easyocr
        except Exception as exc:
            logger.debug("install_ocr_pool: easyocr unavailable (%s)", exc)
            _installed = True
            return

        orig_reader = easyocr.Reader

        worker_threads = os.environ.get("MAI_GYM_OSWORLD_EVAL_WORKER_THREADS")
        if worker_threads is not None:
            import torch

            torch.set_num_threads(max(1, int(worker_threads)))
            torch.set_num_interop_threads(1)

        def _shared_reader(langs: Any, **kwargs: Any) -> Any:
            key = tuple(langs) if isinstance(langs, (list, tuple)) else (str(langs),)
            r = _readers.get(key)
            if r is None:
                with _readers_lock:
                    r = _readers.get(key)
                    if r is None:
                        kwargs.setdefault("gpu", False)
                        r = orig_reader(list(key), **kwargs)
                        _readers[key] = r
                        logger.info(
                            "[ocr_pool] built + cached easyocr Reader langs=%s",
                            key,
                        )
            return r

        class _PooledReader:
            def __init__(self, langs: Any = None, **kwargs: Any) -> None:
                self._r = _shared_reader(["en"] if langs is None else langs, **kwargs)

            def readtext(self, *args: Any, **kwargs: Any) -> Any:
                with _ocr_sem:
                    return self._r.readtext(*args, **kwargs)

            def __getattr__(self, name: str) -> Any:

                return getattr(self._r, name)

        easyocr.Reader = _PooledReader
        _installed = True
        logger.info(
            "[ocr_pool] installed cached+bounded easyocr.Reader (MAI_GYM_OCR_CONCURRENCY=%d)",
            _concurrency(),
        )

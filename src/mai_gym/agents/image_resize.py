from __future__ import annotations

import asyncio

from mai_gym.utils.images import ImageResizer

__all__ = ["ImageResizer", "ResizeScreenshotMixin"]


class ResizeScreenshotMixin:
    _resizer: ImageResizer

    async def _resize_screenshot_for_model(self, b64_str: str) -> str:
        resizer = getattr(self, "_resizer", None)
        if not b64_str or resizer is None or not resizer.enabled:
            return b64_str
        return await asyncio.to_thread(resizer.resize_b64, b64_str)

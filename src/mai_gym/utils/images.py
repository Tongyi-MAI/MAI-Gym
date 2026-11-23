import base64
import logging
import math
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

logger = logging.getLogger(__name__)


def b64_to_pil(
    b64_str: str,
    fallback_size: tuple[int, int] = (1920, 1080),
    fallback_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:

    if not b64_str:
        return Image.new("RGB", fallback_size, color=fallback_color)
    return Image.open(BytesIO(base64.b64decode(b64_str)))


def pil_to_b64(img: Image.Image, fmt: str = "PNG", quality: int | None = None) -> str:

    fmt = fmt.upper()
    save_fmt = "JPEG" if fmt == "JPG" else fmt
    kwargs: dict[str, int] = {}
    if save_fmt in ("WEBP", "JPEG"):
        if save_fmt == "JPEG" and img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        elif img.mode == "P":
            img = img.convert("RGBA")
        if quality is not None:
            kwargs["quality"] = int(quality)
    buf = BytesIO()
    img.save(buf, format=save_fmt, **kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sniff_format(b64_str: str) -> str:

    if not b64_str:
        return "png"
    try:
        head = base64.b64decode(b64_str[:24])
    except Exception:
        return "png"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return "png"


def to_data_url(b64_str: str) -> str:

    return f"data:image/{sniff_format(b64_str)};base64,{b64_str}"


def transcode_b64(b64_str: str, fmt: str = "PNG", quality: int | None = None) -> str:

    if not b64_str:
        return b64_str
    target = "jpeg" if fmt.upper() in ("JPG", "JPEG") else fmt.lower()
    if sniff_format(b64_str) == target:
        return b64_str
    return pil_to_b64(b64_to_pil(b64_str), fmt=fmt, quality=quality)


def resize_max_pixels(img: Image.Image, max_pixels: int) -> Image.Image:

    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = math.sqrt(max_pixels / (w * h))
    new_w, new_h = max(32, int(w * scale)), max(32, int(h * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:

    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {height} / {width}")
    h_bar = _round_by_factor(height, factor)
    w_bar = _round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


@dataclass(frozen=True)
class ImageResizer:
    mode: str = "none"

    width: int | None = None
    height: int | None = None

    max_long_side: int | None = None

    max_pixels: int | None = None
    min_pixels: int = 0
    factor: int = 28

    fmt: str = "PNG"

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def resize_pil(self, img: Image.Image) -> Image.Image:

        w, h = img.size
        if self.mode == "fixed":
            scale = min(self.width / w, self.height / h)
            if scale >= 1.0:
                return img
            return img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
        if self.mode == "longest_edge":
            longest = max(w, h)
            if longest <= self.max_long_side:
                return img
            scale = self.max_long_side / longest
            return img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
        if self.mode == "smart_resize":
            new_h, new_w = smart_resize(
                h,
                w,
                factor=self.factor,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            if (new_w, new_h) == (w, h):
                return img
            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img

    def resize_b64(self, b64_str: str) -> str:

        if not b64_str or not self.enabled:
            return b64_str
        try:
            img = b64_to_pil(b64_str)
            out = self.resize_pil(img)
        except ValueError as exc:
            logger.warning("ImageResizer skipped resize (%s): %s", self.mode, exc)
            return b64_str
        if out is img:
            return b64_str
        return pil_to_b64(out, fmt=self.fmt)

    @classmethod
    def from_config(
        cls,
        *,
        resize_width: int | None = None,
        resize_height: int | None = None,
        resize_longest_edge: int | None = None,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
        image_factor: int | None = None,
        fmt: str = "PNG",
    ) -> "ImageResizer":

        if max_pixels is not None:
            r = cls(
                mode="smart_resize",
                max_pixels=int(max_pixels),
                min_pixels=int(min_pixels) if min_pixels is not None else 0,
                factor=int(image_factor) if image_factor is not None else 28,
                fmt=fmt,
            )
        elif resize_longest_edge is not None:
            r = cls(mode="longest_edge", max_long_side=int(resize_longest_edge), fmt=fmt)
        elif resize_width is not None and resize_height is not None:
            r = cls(mode="fixed", width=int(resize_width), height=int(resize_height), fmt=fmt)
        else:
            return cls(mode="none", fmt=fmt)
        logger.info(
            "ImageResizer mode=%s width=%s height=%s longest=%s max_pixels=%s "
            "min_pixels=%s factor=%s",
            r.mode,
            r.width,
            r.height,
            r.max_long_side,
            r.max_pixels,
            r.min_pixels,
            r.factor,
        )
        return r


__all__ = [
    "ImageResizer",
    "b64_to_pil",
    "pil_to_b64",
    "resize_max_pixels",
    "smart_resize",
    "sniff_format",
    "to_data_url",
    "transcode_b64",
]

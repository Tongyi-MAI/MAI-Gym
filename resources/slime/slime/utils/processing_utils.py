# Modified by MAI-Gym contributors.
import base64
import io
import json
import logging
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

logger = logging.getLogger(__name__)

DEFAULT_PATCH_SIZE = 14


def load_tokenizer(name_or_path: str, **kwargs):
    return AutoTokenizer.from_pretrained(name_or_path, **kwargs)


def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:

    modality_forced = {"return_tensors": "pt"}

    result = dict(multimodal_inputs) if multimodal_inputs else {}

    result["text_kwargs"] = {
        **result.get("text_kwargs", {}),
        "return_tensors": None,
        "return_mm_token_type_ids": False,
    }
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            result[key] = {**result[key], **modality_forced}
        else:
            result[key] = modality_forced.copy()

    return result


def _try_load_glm4v_processor(name_or_path: str, **kwargs):
    try:
        from transformers.models.glm4v.image_processing_glm4v import Glm4vImageProcessor
        from transformers.models.glm4v.processing_glm4v import Glm4vProcessor
        from transformers.models.glm4v.video_processing_glm4v import Glm4vVideoProcessor
    except ImportError:
        return None

    pp_path = Path(name_or_path) / "preprocessor_config.json"
    vp_path = Path(name_or_path) / "video_preprocessor_config.json"
    if not pp_path.exists():
        return None

    skip_keys = {"image_processor_type", "processor_class", "video_processor_type"}
    with open(pp_path) as f:
        pp_cfg = {k: v for k, v in json.load(f).items() if k not in skip_keys}
    image_processor = Glm4vImageProcessor(**pp_cfg)

    video_processor = None
    if vp_path.exists():
        with open(vp_path) as f:
            vp_cfg = {k: v for k, v in json.load(f).items() if k not in skip_keys}
        video_processor = Glm4vVideoProcessor(**vp_cfg)

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    proc = Glm4vProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )
    logger.info(f"Loaded Glm4vProcessor manually for {name_or_path}")
    return proc


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = _try_load_glm4v_processor(name_or_path, **kwargs)

    return proc


def _extract_images_from_messages(messages):
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") not in ("image", "image_url"):
                continue
            image_data = item.get(item["type"])
            if isinstance(image_data, dict):
                image_data = image_data.get("url")
            if image_data is None:
                continue
            if isinstance(image_data, Image.Image):
                images.append(image_data)
            elif isinstance(image_data, str):
                if image_data.startswith("data:"):
                    _, encoded = image_data.split(",", 1)
                    images.append(Image.open(io.BytesIO(base64.b64decode(encoded))))
                else:
                    try:
                        raw = base64.b64decode(image_data)
                        images.append(Image.open(io.BytesIO(raw)))
                    except Exception:
                        images.append(Image.open(image_data))
    return images


def process_vision_info(prompt, processor):
    normalized_prompt = []
    for message in prompt:
        content = message.get("content")
        if isinstance(content, list):
            content = [
                {**item, "image_url": item["image_url"]["url"]}
                if isinstance(item, dict)
                and item.get("type") == "image_url"
                and isinstance(item.get("image_url"), dict)
                and "url" in item["image_url"]
                else item
                for item in content
            ]
            message = {**message, "content": content}
        normalized_prompt.append(message)
    try:
        from qwen_vl_utils import process_vision_info as qwen_process_vision_info

        if hasattr(processor.image_processor, "patch_size"):
            image_patch_size = processor.image_processor.patch_size
        else:
            image_patch_size = DEFAULT_PATCH_SIZE
        images, videos = qwen_process_vision_info(normalized_prompt, image_patch_size=image_patch_size)
    except Exception:
        images = _extract_images_from_messages(normalized_prompt) or None
        videos = None

    return {"images": images, "videos": videos}


def encode_image_for_rollout_engine(image) -> str:
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"

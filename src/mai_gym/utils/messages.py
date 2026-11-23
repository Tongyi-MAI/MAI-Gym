import copy
from typing import Any


def elide_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:

    out = copy.deepcopy(messages)
    for msg in out:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                url = block.get("image", "")
                block["image"] = f"<elided len={len(url)}>"
            elif block.get("type") == "image_url":
                raw = block.get("image_url")
                url = raw["url"] if isinstance(raw, dict) else (raw or "")
                block["image_url"] = f"<elided len={len(url)}>"
    return out


def hide_history_images(
    messages: list[dict[str, Any]],
    retention: int,
) -> list[dict[str, Any]]:

    out = copy.deepcopy(messages)
    image_indices: list[int] = []
    for i in range(len(out) - 1, -1, -1):
        msg = out[i]
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if (
            isinstance(content, list)
            and content
            and isinstance(content[0], dict)
            and content[0].get("type") == "image_url"
        ):
            image_indices.append(i)
    for idx in sorted(image_indices[retention:], reverse=True):
        del out[idx]
    return out

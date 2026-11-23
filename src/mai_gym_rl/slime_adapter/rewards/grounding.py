from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading

import numpy as np

logger = logging.getLogger(__name__)

_print_lock = threading.Lock()


def _truthy_env(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "y", "on")


def _emit(lines: list[str]) -> None:

    if not lines:
        return
    msg = "\n".join(lines)
    with _print_lock:
        print(msg, flush=True)


def _summarise_response(response: str, max_chars: int) -> tuple[int, str]:
    if not isinstance(response, str):
        return 0, f"<non-str type={type(response).__name__}>"
    n = len(response)
    if n <= max_chars:
        tail = response
    else:
        tail = "..." + response[-max_chars:]

    return n, tail.replace("\n", "\\n")


_ACCEPTED_TOOL_NAMES: frozenset[str] = frozenset({"grounding", "computer_use"})


def parse_coordinates(raw_string: str) -> tuple[float, float, str]:

    if not isinstance(raw_string, str):
        return -1, -1, "not_str"

    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    start = raw_string.find(start_tag)
    end = raw_string.find(end_tag)
    if start == -1 or end == -1 or end < start:
        return -1, -1, "no_tool_call_tags"

    json_str = raw_string[start + len(start_tag) : end].strip()
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        return -1, -1, f"tool_json_decode_err: {e}"

    if not isinstance(obj, dict):
        return -1, -1, "tool_obj_not_dict"
    tool_name = obj.get("name")
    if tool_name not in _ACCEPTED_TOOL_NAMES:
        return -1, -1, f"tool_name={tool_name!r}"

    arguments = obj.get("arguments")
    if not isinstance(arguments, dict):
        return -1, -1, "arguments_not_dict"

    coordinate = arguments.get("coordinate")
    if not isinstance(coordinate, list) or len(coordinate) != 2:
        return (
            -1,
            -1,
            f"coordinate_shape={type(coordinate).__name__}/len={len(coordinate) if isinstance(coordinate, list) else 'n/a'}",
        )
    if not all(isinstance(v, (int, float)) for v in coordinate):
        return -1, -1, "coordinate_non_numeric"

    return coordinate[0], coordinate[1], "ok"


def parse_coordinates_computer_use(raw_string: str) -> tuple[float, float, str]:

    return parse_coordinates(raw_string)


_EOS_SENTINELS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>")


def _strip_eos(text: str) -> str:

    if not isinstance(text, str):
        return text
    prev = None
    while text != prev:
        prev = text
        text = text.strip()
        for s in _EOS_SENTINELS:
            if text.endswith(s):
                text = text[: -len(s)]
    return text


_FORMAT_PATTERN = re.compile(
    r"""
    ^
    \s*
    <think>
    (?P<think_content>.*?)
    </think>
    \s*
    <tool_call>
    (?P<tool_json>.*?)
    </tool_call>
    \s*
    $
    """,
    re.DOTALL | re.VERBOSE,
)


def format_reward(text: str) -> tuple[bool, str]:

    if not isinstance(text, str):
        return False, "not_str"

    match = _FORMAT_PATTERN.fullmatch(_strip_eos(text))
    if match is None:
        return False, "regex_no_match"

    tool_json_str = match.group("tool_json").strip()
    try:
        obj = json.loads(tool_json_str)
    except Exception as e:
        return False, f"tool_json_decode_err: {e}"

    if not isinstance(obj, dict):
        return False, "tool_obj_not_dict"
    if "name" not in obj or "arguments" not in obj:
        return False, "tool_obj_missing_name_or_arguments"
    if not isinstance(obj["arguments"], dict):
        return False, "arguments_not_dict"
    return True, "ok"


def _normalise_bboxes(gt) -> list[tuple[float, float, float, float]]:

    boxes = gt[0:4]
    if not isinstance(boxes[0], list):
        boxes = [boxes]
    return [tuple(b) for b in boxes]


def _isin(x: float, y: float, gt) -> bool:
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in _normalise_bboxes(gt))


def _bbox_distance(point, bbox) -> float:
    px, py = point
    x_min, y_min, x_max, y_max = bbox
    dx = max(0.0, x_min - px, px - x_max)
    dy = max(0.0, y_min - py, py - y_max)
    return float(np.sqrt(dx * dx + dy * dy))


def _closest_bbox(point, bboxes):

    best = None
    best_d = float("inf")
    best_dxy = (0.0, 0.0)
    px, py = point
    for box in bboxes:
        x_min, y_min, x_max, y_max = box
        dx = max(0.0, x_min - px, px - x_max)
        dy = max(0.0, y_min - py, py - y_max)
        d = float(np.sqrt(dx * dx + dy * dy))
        if d < best_d:
            best = box
            best_d = d
            best_dxy = (dx, dy)
    return best, best_d, best_dxy


def _should_log() -> bool:
    return _truthy_env("GROUNDING_REWARD_LOG", default="0")


def _debug_file_append(line: str) -> None:
    if os.getenv("DEBUG_MODE") != "true":
        return
    path = os.getenv("GROUNDING_REWARD_DEBUG_PATH", "/tmp/grounding_reward_debug.txt")
    try:
        with open(path, "a") as f:
            f.write(line)
    except OSError as e:
        logger.debug("could not write reward debug file %s: %s", path, e)


def click_reward(
    data_source,
    solution_str: str,
    ground_truth,
    extra_info=None,
    *,
    log_block: list[str] | None = None,
) -> float:
    reward = 0.0
    format_r = 0.0

    full_text = "<think>" + solution_str

    fmt_ok, fmt_reason = format_reward(full_text)
    if fmt_ok:
        format_r = 1.0

    pred_x, pred_y, parse_reason = parse_coordinates(full_text)
    click_r = 0.0
    in_box = False
    dist_info = ""
    try:
        if pred_x != -1 and pred_y != -1:
            in_box = _isin(pred_x, pred_y, ground_truth)
            if in_box:
                reward = 1.0
                click_r = 1.0

            bboxes = _normalise_bboxes(ground_truth)
            _, d, (dx, dy) = _closest_bbox((pred_x, pred_y), bboxes)
            dist_info = f"closest_bbox_dist={d:.3f} (dx={dx:.3f}, dy={dy:.3f})"
    except Exception as e:
        logger.exception("click_reward geometry failed: %s", e)

    if log_block is not None:
        bboxes_repr = _normalise_bboxes(ground_truth)
        log_block.append(
            f"  parse: format_ok={fmt_ok} ({fmt_reason})  coord_parse={parse_reason} "
            f"coord=({pred_x}, {pred_y})"
        )
        log_block.append(f"  bboxes={bboxes_repr}")
        log_block.append(f"  click in_box={in_box}  {dist_info}")
        log_block.append(
            f"  verdict[click]: format_r={format_r}  click_r={click_r}  "
            f"→ reward={reward * format_r}"
        )

    _debug_file_append(
        f"{datetime.datetime.now().isoformat()}\tsolution_str: {full_text!r}"
        f"\tground_truth: {ground_truth!r}\treward: {reward}\n"
    )
    return reward * format_r


def click_reward_computer_use(
    data_source,
    solution_str: str,
    ground_truth,
    extra_info=None,
    *,
    log_block: list[str] | None = None,
) -> float:

    return click_reward(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        log_block=log_block,
    )


def dense_reward(
    data_source,
    solution_str: str,
    ground_truth,
    extra_info=None,
    *,
    log_block: list[str] | None = None,
) -> float:

    reward = 0.0
    parse_reason = ""
    pred_x = pred_y = -1
    scaled_boxes: list[tuple[int, int, int, int]] = []
    distance = float("inf")
    alpha = 0.0
    img_h = img_w = 0
    try:
        raw_boxes = ground_truth[0:4]
        img_h = ground_truth[4]
        img_w = ground_truth[5]
        if not isinstance(raw_boxes[0], list):
            raw_boxes = [raw_boxes]

        for box in raw_boxes:
            x0, y0, x1, y1 = box
            scaled_boxes.append(
                (int(x0 * img_w), int(y0 * img_h), int(x1 * img_w), int(y1 * img_h))
            )

        pred_x, pred_y, parse_reason = parse_coordinates("<think>" + solution_str)
        if pred_x == -1 and pred_y == -1:
            raise ValueError(f"coordinate parse failed: {parse_reason}")
        predicted_point = (pred_x, pred_y)

        _, distance, _ = _closest_bbox(predicted_point, scaled_boxes)

        if distance == 0:
            reward = 2.0
        else:
            corners = [(0, 0), (img_w, 0), (0, img_h), (img_w, img_h)]
            d_max = max(_closest_bbox(c, scaled_boxes)[1] for c in corners) if corners else 1.0
            alpha = 3.0 / d_max if d_max > 0 else 0.01
            reward = float(np.exp(-alpha * distance))
    except Exception as e:
        logger.exception("dense_reward failed: %s", e)
        reward = 0.0

    if log_block is not None:
        log_block.append(f"  parse: coord_parse={parse_reason or 'n/a'} coord=({pred_x}, {pred_y})")
        log_block.append(f"  scaled_bboxes={scaled_boxes}  img_size=(h={img_h}, w={img_w})")
        log_block.append(f"  closest_distance={distance:.3f}  alpha={alpha:.5g}")
        log_block.append(f"  verdict[dense]: reward={reward}")

    return reward


def _ground_truth_from_label(label):

    if isinstance(label, dict):
        gt = label.get("ground_truth")
    else:
        gt = label
    if gt is None:
        raise ValueError("sample.label has no ground_truth")
    return gt


_REWARD_FN_BY_MODE = {
    "click": click_reward,
    "dense": dense_reward,
    "computer_use": click_reward_computer_use,
}


async def custom_rm(args, sample, **_kwargs):

    mode = os.getenv("GROUNDING_REWARD_MODE", "click").lower()
    fn = _REWARD_FN_BY_MODE.get(mode, click_reward)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    data_source = metadata.get("data_source")
    extra_info = metadata.get("extra_info")
    label = sample.label

    do_log = _should_log()
    log_block: list[str] | None = None
    if do_log:
        max_chars = int(os.getenv("GROUNDING_REWARD_LOG_RESPONSE_CHARS", "240") or 240)
        n_chars, tail = _summarise_response(sample.response or "", max_chars)
        idx = getattr(sample, "index", None)
        gid = getattr(sample, "group_index", None)
        sid = getattr(sample, "session_id", None)
        status = getattr(getattr(sample, "status", None), "name", None)
        label_style = label.get("style") if isinstance(label, dict) else None
        log_block = [
            f"[grounding-reward] === sample idx={idx} gid={gid} sid={sid} "
            f"status={status} mode={mode} data_source={data_source!r} ===",
            f"  response_chars={n_chars}  tail='{tail}'",
            f"  label.style={label_style!r}",
        ]

    try:
        ground_truth = _ground_truth_from_label(label)
    except Exception as e:
        if log_block is not None:
            log_block.append(f"  ERROR: cannot read ground_truth from label: {e}")
            log_block.append(f"  label_repr={label!r}")
            log_block.append("  verdict: reward=0.0 (label error)")
            _emit(log_block)
        logger.exception("custom_rm label error: %s", e)
        return 0.0

    if log_block is not None:
        log_block.append(f"  label.ground_truth={ground_truth!r}")

    try:
        reward = fn(
            data_source,
            sample.response or "",
            ground_truth,
            extra_info,
            log_block=log_block,
        )
    except Exception as e:
        if log_block is not None:
            log_block.append(f"  ERROR in {mode}_reward: {e}")
            log_block.append("  verdict: reward=0.0 (exception)")
            _emit(log_block)
        logger.exception("custom_rm %s_reward error: %s", mode, e)
        return 0.0

    if log_block is not None:
        _emit(log_block)
    return reward

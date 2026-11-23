from __future__ import annotations

import base64
import json
import logging
import time
from io import BytesIO
from typing import Any

from PIL import Image

from mai_gym.agents.slime_policy import StepRecord
from mai_gym.io import oss_io

from ..config import OSSCreds

logger = logging.getLogger(__name__)


def _write_with_retry(
    path: str,
    content: str,
    *,
    fs: Any = None,
    max_retries: int = 3,
) -> None:

    for attempt in range(max_retries):
        try:
            oss_io.sync_write_text(path, content, fs=fs)
            return
        except Exception as exc:
            if attempt < max_retries - 1:
                logger.warning(
                    "Write failed (attempt %d/%d) %s: %s",
                    attempt + 1,
                    max_retries,
                    path,
                    exc,
                )
                time.sleep(1)
            else:
                logger.warning(
                    "Write failed after %d retries, skipping: %s",
                    max_retries,
                    path,
                )


def save_trajectory(
    step_records: list[StepRecord],
    trajectory_reward: float,
    task_id: str,
    instruction: str,
    output_dir: str,
    system_prompt: str = "",
    oss_creds: OSSCreds | None = None,
) -> None:

    if not step_records:
        return

    fs = None
    if oss_io.is_oss_path(output_dir):
        if oss_creds is None:
            logger.warning(
                "save_trajectory: %r is an OSS path but no oss_creds were supplied; skipping save",
                output_dir,
            )
            return
        fs = oss_io.get_sync_fs(oss_creds)

    try:
        oss_io.sync_makedirs(output_dir, fs=fs)

        for idx, record in enumerate(step_records):
            if not record.observation_screenshot:
                continue
            try:
                img_bytes = base64.b64decode(record.observation_screenshot)

                img = Image.open(BytesIO(img_bytes))
                buf = BytesIO()
                img.save(buf, format="PNG")
                oss_io.sync_write_bytes(
                    oss_io.join(output_dir, f"step_{idx:04d}.png"),
                    buf.getvalue(),
                    fs=fs,
                )
            except Exception as exc:
                logger.debug("Failed to save screenshot step %d: %s", idx, exc)

        steps_data: list[dict[str, Any]] = []
        for idx, record in enumerate(step_records):
            steps_data.append(
                {
                    "step": idx,
                    "screenshot": f"step_{idx:04d}.png",
                    "response_text": record.response_text,
                    "action_dict": record.action_dict,
                    "format_reward": record.format_reward,
                    "is_terminal": record.is_terminal,
                    "finish_reason": record.finish_reason,
                }
            )

        traj_data = {
            "task_id": task_id,
            "instruction": instruction,
            "system_prompt": system_prompt,
            "n_steps": len(step_records),
            "trajectory_reward": trajectory_reward,
            "steps": steps_data,
        }

        traj_json_str = json.dumps(traj_data, indent=2, ensure_ascii=False)
        score_str = f"{trajectory_reward}\nn_steps: {len(step_records)}\n"

        _write_with_retry(oss_io.join(output_dir, "traj.json"), traj_json_str, fs=fs)
        _write_with_retry(oss_io.join(output_dir, "score.txt"), score_str, fs=fs)

        logger.info(
            "Trajectory saved: task=%s, reward=%.3f, steps=%d, dir=%s",
            task_id,
            trajectory_reward,
            len(step_records),
            output_dir,
        )

    except Exception:
        logger.exception("Failed to save trajectory for task %s", task_id)

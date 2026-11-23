from __future__ import annotations

import asyncio
import json
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from mai_gym.models.config import OSSConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class RewardEvaluator(Protocol):
    async def evaluate(
        self,
        session_id: str,
        task_goal: str,
        *,
        env_reward: float | None = None,
        traj_dir: str | Path | None = None,
    ) -> float: ...


class DeterministicEvaluator:
    async def evaluate(
        self,
        session_id: str,
        task_goal: str,
        *,
        env_reward: float | None = None,
        traj_dir: str | Path | None = None,
    ) -> float:
        return env_reward if env_reward is not None else 0.0


class LLMJudgeEvaluator:
    def __init__(
        self,
        client: Any,
        model: str,
        image_scale: float = 0.5,
        oss_cfg: OSSConfig | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._image_scale = image_scale
        self._oss_cfg = oss_cfg

    async def evaluate(
        self,
        session_id: str,
        task_goal: str,
        *,
        env_reward: float | None = None,
        traj_dir: str | Path | None = None,
    ) -> float:
        if traj_dir is None:
            logger.warning(
                "[LLMJudgeEvaluator] No traj_dir provided for session %s; returning 0.0",
                session_id,
            )
            return 0.0

        from mai_gym.io import oss_io

        if oss_io.is_oss_path(str(traj_dir)):
            if self._oss_cfg is None:
                logger.warning(
                    "[LLMJudgeEvaluator] OSS traj_dir requires manager OSS "
                    "credentials for session %s; returning 0.0",
                    session_id,
                )
                return 0.0
            try:
                return await self._run_judge_oss(task_goal, str(traj_dir))
            except Exception as exc:
                logger.warning(
                    "[LLMJudgeEvaluator] OSS evaluation failed for session %s: %s",
                    session_id,
                    exc,
                )
                return 0.0

        try:
            return await self._run_judge(task_goal, Path(traj_dir))
        except Exception as exc:
            logger.warning(
                "[LLMJudgeEvaluator] Evaluation failed for session %s: %s",
                session_id,
                exc,
            )
            return 0.0

    async def _run_judge(self, task_goal: str, traj_dir: Path) -> float:

        traj_file = traj_dir / "traj.json"
        shots_dir = traj_dir / "screenshots"
        if not traj_file.exists() or not shots_dir.exists():
            logger.warning("Trajectory data not found in %s", traj_dir)
            return 0.0

        shot_paths = self._sorted_screenshot_paths(shots_dir.glob("*.png"))
        screenshots = [(p.name, p.read_bytes()) for p in shot_paths]
        verdict, raw_text = await self._judge_artifacts(
            task_goal,
            traj_file.read_bytes(),
            screenshots,
        )

        try:
            out = traj_dir / "judge_result.txt"
            out.write_text(f"verdict: {verdict:g}\n\n{raw_text}\n", encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write judge_result.txt: %s", exc)

        return verdict

    async def _run_judge_oss(self, task_goal: str, traj_dir: str) -> float:

        from mai_gym.io import oss_io

        assert self._oss_cfg is not None
        fs = oss_io.get_async_fs(self._oss_cfg)
        traj_path = oss_io.join(traj_dir, "traj.json")
        shots_glob = oss_io.join(traj_dir, "screenshots", "*.png")
        traj_bytes = await oss_io.read_bytes(traj_path, fs=fs)
        raw_paths = await fs._glob(shots_glob)

        def _as_oss_uri(path: str) -> str:
            return path if path.startswith("oss://") else f"oss://{path}"

        shot_paths = self._sorted_screenshot_paths([_as_oss_uri(str(path)) for path in raw_paths])
        shot_bytes = await asyncio.gather(*(oss_io.read_bytes(path, fs=fs) for path in shot_paths))
        screenshots = [
            (path.rsplit("/", 1)[-1], data)
            for path, data in zip(shot_paths, shot_bytes, strict=True)
        ]
        verdict, raw_text = await self._judge_artifacts(
            task_goal,
            traj_bytes,
            screenshots,
        )
        await oss_io.write_text(
            oss_io.join(traj_dir, "judge_result.txt"),
            f"verdict: {verdict:g}\n\n{raw_text}\n",
            fs=fs,
        )
        return verdict

    @staticmethod
    def _sorted_screenshot_paths(paths: Any) -> list[Any]:

        shot_re = re.compile(r".*-(\d+)-(\d+)\.png$")
        indexed: list[tuple[int, int, Any]] = []
        for path in paths:
            name = str(path).rsplit("/", 1)[-1]
            match = shot_re.match(name)
            if match:
                indexed.append((int(match.group(1)), int(match.group(2)), path))
        indexed.sort()
        return [path for _, _, path in indexed]

    async def _judge_artifacts(
        self,
        task_goal: str,
        traj_bytes: bytes,
        screenshots: list[tuple[str, bytes]],
    ) -> tuple[float, str]:

        import base64

        from PIL import Image

        data = json.loads(traj_bytes)
        steps: list[dict[str, Any]] = []
        for key in sorted(data.keys(), key=lambda x: int(x)):
            episode = data[key] or {}
            for s in episode.get("traj", []) or []:
                steps.append(s)

        action_lines: list[str] = []
        for i, s in enumerate(steps, start=1):
            step_idx = s.get("step", i)
            action = s.get("action")
            if action is None:
                action_str = s.get("prediction") or "{}"
                if not isinstance(action_str, str):
                    action_str = json.dumps(action_str, ensure_ascii=False)
            else:
                action_str = json.dumps(action, ensure_ascii=False)
            action_lines.append(f"Step {step_idx}: {action_str}")
        action_text = "\n".join(action_lines) if action_lines else "(no actions recorded)"

        user_text = (
            f"TASK INSTRUCTION:\n{task_goal}\n\n"
            f"AGENT ACTIONS:\n{action_text}\n\n"
            f"Screenshots follow in order (Step 1 .. Step {len(screenshots)}, "
            f"each captured BEFORE that step's action)."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for i, (name, image_bytes) in enumerate(screenshots, start=1):
            try:
                with Image.open(BytesIO(image_bytes)) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    w, h = img.size
                    new_w = max(1, int(w * self._image_scale))
                    new_h = max(1, int(h * self._image_scale))
                    resized = img.resize((new_w, new_h), Image.LANCZOS)
                buf = BytesIO()
                resized.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                content.append({"type": "text", "text": f"[Step {i} screenshot]"})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            except Exception as exc:
                logger.warning("Failed to load screenshot %s: %s", name, exc)

        from mai_gym.server._judge_prompt import JUDGE_PROMPT

        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": content},
        ]

        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0,
        )
        raw_text = resp.choices[0].message.content or ""

        ans_re = re.compile(r"<ans>\s*(\d)\s*</ans>", re.DOTALL)
        m = ans_re.search(raw_text)
        verdict = int(m.group(1)) if m else 0

        return float(verdict), raw_text

import logging
from pathlib import Path

from playwright.async_api import Page

logger = logging.getLogger(__name__)


async def run_evaluator(
    page: Page,
    *,
    task_id: str | None = None,
    evaluator_js: str | None = None,
    evaluators_dir: str | None = None,
) -> float:

    js_code = evaluator_js

    if js_code is None and task_id and evaluators_dir:
        eval_path = Path(evaluators_dir) / f"{task_id}.js"
        if eval_path.exists():
            js_code = eval_path.read_text(encoding="utf-8")
        else:
            logger.warning("[Evaluator] No evaluator file for task %s at %s", task_id, eval_path)
            return 0.0

    if js_code is None:
        logger.warning("[Evaluator] No evaluator JS provided")
        return 0.0

    try:
        wrapped = f"(() => {{\n{js_code}\n}})()"
        result = await page.evaluate(wrapped)
        score = float(result) if isinstance(result, (int, float)) else 0.0
        return max(0.0, min(1.0, score))
    except Exception as exc:
        logger.warning("[Evaluator] JS evaluation failed: %s", exc)
        return 0.0

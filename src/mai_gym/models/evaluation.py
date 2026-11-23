import math


class EvaluationScore(float):
    success: bool | None

    def __new__(cls, reward: float, success: bool | None = None) -> "EvaluationScore":
        if not math.isfinite(reward):
            raise ValueError("Verifier reward must be finite")
        if success is not None and not isinstance(success, bool):
            raise ValueError("Verifier success must be a boolean")
        obj = super().__new__(cls, reward)
        obj.success = success
        return obj


def task_succeeded(score: float) -> bool:
    verdict = getattr(score, "success", None)
    return bool(verdict) if verdict is not None else score >= 1.0

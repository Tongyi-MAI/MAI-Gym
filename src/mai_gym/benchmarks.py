from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    env_type: str
    max_steps: int | None
    metric: str = "success_rate"


BENCHMARKS = {
    "mobileworld": BenchmarkSpec("mobileworld", 100),
    "androidworld": BenchmarkSpec("androidworld", None),
    "mobilegym": BenchmarkSpec("mobilegym", 100, "partial_score"),
    "osworld-verified": BenchmarkSpec("osworld", 100),
    "osworld-2": BenchmarkSpec("osworld-2", 500, "binary_and_partial"),
    "macagentbench": BenchmarkSpec("macagentbench", 50),
    "mypcbench": BenchmarkSpec("mypcbench", 100, "perfect_task_rate"),
    "webvoyager": BenchmarkSpec("webvoyager", 15),
    "onlinemind2web": BenchmarkSpec("onlinemind2web", 50),
    "webarena": BenchmarkSpec("webarena", 30),
    "mocked-apps": BenchmarkSpec("mocked-apps", None, "binary_and_partial"),
}

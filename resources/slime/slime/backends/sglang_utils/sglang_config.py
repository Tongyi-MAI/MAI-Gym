# Modified by MAI-Gym contributors.

import dataclasses
import logging

import yaml

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ServerGroupConfig:

    worker_type: str
    num_gpus: int
    num_gpus_per_engine: int | None = None
    overrides: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        valid_types = {"regular", "prefill", "decode", "placeholder", "encoder"}
        assert (
            self.worker_type in valid_types
        ), f"Invalid worker_type '{self.worker_type}', must be one of {valid_types}"
        assert self.num_gpus > 0, f"num_gpus must be > 0, got {self.num_gpus}"


@dataclasses.dataclass
class ModelConfig:

    name: str
    model_path: str | None = None
    num_gpus_per_engine: int | None = None
    server_groups: list[ServerGroupConfig] = dataclasses.field(default_factory=list)
    update_weights: bool | None = None

    def resolve(self, args) -> None:
        default_gpus_per_engine = self.num_gpus_per_engine or args.rollout_num_gpus_per_engine
        default_model_path = self.model_path or args.hf_checkpoint
        for g in self.server_groups:
            if g.num_gpus_per_engine is None:
                g.num_gpus_per_engine = default_gpus_per_engine
            if "model_path" not in g.overrides:
                g.overrides["model_path"] = default_model_path

        if self.server_groups:
            model_paths = {g.overrides["model_path"] for g in self.server_groups}
            assert len(model_paths) == 1, (
                f"Model '{self.name}' has server groups with different model_path values: "
                f"{model_paths}. All server groups within a model must use the same model_path."
            )
            effective_model_path = model_paths.pop()
        else:
            effective_model_path = default_model_path

        if self.update_weights is None:
            if effective_model_path != args.hf_checkpoint:
                logger.warning(
                    f"Model '{self.name}' uses model_path='{effective_model_path}' which differs "
                    f"from hf_checkpoint='{args.hf_checkpoint}'. Defaulting update_weights to False. "
                    f"Set update_weights explicitly in the config to suppress this warning."
                )
                self.update_weights = False
            else:
                self.update_weights = True

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(g.worker_type in ("prefill", "decode") for g in self.server_groups)

    @property
    def has_encoder_disaggregation(self) -> bool:
        return any(g.worker_type == "encoder" for g in self.server_groups)

    @property
    def total_num_gpus(self) -> int:
        return sum(g.num_gpus for g in self.server_groups)


@dataclasses.dataclass
class SglangConfig:

    models: list[ModelConfig]

    @staticmethod
    def from_yaml(path: str) -> "SglangConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        assert "sglang" in data, (
            f"sglang config must have a 'sglang' key, got {list(data.keys())}. "
            f"Wrap your server_groups inside a model entry under 'sglang'."
        )
        models = []
        for m in data["sglang"]:
            raw_groups = m.get("server_groups") or m.get("engine_groups") or []
            groups = [ServerGroupConfig(**g) for g in raw_groups]
            models.append(
                ModelConfig(
                    name=m["name"],
                    model_path=m.get("model_path"),
                    num_gpus_per_engine=m.get("num_gpus_per_engine"),
                    server_groups=groups,
                    update_weights=m.get("update_weights"),
                )
            )
        return SglangConfig(models=models)

    @staticmethod
    def from_prefill_num_servers(args) -> "SglangConfig":
        total_gpus = args.rollout_num_gpus
        prefill_gpus = args.prefill_num_servers * args.rollout_num_gpus_per_engine
        decode_gpus = total_gpus - prefill_gpus
        assert decode_gpus > 0, f"No decode GPUs: total {total_gpus}, prefill {prefill_gpus}"
        return SglangConfig(
            models=[
                ModelConfig(
                    name="default",
                    server_groups=[
                        ServerGroupConfig(worker_type="prefill", num_gpus=prefill_gpus),
                        ServerGroupConfig(worker_type="decode", num_gpus=decode_gpus),
                    ],
                )
            ]
        )

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(m.has_pd_disaggregation for m in self.models)

    @property
    def total_num_gpus(self) -> int:
        return sum(m.total_num_gpus for m in self.models)

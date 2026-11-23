import logging
import os
import types
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, Field, model_validator

from mai_gym.models.environment import DeviceType, InfraType, identity
from mai_gym.models.launch_profiles import MobileWorldPatchConfig, MobileWorldTaskArtifactConfig

logger = logging.getLogger(__name__)


def _normalize_legacy_field(value: Any, *, canonical: str, legacy: str) -> Any:
    if not isinstance(value, dict) or legacy not in value:
        return value
    data = dict(value)
    legacy_val = data.pop(legacy)
    if canonical in data:
        if data[canonical] != legacy_val:
            raise ValueError(f"Conflicting {canonical} and legacy {legacy}")
    else:
        data[canonical] = legacy_val
    return data


class SshConnection(BaseModel):
    host: str
    user: str = "root"
    key: str | None = None


class SshConnectionWrapper(BaseModel):
    ssh: SshConnection


class AttachBackend(BaseModel):
    url: str
    device_id: str | None = None
    container_name: str | None = None
    chromium_port: int = 9222
    vlc_port: int = 8080
    vnc_port: int = 8006


class AttachConnection(BaseModel):
    backends: list[AttachBackend]


class AttachConnectionWrapper(BaseModel):
    attach: AttachConnection


class AgentProxy(BaseModel):
    port: int


class AgentConnection(BaseModel):
    host: str
    port: int = 9090
    proxy: AgentProxy | None = None


class AgentConnectionWrapper(BaseModel):
    agent: AgentConnection


class CommandLaunchConfig(BaseModel):
    command: str
    port: int = 6800


class DockerLaunchConfig(BaseModel):
    image: str
    command: list[str] = Field(default_factory=list)
    port_range_start: int = 7000
    container_port: int = 6800
    privileged: bool = False

    shm_size: str | None = None
    health_timeout: int = 60

    viewer_port_range_start: int = 7860
    vnc_port_range_start: int = 5800
    adb_port_range_start: int = 5556
    env_file: str | None = None
    name_prefix: str = "mai_gym_mw"
    ready_timeout: int = 600

    launch_interval: int = 20

    http_proxy: str | None = None

    path_to_vm: str | None = None

    reuse_existing: bool = True

    env_vars: dict[str, str] | None = None

    volumes: list[str] | None = None

    dev_src_path: str | None = None


class FleetAutoscaleConfig(BaseModel):
    enabled: bool = False
    idle_timeout_seconds: float = Field(default=600.0, ge=0.0)
    request_ttl_seconds: float = Field(default=180.0, gt=0.0)
    heartbeat_interval_seconds: float = Field(default=30.0, gt=0.0)
    reconcile_interval_seconds: float = Field(default=2.0, gt=0.0)
    warmup_timeout_seconds: float = Field(default=1200.0, gt=0.0)
    min_replicas: int = Field(default=0, ge=0)

    scale_up_batch_size: int = Field(default=0, ge=0)
    scale_up_interval_seconds: float = Field(default=15.0, gt=0.0)

    scale_up_max_pending: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_heartbeat_window(self) -> "FleetAutoscaleConfig":
        if self.heartbeat_interval_seconds >= self.request_ttl_seconds:
            raise ValueError("heartbeat_interval_seconds must be less than request_ttl_seconds")
        return self


class K8sLaunchConfig(BaseModel):
    template_id: str
    namespace: str = "default"
    cpu: int = 4
    memory: int = 8
    os_type: Literal["linux", "windows"] = "linux"

    service_port: int = 5000

    chromium_port: int = 9222
    vlc_port: int = 8080
    vnc_port: int = 8006

    kubeconfig: str | None = None

    owner_name: str | None = None

    launch_timeout: int = 600

    ready_request_timeout: float | None = Field(default=None, gt=0.0)

    max_concurrent_launches: int = Field(default=16, ge=1)

    delete_on_shutdown: bool = True

    destroy_on_release: bool = True

    acquisition: Literal["direct", "pool", "adopt"] = "direct"

    pool_size: int | None = None

    pool_name: str | None = None

    pool_reuse_existing: bool = False

    claim_timeout: int = 300

    claim_ttl: int = 900

    create_on_no_stock: bool = True

    lazy_acquire: bool = False

    ready_buffer_size: int = Field(default=0, ge=0)

    fleet_autoscale: FleetAutoscaleConfig = Field(default_factory=FleetAutoscaleConfig)

    mobileworld_patch: MobileWorldPatchConfig = Field(default_factory=MobileWorldPatchConfig)

    mobileworld_task_artifacts: MobileWorldTaskArtifactConfig = Field(
        default_factory=MobileWorldTaskArtifactConfig
    )

    @model_validator(mode="after")
    def _validate_lazy_acquire(self) -> "K8sLaunchConfig":
        if self.ready_buffer_size and not self.lazy_acquire:
            raise ValueError("ready_buffer_size requires lazy_acquire=true")
        if self.lazy_acquire and self.acquisition != "pool":
            raise ValueError("lazy_acquire requires acquisition='pool'")
        if self.lazy_acquire and not self.destroy_on_release:
            raise ValueError("lazy_acquire requires destroy_on_release=true")
        if self.fleet_autoscale.enabled and not self.lazy_acquire:
            raise ValueError("fleet_autoscale requires lazy_acquire=true")
        return self

    pod_bootstrap_enabled: bool = False
    pip_index_url: str | None = "https://pypi.org/simple"
    pip_trusted_host: str | None = None
    apt_mirror_url: str | None = "http://archive.ubuntu.com/ubuntu"
    vm_password: str = Field(default="password", repr=False)
    http_proxy: str | None = Field(default=None, repr=False)
    no_proxy: str = "localhost,127.0.0.1,::1,0.0.0.0,.svc,.cluster.local"
    pod_setup_commands: list[str] = Field(default_factory=list)
    pod_setup_timeout: float = Field(default=360.0, gt=0.0)
    pod_setup_max_attempts: int = Field(default=3, ge=1)
    pod_setup_retry_backoff: float = Field(default=3.0, ge=0.0)

    @model_validator(mode="after")
    def _resolve_pod_bootstrap_secrets(self) -> "K8sLaunchConfig":

        for field_name in ("http_proxy", "vm_password"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.startswith("env/"):
                continue
            var_name = value[4:]
            resolved = os.environ.get(var_name, "")
            if not resolved:
                logger.warning(
                    "Environment variable %r is not set (referenced by k8s launch.%s)",
                    var_name,
                    field_name,
                )
            setattr(self, field_name, resolved)
        return self


class RecoveryConfig(BaseModel):
    strategy: Literal["recover", "relaunch", "abandon"] = "relaunch"
    timeout: int = 60
    auto_relaunch: bool = True


class DeviceConfig(BaseModel):
    name: str
    type: str
    env_type: str
    device_type: DeviceType
    infra_type: InfraType
    env_config: dict[str, Any] = Field(default_factory=dict)
    connection: Annotated[
        Literal["local"] | SshConnectionWrapper | AttachConnectionWrapper | AgentConnectionWrapper,
        Field(default="local"),
    ]
    max_sessions: int = 1

    sessions_per_backend: int | None = None
    launch: Annotated[
        CommandLaunchConfig | DockerLaunchConfig | K8sLaunchConfig | None,
        Field(default=None, discriminator=None),
    ]
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)

    reset_concurrency: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_environment(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        requested = data.get("env_type") or data.get("type")
        if not requested:
            raise ValueError("device requires env_type (or legacy type)")
        env = identity(requested)
        legacy = identity(data.get("type", requested))
        if env.name != legacy.name or (
            env.profile and legacy.profile and env.profile != legacy.profile
        ):
            raise ValueError("conflicting type and env_type")
        if data.get("device_type", env.device_type) != env.device_type:
            raise ValueError(f"{env.name} requires device_type={env.device_type!r}")
        connection = data.get("connection", "local")
        launch = data.get("launch") or {}

        def has_launch(key: str) -> bool:
            return key in launch if isinstance(launch, dict) else hasattr(launch, key)

        if isinstance(connection, (AttachConnectionWrapper, dict)) and (
            isinstance(connection, AttachConnectionWrapper) or "attach" in connection
        ):
            inferred = "attach"
        elif isinstance(connection, (AgentConnectionWrapper, dict)) and (
            isinstance(connection, AgentConnectionWrapper) or "agent" in connection
        ):
            inferred = "agent"
        elif has_launch("template_id"):
            inferred = "k8s"
        elif has_launch("image"):
            inferred = "docker"
        elif connection == "local":
            inferred = "local"
        else:
            inferred = "ssh"
        infra = data.get("infra_type", inferred)
        if infra != inferred:
            raise ValueError(f"infra_type={infra!r} conflicts with connection/launch ({inferred})")
        for alias in (env, legacy):
            if alias.infra_type and alias.infra_type != infra:
                raise ValueError(f"{requested!r} requires infra_type={alias.infra_type!r}")
        data.update(
            type=data.get("type", requested),
            env_type=env.name,
            device_type=env.device_type,
            infra_type=infra,
            connection=connection,
        )
        config = dict(data.get("env_config") or {})
        if env.profile or legacy.profile:
            profile = env.profile or legacy.profile
            if config.get("profile", profile) != profile:
                raise ValueError("env_config.profile conflicts with environment alias")
            config["profile"] = profile
        if isinstance(launch, dict):
            launch = dict(launch)
            for old, new in (
                ("mobileworld_patch", "patch"),
                ("mobileworld_task_artifacts", "task_artifacts"),
            ):
                if old in launch:
                    if new in config and config[new] != launch[old]:
                        raise ValueError(f"conflicting launch.{old} and env_config.{new}")
                    config[new] = launch[old]
                elif new in config:
                    launch[old] = config[new]
            data["launch"] = launch or None
        data["env_config"] = config
        return data

    def accepts_env_type(self, requested: str) -> bool:
        env = identity(requested)
        if env.name != self.env_type:
            return False
        if env.infra_type and env.infra_type != self.infra_type:
            return False
        return env.profile == self.env_config.get("profile")

    @model_validator(mode="after")
    def _validate_launch_required(self) -> "DeviceConfig":

        if isinstance(self.launch, K8sLaunchConfig):
            for key, field, schema in (
                ("patch", "mobileworld_patch", MobileWorldPatchConfig),
                ("task_artifacts", "mobileworld_task_artifacts", MobileWorldTaskArtifactConfig),
            ):
                if key not in self.env_config:
                    continue
                if self.env_type != "mobileworld":
                    raise ValueError(f"env_config.{key} belongs to MobileWorld")
                configured = schema.model_validate(self.env_config[key])
                if (
                    field in self.launch.model_fields_set
                    and getattr(self.launch, field) != configured
                ):
                    raise ValueError(f"Conflicting launch.{field} and env_config.{key}")
                setattr(self.launch, field, configured)
            if not self.is_local:
                raise ValueError(
                    "Kubernetes routing uses the cluster configuration, not connection.ssh/agent"
                )

        if self.attach is None and self.launch is None:
            raise ValueError(
                f"Device {self.name!r}: 'launch' is required unless connection.attach is set."
            )
        if (
            isinstance(self.launch, K8sLaunchConfig)
            and self.launch.fleet_autoscale.enabled
            and self.effective_sessions_per_backend != 1
        ):
            raise ValueError(
                f"Device {self.name!r}: fleet_autoscale requires sessions_per_backend=1"
            )
        if (
            isinstance(self.launch, K8sLaunchConfig)
            and self.launch.fleet_autoscale.enabled
            and self.launch.fleet_autoscale.min_replicas > self.max_sessions
        ):
            raise ValueError(
                f"Device {self.name!r}: fleet_autoscale min_replicas cannot exceed max_sessions"
            )
        if (
            isinstance(self.launch, K8sLaunchConfig)
            and self.launch.mobileworld_patch.enabled
            and self.env_type != "mobileworld"
        ):
            raise ValueError(
                f"Device {self.name!r}: mobileworld_patch requires env_type='mobileworld'"
            )
        return self

    @property
    def effective_sessions_per_backend(self) -> int:
        return self.sessions_per_backend or self.max_sessions

    @property
    def is_local(self) -> bool:
        return self.connection == "local"

    @property
    def ssh(self) -> SshConnection | None:
        if isinstance(self.connection, SshConnectionWrapper):
            return self.connection.ssh
        return None

    @property
    def attach(self) -> AttachConnection | None:
        if isinstance(self.connection, AttachConnectionWrapper):
            return self.connection.attach
        return None

    @property
    def agent(self) -> AgentConnection | None:
        if isinstance(self.connection, AgentConnectionWrapper):
            return self.connection.agent
        return None


class OSSConfig(BaseModel):
    endpoint: str
    access_key_id: str
    access_key_secret: str
    region: str | None = None

    connection_pool_size: int = Field(default=20, ge=1)

    bucket: str | None = None

    @model_validator(mode="after")
    def _resolve_env_vars(self) -> "OSSConfig":
        for field in ("endpoint", "access_key_id", "access_key_secret", "region", "bucket"):
            val = getattr(self, field, None)
            if isinstance(val, str) and val.startswith("env/"):
                var_name = val[4:]
                resolved = os.environ.get(var_name, "")
                if not resolved:
                    logger.warning(
                        "Environment variable %r is not set (referenced by oss.%s)",
                        var_name,
                        field,
                    )
                setattr(self, field, resolved)
        return self


class ScreenshotConfig(BaseModel):
    codec: Literal["webp", "jpeg", "png"] = "png"
    quality: int = 85
    transport: Literal["inline", "oss"] = "inline"

    oss_prefix: str | None = None


class OSWorldProxyConfig(BaseModel):
    enabled: bool = False
    host: str = ""
    port: int = 0
    username: str = ""
    password: str = ""
    protocol: str = "http"

    client_password: str = "password"


class VSCodeExtMirrorConfig(BaseModel):
    base_url: str | None = None
    extensions: list[str] = Field(default_factory=list)


class OSWorldEvalIsolationConfig(BaseModel):
    enabled: bool = True
    timeout_seconds: float = Field(default=300.0, gt=0)
    queue_timeout_seconds: float = Field(default=300.0, gt=0)
    max_concurrency: int = Field(default=32, ge=1)
    worker_cpu_threads: int = Field(default=1, ge=1)
    max_ocr_concurrency: int = Field(default=2, ge=1)
    terminate_grace_seconds: float = Field(default=3.0, ge=0.01)


class ResponseCompressionConfig(BaseModel):
    enabled: bool = True
    workers: int = Field(default=4, ge=1)
    level: int = Field(default=1, ge=1, le=9)
    minimum_size: int = Field(default=1024, ge=0)


class ManagerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8090
    response_compression: ResponseCompressionConfig = Field(
        default_factory=ResponseCompressionConfig,
    )
    post_action_wait: float = 1.5
    screenshot: ScreenshotConfig = ScreenshotConfig()
    traj_log_dir: str | None = None

    oss: OSSConfig | None = None

    auto_prewarm: bool = False

    osworld_examples_dir: str | None = None

    cua_osworld_tasks_dir: str | None = None

    osworld_proxy: OSWorldProxyConfig = Field(default_factory=OSWorldProxyConfig)

    vscode_ext_mirror: VSCodeExtMirrorConfig = Field(default_factory=VSCodeExtMirrorConfig)

    osworld_eval_isolation: OSWorldEvalIsolationConfig = Field(
        default_factory=OSWorldEvalIsolationConfig
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, value: Any) -> Any:
        return _normalize_legacy_field(value, canonical="traj_log_dir", legacy="debug_log_dir")

    @property
    def debug_log_dir(self) -> str | None:
        return self.traj_log_dir

    @debug_log_dir.setter
    def debug_log_dir(self, value: str | None) -> None:
        self.traj_log_dir = value

    @model_validator(mode="after")
    def _resolve_screenshot_oss_prefix(self) -> "ManagerConfig":

        if self.screenshot.transport != "oss":
            return self
        if self.oss is None:
            raise ValueError(
                "manager.screenshot.transport='oss' requires a manager.oss block "
                "(credentials) to upload screenshots."
            )
        if not self.screenshot.oss_prefix:
            if not self.oss.bucket:
                raise ValueError(
                    "manager.screenshot.transport='oss' requires either "
                    "manager.screenshot.oss_prefix (e.g. 'oss://bucket/screenshots') "
                    "or manager.oss.bucket to derive it from."
                )
            self.screenshot.oss_prefix = f"oss://{self.oss.bucket}/screenshots"
        return self


class JudgeConfig(BaseModel):
    base_url: str
    api_key: str = Field(default="EMPTY", repr=False)
    model: str
    image_scale: float = 0.5

    scope: Literal["all", "flagged"] = "all"

    mode: Literal["replace", "alongside"] = "replace"

    @model_validator(mode="after")
    def _resolve_env_vars(self) -> "JudgeConfig":

        if self.mode == "alongside" and self.scope != "all":
            raise ValueError(
                "judge.mode='alongside' requires judge.scope='all' because "
                "flagged tasks may not have a built-in verifier"
            )
        if self.api_key.startswith("env/"):
            var_name = self.api_key[4:]
            resolved = os.environ.get(var_name, "")
            if not resolved:
                logger.warning(
                    "Environment variable %r is not set (referenced by judge.api_key)",
                    var_name,
                )
            self.api_key = resolved
        return self


class AgentConfig(BaseModel):
    type: str
    model_name: str
    base_url: str
    api_key: str = Field(default="EMPTY", repr=False)
    max_steps: int = 50
    runtime_conf: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, value: Any) -> Any:
        return _normalize_legacy_field(value, canonical="base_url", legacy="llm_base_url")

    @property
    def llm_base_url(self) -> str:
        return self.base_url

    @llm_base_url.setter
    def llm_base_url(self, value: str) -> None:
        self.base_url = value

    @model_validator(mode="after")
    def _resolve_env_vars(self) -> "AgentConfig":

        for field in ("api_key", "base_url"):
            val = getattr(self, field)
            if isinstance(val, str) and val.startswith("env/"):
                var_name = val[4:]
                resolved = os.environ.get(var_name, "")
                if not resolved:
                    logger.warning(
                        "Environment variable %r is not set (referenced by agent.%s)",
                        var_name,
                        field,
                    )
                setattr(self, field, resolved)
        return self


class BenchmarkConfig(BaseModel):
    suite: str | None = None
    model: str | None = None
    run_id: str | None = None


class RolloutConfig(BaseModel):
    num_envs: int = 1
    env_type: str = "web_env"

    fleet_request_mode: Literal["auto", "off", "required"] = "auto"

    screenshot_transport: Literal["inline", "oss"] = "inline"

    evaluation_mode: Literal["built_in", "llm_judge", "both"] = "built_in"
    tasks_dir: str | None = None
    domains: list[str] | None = None
    tasks: str | None = None
    no_resume: bool = False

    manager_url: str | None = None

    train_id: str | None = None

    pass_n: int = 1

    retry_until_scored: bool = False

    max_retry_rounds: int = 5

    use_fixed_verifier: bool | None = None

    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    resolve_task_metadata: bool = False
    require_task_steps: bool = False
    require_score: bool = False
    is_eval: bool = False
    task_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, value: Any) -> Any:
        return _normalize_legacy_field(value, canonical="manager_url", legacy="controller_url")

    @property
    def controller_url(self) -> str | None:
        return self.manager_url

    @controller_url.setter
    def controller_url(self, value: str | None) -> None:
        self.manager_url = value


class MobileGymConfig(BaseModel):
    devices: list[DeviceConfig] = Field(default_factory=list)
    manager: ManagerConfig = Field(default_factory=ManagerConfig)
    judge: JudgeConfig | None = None
    agent: AgentConfig | None = None
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)

    source_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MobileGymConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        obj = cls.model_validate(data)
        obj.source_path = str(path)
        return obj


def _coerce_override_value(raw: str, annotation: Any) -> Any:

    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        if raw.strip().lower() in ("null", "none"):
            return None
        members = [a for a in get_args(annotation) if a is not type(None)]
        annotation = members[0] if members else str
    if annotation is bool:
        low = raw.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"cannot parse bool from {raw!r}")
    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)
    if annotation is str:
        return raw

    return yaml.safe_load(raw)


_LEGACY_CLI_OVERRIDE_FIELDS: dict[tuple[type[BaseModel], str], str] = {
    (RolloutConfig, "controller_url"): "manager_url",
    (AgentConfig, "llm_base_url"): "base_url",
    (ManagerConfig, "debug_log_dir"): "traj_log_dir",
}


def apply_cli_overrides(cfg: MobileGymConfig, overrides: Iterable[str]) -> MobileGymConfig:

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        path = [p for p in key.strip().split(".") if p]
        if not path:
            raise ValueError(f"--set has an empty key in {item!r}")

        target: Any = cfg
        for seg in path[:-1]:
            if isinstance(target, BaseModel):
                seg = _LEGACY_CLI_OVERRIDE_FIELDS.get((type(target), seg), seg)
                if seg not in type(target).model_fields:
                    raise ValueError(f"--set {key!r}: {type(target).__name__} has no field {seg!r}")
                target = getattr(target, seg)
            elif isinstance(target, dict):
                target = target.setdefault(seg, {})
            else:
                raise ValueError(f"--set {key!r}: cannot descend into {seg!r}")
            if target is None:
                raise ValueError(
                    f"--set {key!r}: intermediate {seg!r} is None — cannot set a nested key"
                )

        leaf = path[-1]
        if isinstance(target, BaseModel):
            leaf = _LEGACY_CLI_OVERRIDE_FIELDS.get((type(target), leaf), leaf)
            field = type(target).model_fields.get(leaf)
            if field is None:
                raise ValueError(f"--set {key!r}: {type(target).__name__} has no field {leaf!r}")
            try:
                value = _coerce_override_value(raw, field.annotation)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"--set {key!r}: {exc}") from exc
            setattr(target, leaf, value)
        elif isinstance(target, dict):
            target[leaf] = yaml.safe_load(raw)
        else:
            raise ValueError(f"--set {key!r}: target is not a model or dict")
    return cfg

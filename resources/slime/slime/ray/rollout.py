# Modified by MAI-Gym contributors.
import dataclasses
import itertools
import logging
import multiprocessing
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from slime.rollout.base_types import call_rollout_fn
from slime.utils import logging_utils
from slime.utils.health_monitor import RolloutHealthMonitor
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info, init_http_client
from slime.utils.logging_utils import configure_logger, init_tracking
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix
from slime.utils.misc import Box, group_by, load_function
from slime.utils.seqlen_balancing import get_seqlen_balanced_partitions
from slime.utils.types import Sample

from ..utils.metric_utils import has_repetition
from .utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, Lock

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

OFFLOAD_PROBE_TIMEOUT_S = 60.0


@dataclasses.dataclass
class ServerGroup:

    args: Any
    pg: Any
    all_engines: list
    num_gpus_per_engine: int
    num_new_engines: int
    worker_type: str = "regular"
    rank_offset: int = 0
    gpu_offset: int = 0
    sglang_overrides: dict = dataclasses.field(default_factory=dict)
    needs_offload: bool = False
    model_path: str | None = None
    router_ip: str | None = None
    router_port: int | None = None

    @property
    def nodes_per_engine(self):
        return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

    @property
    def engines(self):
        return self.all_engines[:: self.nodes_per_engine]

    def start_engines(self, port_cursors: dict[int, int] | None = None) -> tuple[list, dict[int, int]]:
        if port_cursors is None:
            port_cursors = {}
        if self.args.debug_train_only or self.worker_type == "placeholder":
            self.num_new_engines = 0
            return [], port_cursors

        num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

        pg, reordered_bundle_indices, reordered_gpu_ids = self.pg

        RolloutRayActor = ray.remote(SGLangEngine)

        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = 0.2
            num_cpus = num_gpus

            gpu_index = self.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
                key: os.environ.get(key, default_val)
                for key, default_val in {
                    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "true",
                    "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "true",
                    "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                    "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                    "SLIME_ENABLE_PROFILING": "true",
                }.items()
            }
            env_vars |= {
                key: os.environ[key]
                for key in (
                    "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_DEBUG",
                    "GLOO_SOCKET_IFNAME",
                )
                if os.environ.get(key)
            }
            if os.environ.get("NCCL_SOCKET_IFNAME"):
                env_vars.setdefault("GLOO_SOCKET_IFNAME", os.environ["NCCL_SOCKET_IFNAME"])
            if getattr(self.args, "colocate", False):
                import torch_memory_saver as _tms

                _dynlib = os.path.join(
                    os.path.dirname(os.path.dirname(_tms.__file__)),
                    "torch_memory_saver_hook_mode_preload.abi3.so",
                )
                if os.path.exists(_dynlib):
                    env_vars["LD_PRELOAD"] = _dynlib
                    env_vars["TMS_INIT_ENABLE"] = "1"
                    env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
            rollout_engine = RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                runtime_env={
                    "env_vars": env_vars,
                },
            ).remote(
                self.args,
                rank=global_rank,
                worker_type=self.worker_type,
                base_gpu_id=base_gpu_id,
                sglang_overrides=self.sglang_overrides,
                num_gpus_per_engine=self.num_gpus_per_engine,
            )

            rollout_engines.append((global_rank, rollout_engine))
            self.all_engines[i] = rollout_engine

        self.num_new_engines = len(rollout_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        if self.args.rollout_external:
            addr_and_ports = _allocate_rollout_engine_addr_and_ports_external(
                args=self.args, rollout_engines=rollout_engines
            )
        else:
            base_port = max(port_cursors.values()) if port_cursors else 15000
            addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
                args=self.args,
                rollout_engines=rollout_engines,
                worker_type=self.worker_type,
                num_gpus_per_engine=self.num_gpus_per_engine,
                rank_offset=self.rank_offset,
                base_port=base_port,
            )

        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self.router_ip,
                router_port=self.router_port,
            )
            for rank, engine in rollout_engines
        ]
        return init_handles, port_cursors

    def offload(self):
        if not self.needs_offload:
            return []
        return [engine.release_memory_occupation.remote() for engine in self.engines if engine is not None]

    def onload(self, tags: list[str] | None = None):
        if not self.needs_offload:
            return []
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]

    def onload_weights_from_disk(self):
        if not self.needs_offload or not self.model_path:
            return []
        return [
            engine.update_weights_from_disk.remote(self.model_path) for engine in self.engines if engine is not None
        ]


@dataclasses.dataclass
class RolloutServer:

    server_groups: list[ServerGroup]
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True

    @property
    def engines(self):
        return [e for g in self.server_groups for e in g.engines]

    @property
    def all_engines(self):
        return [e for g in self.server_groups for e in g.all_engines]

    @property
    def num_new_engines(self):
        return sum(g.num_new_engines for g in self.server_groups)

    @num_new_engines.setter
    def num_new_engines(self, value):
        for g in self.server_groups:
            g.num_new_engines = value

    @property
    def engine_gpu_counts(self) -> list[int]:
        return [g.num_gpus_per_engine for g in self.server_groups for _ in g.engines]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        offsets = []
        for g in self.server_groups:
            for j in range(len(g.engines)):
                offsets.append(g.gpu_offset + j * g.num_gpus_per_engine)
        return offsets

    @property
    def nodes_per_engine(self):
        values = {g.nodes_per_engine for g in self.server_groups if g.worker_type != "placeholder"}
        if len(values) != 1:
            raise ValueError(f"Heterogeneous nodes_per_engine across groups: {values}")
        return values.pop()

    def recover(self):
        dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] for g in self.server_groups]

        all_handles = []
        port_cursors: dict[int, int] = {}
        for g in self.server_groups:
            handles, port_cursors = g.start_engines(port_cursors)
            all_handles.extend(handles)
        if all_handles:
            ray.get(all_handles)

        release_handles = []
        updatable_new_engines = []
        non_updatable_groups_engines: list[tuple[str, list]] = []
        for g, dead_indices in zip(self.server_groups, dead_per_group, strict=True):
            logger.info(f"Recovered {g.num_new_engines} dead rollout engines (worker_type={g.worker_type})")
            assert g.num_new_engines == len(dead_indices), "num_new_engines does not match dead_indices length"
            if g.needs_offload and dead_indices:
                new_engines = [g.all_engines[i] for i in dead_indices]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in new_engines)
                if self.update_weights:
                    updatable_new_engines.extend(new_engines)
                elif g.model_path:
                    non_updatable_groups_engines.append((g.model_path, new_engines))

        if release_handles:
            ray.get(release_handles)
            all_resume_engines = updatable_new_engines[:]
            for _model_path, engines in non_updatable_groups_engines:
                all_resume_engines.extend(engines)
            if all_resume_engines:
                ray.get(
                    [
                        engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                        for engine in all_resume_engines
                    ]
                )

    def offload(self):
        pairs = []
        for g in self.server_groups:
            if not g.needs_offload:
                continue
            pairs.extend(
                (engine, engine.release_memory_occupation.remote())
                for engine in g.engines
                if engine is not None
            )
        if not pairs:
            return []

        results, failures = [], []
        for engine, handle in pairs:
            try:
                results.append(ray.get(handle))
            except Exception as exc:  # noqa: BLE001 - re-raised below unless dead
                failures.append((engine, exc))
        if not failures:
            return results

        fatal = []
        for engine, exc in failures:
            try:
                alive = ray.get(
                    engine.scheduler_alive.remote(), timeout=OFFLOAD_PROBE_TIMEOUT_S
                )
            except Exception as probe_exc:  # noqa: BLE001 - unprovable => alive
                logger.warning(f"offload: liveness probe failed, assuming alive: {probe_exc}")
                alive = True
            if alive:
                fatal.append((engine, exc))
            else:
                logger.warning(
                    f"offload: engine {engine} failed to release memory and its "
                    f"scheduler is gone; its GPU memory went with the process. "
                    f"Continuing without it: {exc}"
                )
        if fatal:
            for engine, exc in fatal:
                logger.error(f"offload: engine {engine} failed to release memory: {exc}")
            raise fatal[0][1]
        return results

    def onload(self, tags: list[str] | None = None):
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags))
        return ray.get(handles) if handles else []

    def onload_weights(self):
        handles = []
        for g in self.server_groups:
            if not g.needs_offload:
                continue
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        return ray.get(handles) if handles else []

    def onload_kv(self):
        handles = []
        for g in self.server_groups:
            handles.extend(g.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]))
        return ray.get(handles) if handles else []


def _rollout_traj_cache_dir(args) -> str | None:
    if os.environ.get("MAI_ROLLOUT_TRAJ_CACHE", "0") != "1":
        return None
    d = os.environ.get("MAI_ROLLOUT_TRAJ_CACHE_DIR") or None
    if d is None:
        save = getattr(args, "save", None)
        if not save:
            return None
        d = os.path.join(save, "rollout_traj_cache")
    return d


def _rollout_traj_cache_path(cache_dir: str, rollout_id: int) -> str:
    return os.path.join(cache_dir, f"rollout_{rollout_id}.pt")


def _save_rollout_traj_cache(cache_dir: str, rollout_id: int, samples) -> None:
    if not samples:
        return
    try:
        os.makedirs(cache_dir, exist_ok=True)
        path = _rollout_traj_cache_path(cache_dir, rollout_id)
        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save({"rollout_id": rollout_id, "samples": [s.to_dict() for s in samples]}, tmp)
        os.replace(tmp, path)
        logger.info(
            "[rollout-traj-cache] saved rollout %d (%d samples) -> %s",
            rollout_id, len(samples), path,
        )
    except Exception:  # noqa: BLE001
        logger.exception("[rollout-traj-cache] save failed for rollout %d", rollout_id)


def _load_rollout_traj_cache(cache_dir: str, rollout_id: int):
    path = _rollout_traj_cache_path(cache_dir, rollout_id)
    if not os.path.exists(path):
        return None
    try:
        blob = torch.load(path, weights_only=False)
        samples = [Sample.from_dict(s) for s in blob["samples"]]
        if not samples:
            logger.warning("[rollout-traj-cache] empty rollout %d; re-sampling", rollout_id)
            return None
        logger.info(
            "[rollout-traj-cache] HIT rollout %d (%d samples) <- %s (skip sampling)",
            rollout_id, len(samples), path,
        )
        return samples
    except Exception:  # noqa: BLE001 — corrupt/partial cache => re-sample, don't crash
        logger.exception("[rollout-traj-cache] load failed for rollout %d; re-sampling", rollout_id)
        return None


def _cleanup_rollout_traj_cache(cache_dir: str | None, upto_rollout_id: int) -> None:
    import re

    if not cache_dir or not os.path.isdir(cache_dir):
        return
    pat = re.compile(r"^rollout_(\d+)\.pt$")
    try:
        for name in os.listdir(cache_dir):
            m = pat.match(name)
            if m and int(m.group(1)) <= upto_rollout_id:
                try:
                    os.remove(os.path.join(cache_dir, name))
                except OSError:
                    pass
    except Exception:  # noqa: BLE001
        logger.exception("[rollout-traj-cache] cleanup failed")


@ray.remote
class RolloutManager:

    def __init__(self, args, pg):
        configure_logger()

        self.pg = pg
        self.args = args

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        self.custom_convert_samples_to_train_data_func = None
        if self.args.custom_convert_samples_to_train_data_path is not None:
            self.custom_convert_samples_to_train_data_func = load_function(
                self.args.custom_convert_samples_to_train_data_path
            )
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            self.servers: dict[str, RolloutServer] = {}
        else:
            if int(os.environ.get("MAI_GYM_ROLLOUT_WORKERS") or "0") != 0 and args.use_distributed_post:
                raise ValueError("MAI distributed trajectories cannot nest distributed POST")
            init_http_client(args)
            self.servers = start_rollout_servers(args, pg)

        init_tracking(args, primary=False)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1

        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
            self._ci_fault_injection_pending = self.args.ci_test

    def _get_metrics_router_addr(self) -> str | None:
        srv = self.server
        if srv is None or srv.router_ip is None:
            return None
        return f"http://{srv.router_ip}:{srv.router_port}"

    def get_metrics_router_addr(self) -> str | None:
        return self._get_metrics_router_addr()

    def _try_ci_fault_injection(self):
        if not self._ci_fault_injection_pending:
            return

        self._ci_fault_injection_pending = False

        if self.server and self.server.server_groups[0].all_engines and self.server.server_groups[0].all_engines[0]:
            logger.info("CI Fault Injection: Simulating crash on engine 0 during generate")
            try:
                self.server.server_groups[0].all_engines[0].simulate_crash.remote()
                wait_time = self.args.rollout_health_check_interval + self.args.rollout_health_check_timeout + 5
                logger.info(f"CI Fault Injection: Waiting {wait_time}s for health monitor to detect crash")
                time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"CI Fault Injection failed: {e}")

    def dispose(self):
        for monitor in self._health_monitors:
            monitor.stop()
        logging_utils.finish_tracking(self.args)

    @property
    def server(self) -> RolloutServer | None:
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    def _get_updatable_server(self) -> RolloutServer | None:
        for srv in self.servers.values():
            if srv.update_weights:
                return srv
        return None

    @property
    def rollout_engines(self):
        return [e for srv in self.servers.values() for e in srv.engines]

    def get_updatable_engines_and_lock(self):
        srv = self._get_updatable_server()
        engines = srv.engines if srv else []
        gpu_counts = srv.engine_gpu_counts if srv else []
        gpu_offsets = srv.engine_gpu_offsets if srv else []
        num_new = srv.num_new_engines if srv else 0
        return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets

    def get_num_rollout_per_epoch(self):
        assert self.args.rollout_global_dataset
        if os.environ.get("MAI_TASK_SAMPLING") == "random_epoch":
            size, batch = len(self.data_source), self.args.rollout_batch_size
            if size <= 0 or batch <= 0:
                raise ValueError("random epochs require positive dataset and batch sizes")
            return (size + batch - 1) // batch
        return len(self.data_source) // self.args.rollout_batch_size

    def generate(self, rollout_id):
        start_time = time.time()
        self.rollout_id = rollout_id
        self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            self._try_ci_fault_injection()
        data, metrics = self._get_rollout_data(rollout_id=rollout_id)
        if not data:
            logger.warning("Rollout %d is empty; skipping optimizer update", rollout_id)
            return None
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)
        _log_rollout_data(rollout_id, self.args, data, metrics, time.time() - start_time)
        if self.args.debug_rollout_only:
            return
        data = self._convert_samples_to_train_data(data)
        return self._split_train_data_by_dp(data, self.train_parallel_config["dp_size"])

    def _publish_env_rollout_nodes(self):
        if int(os.environ.get("MAI_GYM_ROLLOUT_WORKERS") or "0") == 0:
            return
        if self.args.use_distributed_post:
            raise ValueError("MAI distributed trajectories cannot nest distributed POST")
        server = self._get_updatable_server() or self.server
        engines = [
            engine for group in server.server_groups for engine in group.all_engines
            if engine is not None
        ] if server is not None else []
        if not engines:
            raise RuntimeError("No active rollout engines for trajectory worker placement")
        self.args.mai_gym_rollout_node_ids = sorted(set(ray.get(
            [engine.get_runtime_node_id.remote() for engine in engines], timeout=60,
        )))

    def eval(self, rollout_id):
        if self.args.debug_train_only:
            return
        self.health_monitoring_resume()

        self._publish_env_rollout_nodes()
        result = call_rollout_fn(self.eval_generate_rollout, self.args, rollout_id, self.data_source, evaluation=True)
        data = result.data
        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=True)
        _log_eval_rollout_data(rollout_id, self.args, data, result.metrics)

    def save(self, rollout_id):
        self.data_source.save(rollout_id)
        keep_env = os.environ.get("MAI_ROLLOUT_TRAJ_CACHE_KEEP")
        if keep_env is not None:
            try:
                keep = max(0, int(keep_env))
            except ValueError:
                keep = 0
            _cleanup_rollout_traj_cache(_rollout_traj_cache_dir(self.args), rollout_id - keep)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)

    def offload(self):
        self.health_monitoring_pause()
        for srv in self.servers.values():
            srv.offload()

    def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            srv.onload(tags)

    def onload_weights(self):
        for srv in self.servers.values():
            srv.onload_weights()

    def onload_kv(self):
        for srv in self.servers.values():
            srv.onload_kv()

    def recover_updatable_engines(self):
        self.health_monitoring_pause()
        srv = self._get_updatable_server()
        if self.rollout_id == -1 or srv is None:
            engines = srv.engines if srv else []
            gpu_counts = srv.engine_gpu_counts if srv else []
            gpu_offsets = srv.engine_gpu_offsets if srv else []
            return engines, self.rollout_engine_lock, (srv.num_new_engines if srv else 0), gpu_counts, gpu_offsets

        srv.recover()
        return (
            srv.engines,
            self.rollout_engine_lock,
            srv.num_new_engines,
            srv.engine_gpu_counts,
            srv.engine_gpu_offsets,
        )

    def clear_updatable_num_new_engines(self):
        srv = self._get_updatable_server()
        if srv:
            srv.num_new_engines = 0

    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    def check_weights(self, action: str):
        return ray.get([engine.check_weights.remote(action=action) for engine in self.rollout_engines])

    def _get_rollout_data(self, rollout_id):
        cache_dir = _rollout_traj_cache_dir(self.args)
        if cache_dir is not None:
            cached = _load_rollout_traj_cache(cache_dir, rollout_id)
            if cached is not None:
                if not self.args.disable_rollout_trim_samples and self.args.use_dynamic_global_batch_size:
                    self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(len(cached))
                return cached, None

        if self.args.load_debug_rollout_data:
            data = torch.load(
                self.args.load_debug_rollout_data.format(rollout_id=rollout_id),
                weights_only=False,
            )["samples"]
            data = [Sample.from_dict(sample) for sample in data]
            if (ratio := self.args.load_debug_rollout_data_subsample) is not None:
                original_num_rows = len(data)
                rough_subsample_num_rows = int(original_num_rows * ratio)
                data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
                logger.info(
                    f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
                )
            metrics = None
        else:
            self._publish_env_rollout_nodes()
            data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
            metrics = data.metrics
            data = data.samples
            while data and isinstance(data[0], list):
                data = list(itertools.chain.from_iterable(data))

            if data and not self.args.disable_rollout_trim_samples and not self.args.debug_rollout_only:
                global_batch_size = self.args.global_batch_size
                if self.args.use_dynamic_global_batch_size:
                    logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
                    self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(len(data))
                    global_batch_size = self._dynamic_global_batch_size

                if len(data) % global_batch_size != 0:
                    origin_data_length = len(data)
                    trim_len = (len(data) // global_batch_size) * global_batch_size
                    allow_upsample = os.environ.get("SLIME_ALLOW_UPSAMPLE", "0") == "1"

                    if trim_len == 0 and allow_upsample:
                        upsample_len = global_batch_size
                        pad_count = upsample_len - len(data)
                        pad_indices = [random.randint(0, len(data) - 1) for _ in range(pad_count)]
                        data = data + [data[i] for i in pad_indices]
                        logger.info(
                            f"upsample number of samples from {origin_data_length} to {len(data)} "
                            f"(+{pad_count} duplicated samples)"
                        )
                    elif trim_len == 0:
                        raise ValueError(
                            f"Not enough samples {len(data)} for global_batch_size {global_batch_size}. "
                            f"Set SLIME_ALLOW_UPSAMPLE=1 to pad with duplicated samples instead."
                        )
                    else:
                        data = data[:trim_len]
                        logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
                logger.info(f"Final collected {len(data)} samples from rollout to train")

        if cache_dir is not None and not self.args.load_debug_rollout_data:
            _save_rollout_traj_cache(cache_dir, rollout_id, data)
        return data, metrics

    def _compute_dynamic_global_batch_size(self, num_samples: int) -> int:
        dp_size = self.train_parallel_config["dp_size"]
        original_gbs = self.args.global_batch_size

        dynamic_gbs = (num_samples // dp_size) * dp_size

        if dynamic_gbs == 0:
            dynamic_gbs = dp_size
            logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

        wasted = num_samples - dynamic_gbs

        if dynamic_gbs != original_gbs or wasted > 0:
            logger.info(
                f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} (num_samples={num_samples}, dp_size={dp_size}, num_steps=1, wasted={wasted})"
            )

        return dynamic_gbs

    def _save_debug_rollout_data(self, data, rollout_id, evaluation: bool):
        if (path_template := self.args.save_debug_rollout_data) is not None:
            path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
            logger.info(f"Save debug rollout data to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

            if evaluation:
                dump_data = dict(
                    samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
                )
            else:
                dump_data = dict(
                    samples=[sample.to_dict() for sample in data],
                )

            torch.save(dict(rollout_id=rollout_id, **dump_data), path)

    def _post_process_rewards(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: list[Sample] | list[list[Sample]]):
        if self.custom_convert_samples_to_train_data_func is not None:
            return self.custom_convert_samples_to_train_data_func(self.args, samples)

        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }

        loss_masks = []
        for sample in samples:
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length

            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            if sample.remove_sample:
                sample.loss_mask = [0] * sample.response_length
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        if any(sample.metadata and "raw_reward" in sample.metadata for sample in samples):
            train_data["raw_reward"] = [
                sample.metadata["raw_reward"] if sample.metadata and "raw_reward" in sample.metadata else sample.reward
                for sample in samples
            ]

        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].rollout_routed_experts is not None:
            train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        if any(sample.multimodal_train_inputs is not None for sample in samples):
            train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

        if any(sample.multimodal_inputs is not None for sample in samples):
            train_data["multimodal_inputs"] = [sample.multimodal_inputs for sample in samples]

        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        return train_data

    def set_train_parallel_config(self, config: dict):
        self.train_parallel_config = config

    def _split_train_data_by_dp(self, data, dp_size):
        rollout_data = {}

        if "prompt" in data:
            rollout_data["prompt"] = data["prompt"]

        total_lengths = [len(t) for t in data["tokens"]]
        data["total_lengths"] = total_lengths

        if self.args.balance_data:
            partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
        else:
            partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

        rollout_data_refs = []

        for i in range(dp_size):
            rollout_data = {}
            partition = partitions[i]
            rollout_data["partition"] = partition
            for key in [
                "tokens",
                "multimodal_train_inputs",
                "multimodal_inputs",
                "response_lengths",
                "rewards",
                "truncated",
                "loss_masks",
                "round_number",
                "sample_indices",
                "rollout_log_probs",
                "rollout_routed_experts",
                "prompt",
                "teacher_log_probs",
            ]:
                if key not in data:
                    continue
                val = [data[key][j] for j in partition]
                rollout_data[key] = val
            for key in [
                "raw_reward",
                "total_lengths",
            ]:
                if key not in data:
                    continue
                rollout_data[key] = data[key]
            if hasattr(self, "_dynamic_global_batch_size"):
                rollout_data["dynamic_global_batch_size"] = self._dynamic_global_batch_size
            rollout_data_refs.append(Box(ray.put(rollout_data)))
        return rollout_data_refs


def _allocate_rollout_engine_addr_and_ports_external(args, rollout_engines):
    addr_and_ports = {}
    for rank, _ in rollout_engines:
        addr = args.rollout_external_engine_addrs[rank]
        [host, port] = addr.split(":")
        addr_and_ports[rank] = dict(
            dist_init_addr=addr,
            nccl_port=None,
            host=host,
            port=int(port),
        )
    return addr_and_ports


def _allocate_rollout_engine_addr_and_ports_normal(
    *,
    args,
    rollout_engines,
    worker_type="regular",
    num_gpus_per_engine=None,
    rank_offset=0,
    base_port=15000,
):
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    num_engines_per_node = max(1, args.num_gpus_per_node // _gpus_per_engine)
    addr_and_ports: dict[int, dict] = {}

    node_port_cursor: dict[int, int] = {}

    visited_nodes = set()
    for rank, engine in rollout_engines:
        local_rank = rank - rank_offset
        node_index = local_rank // num_engines_per_node
        if node_index in visited_nodes:
            continue
        visited_nodes.add(node_index)
        num_engines_on_this_node = num_engines_per_node - (local_rank % num_engines_per_node)

        def get_addr_and_ports(engine, node_idx):
            start_port = node_port_cursor.get(node_idx, base_port)

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                node_port_cursor[node_idx] = start_port
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports(engine, node_index)

        for i in range(num_engines_on_this_node):
            current_rank = rank + i
            addr_and_ports.setdefault(current_rank, {})
            addr_and_ports[current_rank]["host"] = get_addr()
            addr_and_ports[current_rank]["port"] = get_port()
            addr_and_ports[current_rank]["nccl_port"] = get_port()

            if worker_type == "prefill":
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()

        if _gpus_per_engine > args.num_gpus_per_node:
            num_node_per_engine = _gpus_per_engine // args.num_gpus_per_node
            if local_rank % num_node_per_engine == 0:
                dist_init_addr = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports.setdefault(rank + i, {})
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_on_this_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(30 + args.sglang_dp_size)}"

    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")

    return addr_and_ports, node_port_cursor


def _start_router(args, *, has_pd_disaggregation: bool = False, force_new: bool = False) -> tuple[str, int]:
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    router_ip = _wrap_ipv6(get_host_info()[1])
    if force_new:
        router_port = find_available_port(random.randint(3000, 4000))
    else:
        router_port = args.sglang_router_port
        if router_port is None:
            router_port = find_available_port(random.randint(3000, 4000))

    from sglang_router.launch_router import RouterArgs

    from slime.utils.http_utils import run_router

    router_args = RouterArgs.from_cli_args(args, use_router_prefix=True)
    router_args.host = router_ip
    router_args.port = router_port
    router_args.prometheus_port = find_available_port(random.randint(4000, 5000))
    router_args.log_level = "warn"
    router_args.request_timeout_secs = args.sglang_router_request_timeout_secs

    if has_pd_disaggregation:
        router_args.pd_disaggregation = True
        router_args.disable_circuit_breaker = True

    router_args.disable_health_check = True

    logger.info(f"Launch router with args: {router_args}")

    process = multiprocessing.Process(
        target=run_router,
        args=(router_args,),
    )
    process.daemon = True
    process.start()
    time.sleep(3)
    assert process.is_alive()
    logger.info(f"Router launched at {router_ip}:{router_port}, Prometheus port: {router_args.prometheus_port}")
    return router_ip, router_port


def _compute_rollout_offset(args) -> int:
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    if args.debug_rollout_only:
        return 0
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    return num


def start_rollout_servers(args, pg) -> dict[str, RolloutServer]:
    config = _resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    gpu_offset = 0
    engine_offset = 0

    rollout_pg_offset = _compute_rollout_offset(args)
    megatron_num_gpus = _compute_megatron_num_gpus(args)

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)

        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        server_groups: list[ServerGroup] = []
        port_cursors: dict[int, int] = {}

        has_epd = model_cfg.has_encoder_disaggregation

        def _make_group(group_cfg, router_ip, router_port, overrides_extra=None):
            nonlocal engine_offset, gpu_offset
            gpus_per_engine = group_cfg.num_gpus_per_engine
            num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
            num_engines = group_cfg.num_gpus // num_gpu_per_engine_local

            group_abs_start = rollout_pg_offset + gpu_offset
            needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
            overrides = dict(group_cfg.overrides)
            if overrides_extra:
                for k, v in overrides_extra.items():
                    overrides.setdefault(k, v)
            if args.offload_rollout and not needs_offload:
                overrides.setdefault("enable_memory_saver", False)
            logger.info(
                f"Engine group '{group_cfg.worker_type}' gpu_offset={gpu_offset} "
                f"(abs={group_abs_start}): needs_offload={needs_offload}"
            )

            group = ServerGroup(
                args=args,
                pg=pg,
                all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else [],
                num_gpus_per_engine=gpus_per_engine,
                num_new_engines=0,
                worker_type=group_cfg.worker_type,
                rank_offset=engine_offset,
                gpu_offset=gpu_offset,
                sglang_overrides=overrides,
                needs_offload=needs_offload,
                model_path=overrides.get("model_path", args.hf_checkpoint),
                router_ip=router_ip,
                router_port=router_port,
            )
            engine_offset += num_engines
            gpu_offset += group_cfg.num_gpus
            return group

        if has_epd:
            encoder_urls: list[str] = []
            for group_cfg in model_cfg.server_groups:
                if group_cfg.worker_type != "encoder":
                    continue
                group = _make_group(group_cfg, router_ip, router_port)
                handles, port_cursors = group.start_engines(port_cursors)
                if handles:
                    ray.get(handles)
                urls = ray.get([e.get_url.remote() for e in group.engines])
                encoder_urls.extend(u for u in urls if u is not None)
                server_groups.append(group)

            logger.info(f"EPD phase 1 done: collected {len(encoder_urls)} encoder URLs: {encoder_urls}")

            non_encoder_handles: list = []
            for group_cfg in model_cfg.server_groups:
                if group_cfg.worker_type == "encoder":
                    continue
                overrides_extra = {}
                if encoder_urls and group_cfg.worker_type in ("prefill", "regular"):
                    overrides_extra["language_only"] = True
                    overrides_extra["encoder_urls"] = encoder_urls
                group = _make_group(group_cfg, router_ip, router_port, overrides_extra=overrides_extra)
                handles, port_cursors = group.start_engines(port_cursors)
                non_encoder_handles.extend(handles)
                server_groups.append(group)

            if non_encoder_handles:
                ray.get(non_encoder_handles)
        else:
            all_init_handles: list = []
            for group_cfg in model_cfg.server_groups:
                group = _make_group(group_cfg, router_ip, router_port)
                handles, port_cursors = group.start_engines(port_cursors)
                all_init_handles.extend(handles)
                server_groups.append(group)

            if all_init_handles:
                ray.get(all_init_handles)

        servers[model_cfg.name] = RolloutServer(
            server_groups=server_groups,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
        )

    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


def _resolve_sglang_config(args) -> SglangConfig:
    if getattr(args, "sglang_config", None) is not None:
        config = SglangConfig.from_yaml(args.sglang_config)
        expected = args.rollout_num_gpus
        actual = config.total_num_gpus
        assert actual == expected, f"sglang_config total GPUs ({actual}) != rollout_num_gpus ({expected})"
        return config

    if args.prefill_num_servers is not None:
        return SglangConfig.from_prefill_num_servers(args)

    return SglangConfig(
        models=[
            ModelConfig(
                name="default",
                server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
            )
        ]
    )


def _log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate or getattr(args, "log_eval_passrate", False):
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def _log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    return log_dict


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}

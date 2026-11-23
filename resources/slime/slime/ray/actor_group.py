# Modified by MAI-Gym contributors.
import os

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST


class RayTrainGroup:

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int], list[int]],
        num_gpus_per_actor: float = 1,
        role: str = "actor",
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.role = role

        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        assert pg is not None
        pg, reordered_bundle_indices, _reordered_gpu_ids = pg

        env_vars = {
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1"),
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **self.args.train_env_vars,
        }

        if self.args.offload_train and self.args.train_backend == "megatron":
            import torch_memory_saver

            dynlib_path = os.path.join(
                os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                "torch_memory_saver_hook_mode_preload.abi3.so",
            )
            assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

            env_vars["LD_PRELOAD"] = dynlib_path
            env_vars["TMS_INIT_ENABLE"] = "1"
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

        if self.args.use_routing_replay and self.role == "actor":
            env_vars["ENABLE_ROUTING_REPLAY"] = "1"

        from slime.backends.megatron_utils.actor import MegatronTrainRayActor

        actor_impl = MegatronTrainRayActor

        TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(actor_impl)

        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    def async_init(self, args, role, with_ref=False, with_opd_teacher=False):
        self.args = args
        return [
            actor.init.remote(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher)
            for actor in self._actor_handlers
        ]

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        if isinstance(external_data, list):
            assert len(external_data) == len(self._actor_handlers)
            return [
                actor.train.remote(rollout_id, rollout_data_ref, external_data=ed)
                for actor, ed in zip(self._actor_handlers, external_data, strict=False)
            ]
        return [
            actor.train.remote(rollout_id, rollout_data_ref, external_data=external_data)
            for actor in self._actor_handlers
        ]

    def save_model(self, rollout_id, force_sync=False):
        return ray.get([actor.save_model.remote(rollout_id, force_sync=force_sync) for actor in self._actor_handlers])

    def update_weights(self):
        return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])

    def onload(self):
        return ray.get([actor.wake_up.remote() for actor in self._actor_handlers])

    def offload(self):
        return ray.get([actor.sleep.remote() for actor in self._actor_handlers])

    def clear_memory(self):
        return ray.get([actor.clear_memory.remote() for actor in self._actor_handlers])

    def set_rollout_manager(self, rollout_manager):
        return ray.get([actor.set_rollout_manager.remote(rollout_manager) for actor in self._actor_handlers])

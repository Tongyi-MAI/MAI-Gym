# Modified by MAI-Gym contributors.
# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ray/utils.py#L1
import os

import ray
import torch
from slime.ray.ray_actor import RayActor


NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]


def ray_noset_visible_devices(env_vars=os.environ):
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


@ray.remote
class Lock(RayActor):
    def __init__(self):
        self._locked = False

    def acquire(self):
        if not self._locked:
            self._locked = True
            return True
        return False

    def release(self):
        assert self._locked, "Lock is not acquired, cannot release."
        self._locked = False

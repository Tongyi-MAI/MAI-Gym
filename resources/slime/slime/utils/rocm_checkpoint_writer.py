# Modified by MAI-Gym contributors.
import torch
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync


class ROCmFileSystemWriterAsync(FileSystemWriterAsync):

    @staticmethod
    def preload_tensors(*args, **kwargs):
        if torch.version.hip:
            print("HIP/ROCm detected: setting non_blocking=False in preload_tensors")
            if "non_blocking" in kwargs:
                kwargs["non_blocking"] = False
            elif len(args) > 1 and isinstance(args[-1], bool):
                args = args[:-1] + (False,)

        return FileSystemWriterAsync.preload_tensors(*args, **kwargs)

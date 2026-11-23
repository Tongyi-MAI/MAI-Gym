# Modified by MAI-Gym contributors.
import datetime
import logging
import os
from slime.utils.misc import SingletonMeta

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

__all__ = ["_TensorboardAdapter"]

logger = logging.getLogger(__name__)


class _TensorboardAdapter(metaclass=SingletonMeta):
    _writer = None


    def __init__(self, args):
        assert args.use_tensorboard, f"{args.use_tensorboard=}"
        tb_project_name = args.tb_project_name
        tb_experiment_name = args.tb_experiment_name
        if not os.environ.get("TENSORBOARD_DIR") and getattr(args, "tensorboard_dir", None):
            os.environ["TENSORBOARD_DIR"] = args.tensorboard_dir
        if tb_project_name is not None or os.environ.get("TENSORBOARD_DIR", None):
            if tb_project_name is not None and tb_experiment_name is None:
                tb_experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._initialize(tb_project_name, tb_experiment_name)
        else:
            raise ValueError("tb_project_name and tb_experiment_name, or TENSORBOARD_DIR are required")

    def _initialize(self, tb_project_name, tb_experiment_name):
        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{tb_project_name}/{tb_experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger.info(f"Saving tensorboard log to {tensorboard_dir}.")
        self._writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self._writer.add_scalar(key, data[key], step)

    def finish(self):
        self._writer.close()

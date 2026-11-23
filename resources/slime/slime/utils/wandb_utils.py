# Modified by MAI-Gym contributors.
import logging
import os
import re
from copy import deepcopy

import wandb

logger = logging.getLogger(__name__)


def _is_offline_mode(args) -> bool:
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    args.wandb_run_id = wandb.run.id


def reinit_wandb_primary_with_open_metrics(args, router_addr):
    if not args.use_wandb or _is_offline_mode(args):
        return
    if getattr(args, "wandb_mode", None) == "disabled":
        return
    if router_addr is None:
        return
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    import sglang_router

    if "slime" not in sglang_router.__version__:
        logger.warning(
            "Only customized sglang_router from https://github.com/zhuzilin/sgl-router supports uploading metrics."
        )
        return

    logger.info(f"Re-initializing primary W&B with SGLang metrics at {router_addr}.")

    wandb.finish()

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(
            mode="shared",
            x_primary=True,
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        ),
    }

    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)
    _init_wandb_common()


def _args_to_config_dict(args):
    return _redact_config(deepcopy(args.__dict__))


_PRIVATE_CONFIG_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?key|secret|password|credential|authorization|cookie|"
    r"(?:^|[_-])token$|^wandb_key$|checkpoint|(?:^|[_-])(?:path|dir|root|host|addr|url|uri|endpoint|file|directory)$|"
    r"^(?:load|save|ref_load)$)",
    re.IGNORECASE,
)


def _redact_config(value, key=""):
    if _PRIVATE_CONFIG_KEY.search(str(key)):
        return None if value is None else "[redacted]"
    if isinstance(value, dict):
        return {k: _redact_config(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_config(v) for v in value]
    if isinstance(value, os.PathLike):
        return "[redacted]"
    if isinstance(value, str) and (value.startswith(("/", "~/", "http://", "https://", "oss://")) or re.match(r"^[A-Za-z]:[\\/]", value)):
        return "[redacted]"
    return value



def _compute_config_for_logging(args):
    output = _args_to_config_dict(args)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    return output


def init_wandb_secondary(args):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _args_to_config_dict(args),
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")

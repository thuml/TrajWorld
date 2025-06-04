import os
import random
import sys

import numpy as np
from omegaconf import OmegaConf

from dynamics.config import MOPOConfig
from dynamics.trajworld_dynamics import TrajWorldDynamics
from dynamics.logger import Logger, make_log_dirs
from data.mix_dataloader import get_final_dataloader_v2, get_debug_dataloader, get_pendulum_dataloader, get_jat_mujoco_dataloader

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

conf_dict = OmegaConf.from_cli()
config = MOPOConfig(**conf_dict)

"""
python scripts/pretrain/pretrain_trajworld.py history_length=20 log_root_dir=log_pretrain_trajworld exp_name=merge_all n_blocks=6
"""

if __name__ == "__main__":
    log_dirs = make_log_dirs(config.env_name, config.algo, config.seed, vars(config),
                             record_params=["history_length"], root_dir=config.log_root_dir, exp_prefix=config.exp_name)
    with open(os.path.join(log_dirs, "cmd.sh"), "w") as f:
        f.write("python " + " ".join(sys.argv))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    if config.wandb:
        output_config["wandb"] = "wandb"
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(config))

    train_dataloader, val_dataloader = get_final_dataloader_v2(
        config.trm_batch_size, config.history_length, num_workers=4, root_path='heterogeneous_rl_datasets', in_memory=True)

    # seed
    random.seed(config.seed)
    np.random.seed(config.seed)

    config.obs_dim, config.act_dim = 10, 10  # dummy number
    model = TrajWorldDynamics(config, max_values=1.0, min_values=0.0)

    if config.load_pt_dynamics_path:
        model.load_weights(config.load_pt_dynamics_path)

    model.train_with_dataloader(train_dataloader, val_dataloader, logger, start_updated_iters=config.start_updated_iters)

import os
import random
import sys

import numpy as np
from omegaconf import OmegaConf

from dynamics.config import MOPOConfig
from dynamics.mlp_ensemble_dynamics import EnsembleDynamics
from dynamics.logger import Logger, make_log_dirs, snapshot_src
from data.mix_dataloader import get_final_dataloader_v2, get_debug_dataloader

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

conf_dict = OmegaConf.from_cli()
config = MOPOConfig(**conf_dict)

"""
python scripts/pretrain/pretrain_mlp.py history_length=2 log_root_dir=log_pretrain_mlp exp_name=merge_all
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

    train_dataloader, val_dataloader = get_debug_dataloader(
        config.batch_size * config.n_ensemble, config.history_length, num_workers=16, root_path='/home/NAS/rl_data/heterogeneous_rl_datasets', in_memory=False)

    # seed
    random.seed(config.seed)
    np.random.seed(config.seed)

    config.obs_dim, config.act_dim = 90, 30
    config.dynamics_hidden_dims = (640, 640, 640, 640)
    model = EnsembleDynamics(config)

    if config.load_pt_dynamics_path:
        model.load_weights(config.load_pt_dynamics_path)

    model.train_with_dataloader(train_dataloader, val_dataloader, logger, start_updated_iters=config.start_updated_iters)

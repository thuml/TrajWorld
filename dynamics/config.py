from pydantic import BaseModel
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

class MOPOConfig(BaseModel):

    algo: str = "MOPO"
    project: str = "train-MOPO"
    exp_name: str = None
    env_name: str = "halfcheetah-medium-v2"
    dataset_path: str = None
    reward_clip_min: int = None
    reward_clip_max: int = None

    seed: int = 1
    discount: float = 0.99
    obs_dim: int = None
    act_dim: int = None
    normalize_state: bool = False
    normalize_reward: bool = False
    normalize_history: bool = False
    data_size: int = int(1e8)
    log_interval: int = 1000
    log_root_dir: str = None
    load_dynamics_path: str = None
    load_policy_path: str = None
    n_jitted_updates: int = 8
    n_envs: int = 1

    epoch: int = 300
    step_per_epoch: int = 10000
    eval_episodes: int = 10
    batch_size: int = 256

    # SOFT ACTOR CRITIC
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: float = None
    num_qs: int = 2
    sample_num_qs: int = 2
    target_q_type: str = 'min'
    bc_weight: float = 0.0

    # NETWORK
    hidden_dims: Sequence[int] = (256, 256)
    add_layer_norm: bool = False
    add_critic_layer_norm: bool = False
    actor_dropout: float = None
    critic_lr: float = 3e-4
    actor_lr: float = 1e-4
    alpha_lr: float = 1e-4
    actor_cosine_decay_steps: int = None
    critic_clip_grads: float = 10000

    # DYNAMICS
    dynamics_lr: float = 1e-3
    dynamics_weight_decay: Sequence[float] = (2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4)
    dynamics_hidden_dims: Sequence[int] = (200, 200, 200, 200)
    n_ensemble: int = 7
    n_elites: int = 5
    rollout_freq: int = 1000
    rollout_batch_size: int = 50000
    rollout_length: int = 5
    max_parallel_rollouts: int = 500
    penalty_coef: float = 0.5
    model_retain_epochs: int = 5
    real_ratio: float = 0.05
    dynamics_max_epochs: int = None
    dynamics_max_epochs_since_update: int = 5
    dynamics_update_freq: int = 0
    penalty_mode: str = None
    pred_std_times: int = 3

    # TRANSFORMER
    history_length: int = 20
    trm_batch_size: int = 64
    trm_epoch_steps: int = 5000
    uniform_bin: int = 256

    n_blocks: int = 6
    embed_dim: int = 256 #384 for TDM
    n_heads: int = 4
    dropout_p: float = 0.05
    n_sincos: int = 10

    trm_variate_embed: bool = True
    trm_shuffle_variate: bool = False
    trm_weighted_ce_loss: bool = False
    trm_lr: float = 0.0001
    trm_weight_decay: float = 1e-5
    trm_clip_grads: float = 0.25
    trm_warmup_steps: int = 10000
    trm_max_steps: int = 10000000
    trm_jitted_updates: int = None
    trm_lookback_window: int = None
    trm_mask_ratio: float = 0.0
    trm_input_discrete: str = 'gauss'
    trm_target_discrete: str = 'onehot'
    trm_prompt: bool = False
    prompt_size: int = 0
    uncertainty_temp: float = 1.0
    use_kv_cache: bool = False
    train_model_only: bool = False
    test_env_name: str = None
    
    pt_max_iters: int = 1000000
    pt_val_iters: int = 100
    pt_val_interval: int = 10000
    pt_log_interval: int = 100
    load_pt_dynamics_path: str = None
    start_updated_iters: int = 0
    mf_only: bool = False
    
    force_max_reward: float = None
    use_ref_reward: bool = False
    use_diffuser: bool = False
    use_med_diffuser: bool = False
    denoise_iters: int = 3

    wandb: bool = False
    cmd: str = None
    holdout_eps_len: int = 999
    holdout_size: int = 4995

    lora_dim: int = 0
    freeze_first_layer: bool = False

    epoch_start: int = 0
    pt_path: str = None
    mlp_huge: bool = False

    percentage_for_training: float = 1.0
    id: int=0
    actor_lr_decay_alpha: float = 0.0
    clip_negative_rewards: bool = False
    train_model: bool = True

    lora_pretrain_path: str = None
    trm_logit_norm: float = 0.0
    actor_lr_decay_alpha: float = 0.0

    def __hash__(
        self,
    ):  # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())

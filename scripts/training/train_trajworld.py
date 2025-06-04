import os
import time
from typing import Dict, Tuple, Any, Callable, Optional, List
from functools import partial
import collections
import random
import sys
import gym
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pickle
import imageio
import d4rl.gym_mujoco
from matplotlib import pyplot as plt

from dynamics.config import MOPOConfig
from data.data import Batch, ReplayBuffer
from data.history_data import get_history_dataset, HistoryReplayBuffer
from dynamics.trajworld_dynamics import TrajWorldDynamics, transform_to_probs, transform_from_probs
from dynamics.logger import Logger, make_log_dirs, snapshot_src
from policy.sac import SACPolicy
from architecture.mlp_ensemble import sample_actions
from dynamics.utils import Params

from architecture.trm_traj import TrajWorldTransformer

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

"""
python scripts/training/train_trajworld.py env_name=hopper-medium-replay-v2 log_root_dir=log_model_new trm_epoch_steps=5000 dynamics_max_epochs_since_update=300 dynamics_max_epochs=50 seed=183 train_model_only=true exp_name=trajworld_ft trm_lr=1e-5 load_pt_dynamics_path="mergeall_pt/model/trm_dynamics990000.pkl" n_blocks=6
python scripts/training/train_trajworld.py env_name=hopper-medium-replay-v2 log_root_dir=log_model_new trm_epoch_steps=5000 dynamics_max_epochs_since_update=300 dynamics_max_epochs=50 seed=183 train_model_only=true exp_name=trajworld_scratch trm_lr=1e-5 n_blocks=6
"""



conf_dict = OmegaConf.from_cli()
config = MOPOConfig(**conf_dict)
config.hidden_dims = tuple(config.hidden_dims)


@partial(jax.jit, static_argnames=(
    "obs_dim",
    "act_dim",
    "trm_lookback_window",
    "rollout_length",
    "actor_apply_fn",
    "dynamics_empty_cache",
    "dynamics_terminal_fn",
    "dynamics_apply_fn",
    "penalty_coef",
    "uncertainty_mode",
    "max_batch_size",
    "use_kv_cache",
    "use_diffuser",
    "diffuser_iters",
    "diffuser_apply_fn",
    "pred_std_times",
))
def _rollout_kernel(
    rng: jax.random.PRNGKey,
    obs_dim: int,
    act_dim: int,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
    trm_lookback_window: int,
    init_histories: jnp.ndarray,
    init_history_masks: jnp.ndarray,
    rollout_length: int,
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    dynamics_min_values: jnp.ndarray,
    dynamics_max_values: jnp.ndarray,
    dynamics_support: jnp.ndarray,
    dynamics_sigma: float,
    dynamics_obs_act_indicator: jnp.ndarray,
    dynamics_empty_cache: Callable[..., Any],
    dynamics_terminal_fn: Callable[..., Any],
    dynamics_apply_fn: Callable[..., Any],
    dynamics_params: Params,
    penalty_coef: float,
    uncertainty_mode: str,
    uncertainty_temp: float,
    max_batch_size: int = 500,
    use_kv_cache: bool = False,
    use_diffuser: bool = False,
    diffuser_iters: int = None,
    diffuser_param: Params = None,
    diffuser_apply_fn: Callable[..., Any] = None,
    diffuser_mu: jnp.ndarray = None,
    diffuser_std: jnp.ndarray = None,
    pred_std_times: int = 3,
) -> Tuple[Dict[str, jnp.ndarray], Dict]:
    all_dim = obs_dim + 1 + act_dim

    num_transitions = 0
    rewards_arr = jnp.array([])
    raw_rewards_arr = jnp.array([])
    penalty_arr = jnp.array([])
    rollout_transitions = collections.defaultdict(list)

    # rollout
    histories, history_masks = init_histories[:, -trm_lookback_window-1: -1], init_history_masks[:, -trm_lookback_window-1: -1]
    masks = None
    if use_kv_cache:
        caches = dynamics_empty_cache(batch_size=histories.shape[0] * all_dim)
    for _ in range(rollout_length):
        observations = histories[:, -1, :obs_dim]
        rng, actions = sample_actions(
            rng, actor_apply_fn, actor_params,
            (observations - obs_mean) / (obs_std + 1e-5),
            temperature=1.0)
        actions = jnp.clip(actions, -1, 1)  # TODO: make this a parameter
        actions = jax.lax.stop_gradient(actions)

        # put new actions into histories
        if isinstance(histories, np.ndarray):
            histories[:, -1, obs_dim + 1:] = actions
        else:
            histories = histories.at[:, -1, obs_dim + 1:].set(actions)

        def step(
            rng: jax.random.PRNGKey,
            inputs: jnp.ndarray,
            padding_mask: jnp.ndarray,
            caches: Optional[List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]] = None,
        ):
            """
            history: [B, T, M]
            padding_mask: [B, T]
            """
            obs = inputs[:, -1, :obs_dim]
            action = inputs[:, -1, obs_dim + 1:]

            inputs = jnp.clip(inputs, dynamics_min_values, dynamics_max_values)
            inputs = transform_to_probs(inputs, dynamics_support, dynamics_sigma)
            # inputs = transform_to_onehot(inputs, self.support, self.sigma)
            obs_act_indicator = jnp.tile(dynamics_obs_act_indicator, (inputs.shape[0], inputs.shape[1], 1))

            if uncertainty_mode == "pred_std":
                inputs = jnp.concatenate([inputs] * pred_std_times, axis=0)
                obs_act_indicator = jnp.concatenate([obs_act_indicator] * pred_std_times, axis=0)
                padding_mask = jnp.concatenate([padding_mask] * pred_std_times, axis=0)
                caches = jax.tree.map(lambda x: jnp.concatenate([x] * pred_std_times, axis=0), caches)
                
                rng, key = jax.random.split(rng)
                rng, variate_key = jax.random.split(rng)
                if caches is None:
                    pred = dynamics_apply_fn(dynamics_params, inputs, obs_act_indicator,
                                            padding_mask, training=True, rngs={'dropout': key})
                else:
                    pred, updated_caches = dynamics_apply_fn(dynamics_params, inputs, obs_act_indicator,
                                                        padding_mask, caches=caches, training=True,
                                                        rngs={'dropout': key},
                                                        variate_key = variate_key,
                                                        method=TrajWorldTransformer.call_kv_cache
                                                        )
                pred = pred[:, -1, :obs_dim + 1]
                pred_prob = jax.nn.softmax(pred)

                rng, key = jax.random.split(rng)
                samples = transform_from_probs(pred_prob, dynamics_support[:obs_dim + 1])
                
                samples = samples.reshape(pred_std_times, -1, *samples.shape[1:])
                pred_std = (jnp.std(samples, axis=0) / (dynamics_max_values[:obs_dim + 1] - dynamics_min_values[:obs_dim + 1])).sum(-1)
                samples = samples[0]
                updated_caches = jax.tree.map(lambda x: x[:x.shape[0] // pred_std_times], updated_caches)
            else:
                rng, key = jax.random.split(rng)
                if caches is None:
                    pred = dynamics_apply_fn(dynamics_params, inputs, obs_act_indicator,
                                            padding_mask, training=False, rngs={'dropout': key})
                else:
                    pred, updated_caches = dynamics_apply_fn(dynamics_params, inputs, obs_act_indicator,
                                                            padding_mask, caches=caches, training=False,
                                                            # rngs={'dropout': key}  # TODO: need fix?
                                                            method=TrajWorldTransformer.call_kv_cache
                                                            )
                pred = pred[:, -1, :obs_dim + 1]
                pred_prob = jax.nn.softmax(pred)

                rng, key = jax.random.split(rng)
                # samples = transform_from_probs_sample(pred_prob, dynamics_support[:obs_dim + 1], key)
                samples = transform_from_probs(pred_prob, dynamics_support[:obs_dim + 1])

            next_obs = samples[..., :-1]
            reward = samples[..., -1:]
            terminal = dynamics_terminal_fn(obs, action, next_obs)
            info = {}
            info["raw_reward"] = reward
            pred_prob = jax.nn.softmax(pred / uncertainty_temp)
            info["confidence"] = pred_prob.max(-1)

            if penalty_coef:
                if uncertainty_mode == "entropy":
                    ent = -jnp.sum(pred_prob * jnp.log(pred_prob + 1e-8), axis=-1)
                    ent = jnp.maximum(ent + jnp.log(0.5), 0)
                    penalty = ent.mean(-1)
                elif uncertainty_mode == "entropy_max":
                    ent = -jnp.sum(pred_prob * jnp.log(pred_prob + 1e-8), axis=-1)
                    ent = jnp.maximum(ent + jnp.log(0.5), 0)
                    penalty = ent.max(-1)
                elif uncertainty_mode == "entropy_max_noclip":
                    ent = -jnp.sum(pred_prob * jnp.log(pred_prob + 1e-8), axis=-1)
                    penalty = ent.max(-1)
                elif uncertainty_mode == "confidence":
                    penalty = 1 - pred_prob.max(-1).mean(-1)
                elif uncertainty_mode == "pred_std":
                    penalty = pred_std
                else:
                    raise NotImplementedError
                penalty = jnp.expand_dims(penalty, 1).astype(jnp.float32)
                assert penalty.shape == reward.shape
                reward = reward - penalty_coef * penalty
                info["penalty"] = penalty

            if caches is None:
                return rng, next_obs, reward, terminal, info
            else:
                return rng, next_obs, reward, terminal, info, updated_caches

        next_observations, rewards, terminals, info = [], [], [], []
        if use_kv_cache:
            updated_caches = []
            for i in range(0, len(observations), max_batch_size):
                rng, _next_observations, _rewards, _terminals, _info, _updated_caches = step(
                    rng, histories[i:i + max_batch_size], history_masks[i:i + max_batch_size],
                    caches=jax.tree.map(lambda x: x[i * all_dim: (i + max_batch_size) * all_dim], caches))
                next_observations.append(_next_observations)
                rewards.append(_rewards)
                terminals.append(_terminals)
                info.append(_info)
                updated_caches.append(_updated_caches)
            caches = jax.tree.map(lambda *args: jnp.concatenate(args, axis=0), *updated_caches)
        else:
            for i in range(0, len(observations), max_batch_size):
                rng, _next_observations, _rewards, _terminals, _info = step(
                    rng, histories[i:i + max_batch_size, -trm_lookback_window:], history_masks[i:i + max_batch_size, -trm_lookback_window:])
                next_observations.append(_next_observations)
                rewards.append(_rewards)
                terminals.append(_terminals)
                info.append(_info)
        next_observations = jnp.concatenate(next_observations, axis=0)
        rewards = jnp.concatenate(rewards, axis=0)
        terminals = jnp.concatenate(terminals, axis=0)
        info = jax.tree.map(lambda *args: jnp.concatenate(args, axis=0), *info)
    
        if use_diffuser:
            def denoise(obs: jnp.ndarray, act: jnp.ndarray, next_obs: jnp.ndarray, rew: jnp.ndarray):
                obs_dim, act_dim = obs.shape[-1], act.shape[-1]
                inputs = jnp.concat([obs, act, next_obs, rew], axis=-1)
                inputs = (inputs - diffuser_mu) / diffuser_std
                noise = diffuser_apply_fn(diffuser_param, inputs)
                outputs = inputs[..., -obs_dim-1:] - noise
                outputs = jnp.concat([obs, act, outputs], axis=-1)
                outputs = diffuser_std * outputs + diffuser_mu
                outputs = outputs[..., obs_dim + act_dim:]
                return outputs[..., :-1], outputs[..., -1:]

            for j in range(diffuser_iters):
                next_observations, info["raw_reward"] = denoise(observations, actions, next_observations, info["raw_reward"])

        rollout_transitions["observations"].append(observations)
        rollout_transitions["actions"].append(actions)
        rollout_transitions["rewards"].append(rewards)
        rollout_transitions["raw_rewards"].append(info["raw_reward"])
        rollout_transitions["dones"].append(terminals)
        rollout_transitions["next_observations"].append(next_observations)

        if _ == 0:
            masks = jnp.zeros_like(terminals)
        rollout_transitions["masks"].append(masks)
        masks |= terminals

        num_transitions += len(observations)
        rewards_arr = jnp.concatenate([rewards_arr, rewards.flatten()])
        raw_rewards_arr = jnp.concatenate([raw_rewards_arr, info["raw_reward"].flatten()])
        penalty_arr = jnp.concatenate([penalty_arr, info["penalty"].flatten()])

        # update histories
        if use_kv_cache:
            next_column = jnp.concatenate([next_observations, info["raw_reward"], jnp.zeros(
                (observations.shape[0], act_dim))], axis=-1)
            histories = next_column[:, None]
            next_column = jnp.ones((observations.shape[0], 1))
            history_masks = next_column
        else:
            next_column = jnp.concatenate([next_observations, info["raw_reward"], jnp.zeros(
                (observations.shape[0], act_dim))], axis=-1)
            histories = jnp.concatenate([histories, next_column[:, None]], axis=1)[:, 1:]
            next_column = jnp.ones((observations.shape[0], 1))
            history_masks = jnp.concatenate([history_masks, next_column], axis=1)[:, 1:]

    for k, v in rollout_transitions.items():
        rollout_transitions[k] = jnp.concatenate(v, axis=0)

    rollout_transitions["rewards"] = rollout_transitions["rewards"].squeeze(-1)
    rollout_transitions["dones"] = rollout_transitions["dones"].squeeze(-1)
    rollout_transitions["raw_rewards"] = rollout_transitions["raw_rewards"].squeeze(-1)

    return rng, rollout_transitions, {
        "reward_mean": rewards_arr.mean(),
        "raw_reward_mean": raw_rewards_arr.mean(),
        "raw_reward_max": raw_rewards_arr.max(),
        "penalty_mean": penalty_arr.mean(),
        "penalty_std": penalty_arr.std()
    }


class MOPOTrainer(object):

    def __init__(self,
                 config: MOPOConfig,
                 max_values: np.ndarray,
                 min_values: np.ndarray,
                 history_buffer: HistoryReplayBuffer,
                 real_buffer: ReplayBuffer,
                 fake_buffer: ReplayBuffer,
                 logger: Logger,
                 eval_env, obs_mean, obs_std,
                 test_history_buffer=Optional[HistoryReplayBuffer]):
        if config.trm_lookback_window is None:
            self._trm_lookback_window = config.history_length - 1
        else:
            self._trm_lookback_window = min(config.trm_lookback_window, config.history_length - 1)

        if config.n_jitted_updates is not None:
            assert config.rollout_freq % config.n_jitted_updates == 0
            assert config.step_per_epoch % config.n_jitted_updates == 0
            assert config.dynamics_update_freq % config.n_jitted_updates == 0

        self.config = config
        self.rng = jax.random.PRNGKey(config.seed)
        self.policy = SACPolicy(config)
        self.model = TrajWorldDynamics(config, max_values, min_values)
        self.history_buffer = history_buffer
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.logger = logger
        self.eval_env = eval_env
        self.obs_dim = config.obs_dim
        self.act_dim = config.act_dim
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.test_history_buffer = test_history_buffer

    def jitted_rollout(
        self,
        rng: jax.random.PRNGKey,
        init_histories: jnp.ndarray,
        init_history_masks: jnp.ndarray,
        rollout_length: int,
        max_batch_size: int = 500,
        use_kv_cache: bool = False
    ):
        if self.config.use_med_diffuser:
            extra_args = [True, self.config.denoise_iters,
                self.diffuser.dynamics.params, self.diffuser.dynamics.apply_fn, self.diffuser.scaler.mu, self.diffuser.scaler.std]
        else:
            extra_args = []
        rng, rollout_transitions, rollout_info = _rollout_kernel(
            rng, self.obs_dim, self.act_dim, self.obs_mean, self.obs_std, self._trm_lookback_window,
            init_histories, init_history_masks, rollout_length,
            self.policy.actor.apply_fn, self.policy.actor.params,
            self.model.min_values, self.model.max_values,
            self.model.support, self.model.sigma,
            self.model.obs_act_indicator,
            self.model.empty_cache, self.model.terminal_fn,
            self.model.trm.apply_fn, self.model.trm.params,
            self.model.penalty_coef, self.model.uncertainty_mode, self.model.uncertainty_temp,
            max_batch_size, use_kv_cache,
            *extra_args,
            pred_std_times=self.config.pred_std_times)
        rollout_masks = ~rollout_transitions["masks"].flatten()
        rollout_info["num_transitions"] = rollout_masks.sum()
        del rollout_transitions["masks"]
        rollout_transitions = jax.tree.map(lambda x: np.asarray(x[rollout_masks]), rollout_transitions)
        return rng, rollout_transitions, rollout_info

    def replay_rollouts(self, history):
        log_dir = os.path.join(self.logger._dir, "gifs")
        os.makedirs(log_dir, exist_ok=True)
        
        env = self.eval_env
        real_frames, fake_frames, state_dist, reward_dist = [], [], [], []
        real_states, fake_states = [], []
        real_rewards, fake_rewards = [], []

        # real rollouts
        for i in range(len(history)):
            state, reward, action = history[i, :self.obs_dim], history[i, self.obs_dim], history[i, self.obs_dim + 1:]
            if i == 0:
                env.reset()
                # reset
                if hasattr(env, 'reset_with_obs'):
                    env.reset_with_obs(state)
                else:
                    env.sim.reset()
                    qpos = env.init_qpos
                    qpos[1:] = state[:env.model.nq - 1]
                    qvel = state[env.model.nq - 1:]
                    env.set_state(qpos, qvel)
            else:
                real_states.append(next_state)
                fake_states.append(state)
                real_rewards.append(next_reward)
                fake_rewards.append(reward)
                state_dist.append(np.linalg.norm(state - next_state))
                reward_dist.append(np.abs(reward - next_reward))

            real_frames.append(env.render(mode='rgb_array', height=128, width=128))
            next_state, next_reward, done, _ = env.step(action)

        # fake_rollouts
        for i in range(len(history)):
            state, reward, action = history[i, :self.obs_dim], history[i, self.obs_dim], history[i, self.obs_dim + 1:]
            env.reset()
            # reset
            if hasattr(env, 'reset_with_obs'):
                env.reset_with_obs(state)
            else:
                env.sim.reset()
                qpos = env.init_qpos
                qpos[1:] = state[:env.model.nq - 1]
                qvel = state[env.model.nq - 1:]
                env.set_state(qpos, qvel)

            fake_frames.append(env.render(mode='rgb_array', height=128, width=128))

        imageio.mimsave(os.path.join(log_dir, f'rollouts{self.log_epoch}.gif'), [np.concatenate([
            x, y, np.abs((np.float32(x)-np.float32(y))*5.).clip(0, 255).astype(np.uint8)], 1) for x, y in zip(real_frames, fake_frames)], loop=0)
        print(f"State dist: {state_dist}")
        print(f"Reward dist: {reward_dist}")


    def rollout_diagnostic(
        self,
        rng: jax.random.PRNGKey,
        init_histories: jnp.ndarray,
        init_history_masks: jnp.ndarray,
        max_batch_size: int = 500,
    ) -> Tuple[Dict[str, jnp.ndarray], Dict]:
        # rollout
        histories, history_masks = init_histories[:, -
                                                  self._trm_lookback_window:], init_history_masks[:, -self._trm_lookback_window:]
        original_actions = histories[:, -1, self.obs_dim + 1:]

        penalties = []
        noise_stds = np.array(range(0, 11)) / 10
        for noise_std in noise_stds:
            observations = histories[:, -1, :self.obs_dim]
            actions = original_actions + jax.random.normal(rng, original_actions.shape) * noise_std
            actions = jnp.clip(actions, -1, 1)

            # put new actions into histories
            if isinstance(histories, np.ndarray):
                histories[:, -1, self.obs_dim + 1:] = actions
            else:
                histories = histories.at[:, -1, self.obs_dim + 1:].set(actions)

            next_observations, rewards, terminals, info = [], [], [], []
            for i in range(0, len(observations), max_batch_size):
                rng, _next_observations, _rewards, _terminals, _info = self.model.step(
                    rng, histories[i:i + max_batch_size, -self._trm_lookback_window:], history_masks[i:i + max_batch_size, -self._trm_lookback_window:])
                next_observations.append(_next_observations)
                rewards.append(_rewards)
                terminals.append(_terminals)
                info.append(_info)
            next_observations = jnp.concatenate(next_observations, axis=0)
            rewards = jnp.concatenate(rewards, axis=0)
            terminals = jnp.concatenate(terminals, axis=0)
            info = jax.tree.map(lambda *args: jnp.concatenate(args, axis=0), *info)

            penalties.append(jnp.mean(info["penalty"].flatten()))

        from matplotlib import pyplot as plt
        plt.plot(noise_stds, penalties)
        plt.xlabel("Noise Std")
        plt.ylabel("Penalty")
        plt.savefig("penalty_noise_std.png")

    def rollout(
        self,
        rng: jax.random.PRNGKey,
        init_histories: jnp.ndarray,
        init_history_masks: jnp.ndarray,
        rollout_length: int,
        max_batch_size: int = 500,
        use_kv_cache: bool = False,
    ) -> Tuple[Dict[str, jnp.ndarray], Dict]:
        all_dim = self.obs_dim + 1 + self.act_dim

        num_transitions = 0
        rewards_arr = jnp.array([])
        raw_rewards_arr = jnp.array([])
        penalty_arr = jnp.array([])
        rollout_transitions = collections.defaultdict(list)

        # rollout
        histories, history_masks = init_histories[:, -
                                                  self._trm_lookback_window-1:-1], init_history_masks[:, -self._trm_lookback_window-1:-1]
        masks = None
        if use_kv_cache:
            caches = self.model.empty_cache(batch_size=histories.shape[0] * all_dim)
        for _ in range(rollout_length):
            observations = histories[:, -1, :self.obs_dim]
            rng, actions = self.policy.select_action(
                rng, (observations - self.obs_mean) / (self.obs_std + 1e-5), temperature=1.0)
            actions = jnp.clip(actions, -1, 1)
            actions = jax.lax.stop_gradient(actions)

            # put new actions into histories
            if isinstance(histories, np.ndarray):
                histories[:, -1, self.obs_dim + 1:] = actions
            else:
                histories = histories.at[:, -1, self.obs_dim + 1:].set(actions)

            next_observations, rewards, terminals, info = [], [], [], []
            if use_kv_cache:
                updated_caches = []
                for i in range(0, len(observations), max_batch_size):
                    rng, _next_observations, _rewards, _terminals, _info, _updated_caches = self.model.step(
                        rng, histories[i:i + max_batch_size], history_masks[i:i + max_batch_size],
                        caches=jax.tree.map(lambda x: x[i * all_dim: (i + max_batch_size) * all_dim], caches))
                    next_observations.append(_next_observations)
                    rewards.append(_rewards)
                    terminals.append(_terminals)
                    info.append(_info)
                    updated_caches.append(_updated_caches)
                caches = jax.tree.map(lambda *args: jnp.concatenate(args, axis=0), *updated_caches)
            else:
                for i in range(0, len(observations), max_batch_size):
                    rng, _next_observations, _rewards, _terminals, _info = self.model.step(
                        rng, histories[i:i + max_batch_size, -self._trm_lookback_window:], history_masks[i:i + max_batch_size, -self._trm_lookback_window:])
                    next_observations.append(_next_observations)
                    rewards.append(_rewards)
                    terminals.append(_terminals)
                    info.append(_info)
            next_observations = jnp.concatenate(next_observations, axis=0)
            rewards = jnp.concatenate(rewards, axis=0)
            terminals = jnp.concatenate(terminals, axis=0)
            info = jax.tree.map(lambda *args: jnp.concatenate(args, axis=0), *info)
            
            rng, ref_next_observations, ref_rewards, ref_terminals, ref_info = self.ref_model.step(rng, observations, actions)
            print("next obs mse", jnp.mean((next_observations - ref_next_observations) ** 2))
            print("reward mse", jnp.mean((info["raw_reward"] - ref_info["raw_reward"]) ** 2))
            print("penalty", info["penalty"].mean(), info["penalty"].std(), info["penalty"].min())
            print("ref_penalty", ref_info["penalty"].mean(), ref_info["penalty"].std(), ref_info["penalty"].min())
            plt.clf()
            plt.hist(info["penalty"].flatten(), bins=100, alpha=0.5, label="penalty")
            plt.hist(ref_info["penalty"].flatten(), bins=100, alpha=0.5, label="ref penalty")
            plt.legend()
            plt.savefig(f"penalty_hist_{_}.png")
            
            rollout_transitions["observations"].append(observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["dones"].append(terminals)
            rollout_transitions["next_observations"].append(next_observations)

            if _ == 0:
                masks = jnp.zeros_like(terminals)
            rollout_transitions["masks"].append(masks)
            masks |= terminals

            num_transitions += len(observations)
            rewards_arr = jnp.concatenate([rewards_arr, rewards.flatten()])
            raw_rewards_arr = jnp.concatenate([raw_rewards_arr, info["raw_reward"].flatten()])
            penalty_arr = jnp.concatenate([penalty_arr, info["penalty"].flatten()])

            # update histories
            if use_kv_cache:
                next_column = jnp.concatenate([next_observations, info["raw_reward"], jnp.zeros(
                    (observations.shape[0], self.act_dim))], axis=-1)
                histories = next_column[:, None]
                next_column = jnp.ones((observations.shape[0], 1))
                history_masks = next_column
            else:
                next_column = jnp.concatenate([next_observations, info["raw_reward"], jnp.zeros(
                    (observations.shape[0], self.act_dim))], axis=-1)
                histories = jnp.concatenate([histories, next_column[:, None]], axis=1)[:, 1:]
                next_column = jnp.ones((observations.shape[0], 1))
                history_masks = jnp.concatenate([history_masks, next_column], axis=1)[:, 1:]

        # self.replay_rollouts(histories[0])

        for k, v in rollout_transitions.items():
            rollout_transitions[k] = jnp.concatenate(v, axis=0)

        rollout_transitions["rewards"] = rollout_transitions["rewards"].squeeze(-1)
        rollout_transitions["dones"] = rollout_transitions["dones"].squeeze(-1)
        rollout_masks = ~rollout_transitions["masks"].flatten()
        del rollout_transitions["masks"]
        rollout_transitions = jax.tree.map(lambda x: np.asarray(x[rollout_masks]), rollout_transitions)

        return rng, rollout_transitions, {
            "num_transitions": rollout_masks.sum(),
            "reward_mean": rewards_arr.mean(),
            "raw_reward_mean": raw_rewards_arr.mean(),
            "raw_reward_max": raw_rewards_arr.max(),
            "penalty_mean": penalty_arr.mean(),
        }

    def learn(self, real_batch: Batch, fake_batch: Batch) -> Dict[str, float]:
        mix_batch = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), real_batch, fake_batch)
        return self.policy.update(mix_batch)

    def learn_n_times(self, real_batch: Batch, fake_batch: Batch) -> Dict[str, float]:
        real_batch = jax.tree.map(lambda x: x.reshape(self.config.n_jitted_updates, -1, *x.shape[1:]), real_batch)
        fake_batch = jax.tree.map(lambda x: x.reshape(self.config.n_jitted_updates, -1, *x.shape[1:]), fake_batch)
        mix_batch = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), real_batch, fake_batch)
        mix_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), mix_batch)
        return self.policy.update_n_times(mix_batch, self.config.n_jitted_updates, self.config.batch_size)

    def train_model(self):
        self.model.train(self.history_buffer.sample_all(), self.logger, max_epochs=self.config.dynamics_max_epochs,
                         max_epochs_since_update=self.config.dynamics_max_epochs_since_update,
                         holdout_eps_len=99 if 'maniskill' in self.config.env_name else self.config.holdout_eps_len,
                         holdout_size=1000 if 'maniskill' in self.config.env_name else self.config.holdout_size,
                         test_data=self.test_history_buffer.sample_all() if self.test_history_buffer is not None else None,
                         epoch_start=self.config.epoch_start,
                         percentage_for_training=self.config.percentage_for_training)

    def _evaluate(self, render=False):
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0
        frames = []
        if render:
            frames.append(self.eval_env.render(mode='rgb_array', height=128, width=128))

        rng = jax.random.PRNGKey(0)
        while num_episodes < self.config.eval_episodes:
            rng, action = self.policy.select_action(
                rng, ((obs - self.obs_mean) / (self.obs_std + 1e-5)).reshape(1, -1), temperature=0.0)

            next_obs, reward, terminal, info = self.eval_env.step(np.array(action.flatten()))

            episode_reward += reward
            episode_length += 1

            obs = next_obs
            if num_episodes == 0 and render:
                frames.append(self.eval_env.render(mode='rgb_array', height=128, width=128))

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                if 'maniskill' in self.config.env_name:
                    eval_ep_info_buffer[-1]['episode_success'] = info['success']
                num_episodes += 1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()

                print(f"Eval episode {num_episodes}: {eval_ep_info_buffer[-1]}")

        results = {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
        if 'maniskill' in self.config.env_name:
            results["eval/episode_success"] = [ep_info["episode_success"] for ep_info in eval_ep_info_buffer]
        if render:
            os.makedirs(os.path.join(self.logger._dir, "gifs"), exist_ok=True)
            imageio.mimsave(os.path.join(self.logger._dir, f'gifs/eval{self.log_epoch}.gif'), frames)
        return results

    def _evaluate_vec(self):
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        rng = jax.random.PRNGKey(0)
        while num_episodes < self.config.eval_episodes:
            rng, action = self.policy.select_action(
                rng, ((obs - self.obs_mean) / (self.obs_std + 1e-5)).reshape(self.config.n_envs, -1), temperature=0.0)

            next_obs, reward, terminal, info = self.eval_env.step(np.array(action))

            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal.all():
                for i in range(self.config.n_envs):
                    eval_ep_info_buffer.append(
                        {"episode_reward": episode_reward[i], "episode_length": episode_length}
                    )
                    if 'maniskill' in self.config.env_name:
                        eval_ep_info_buffer[-1]['episode_success'] = info[i]['success']
                num_episodes += self.config.n_envs
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()

        results = {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
        if 'maniskill' in self.config.env_name:
            results["eval/episode_success"] = [ep_info["episode_success"] for ep_info in eval_ep_info_buffer]
        return results

    def evaluate(self):
        start_time = time.time()
        results = self._evaluate_vec() if self.config.n_envs > 1 else self._evaluate(render=False)
        eval_time = time.time() - start_time
        results["eval/time"] = eval_time
        reward_mean, reward_std = np.mean(results["eval/episode_reward"]), np.std(results["eval/episode_reward"])
        if 'maniskill' in self.config.env_name:
            success_rate = np.mean(results["eval/episode_success"])

        print("evaluation time: {:.2f}s".format(eval_time))
        if 'maniskill' in self.config.env_name:
            print(f"evaluation results: {reward_mean} ± {reward_std} ({success_rate}% success)")
        else:
            print(f"evaluation results: {reward_mean} ± {reward_std}")
        return results


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    log_dirs = make_log_dirs(config.env_name, config.algo, config.seed, vars(config),
                             record_params=["penalty_coef", "rollout_length", "bc_weight", "target_q_type"], root_dir=config.log_root_dir, exp_prefix=config.exp_name)
    with open(os.path.join(log_dirs, "cmd.sh"), "w") as f:
        config.cmd = "python " + " ".join(sys.argv)
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

    env = gym.make(config.env_name)
    example_env = env
    hist_dataset, dataset, max_values, min_values, obs_mean, obs_std = get_history_dataset(
        env, config,
        dataset=pickle.load(open(config.dataset_path, 'rb')) if config.dataset_path is not None else None)

    if config.test_env_name is not None:
        test_env = gym.make(config.test_env_name)
        test_hist_dataset, test_dataset, test_max_values, test_min_values, obs_mean, obs_std = get_history_dataset(
            test_env, config)
        max_values = np.maximum(max_values, test_max_values)
        min_values = np.minimum(min_values, test_min_values)
    
    config.act_dim = dataset.actions.shape[-1]
    config.obs_dim = dataset.observations.shape[-1]
    config.target_entropy = config.target_entropy if config.target_entropy else -config.act_dim
    # config.actor_cosine_decay_steps = config.step_per_epoch * config.epoch
    if config.force_max_reward is not None:
        max_values[config.obs_dim] = config.force_max_reward
        # max_values[8] = 11.0

    # seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    env.seed(config.seed)

    history_buffer = HistoryReplayBuffer(
        config.obs_dim,
        config.act_dim,
        len(dataset.observations)*2,
        config.history_length,
    )
    history_buffer.initialize_with_dataset(hist_dataset)
    if config.test_env_name is not None:
        test_history_buffer = HistoryReplayBuffer(
            config.obs_dim,
            config.act_dim,
            len(test_dataset.observations),
            config.history_length,
        )
        test_history_buffer.initialize_with_dataset(test_hist_dataset)
    else:
        test_history_buffer = None
    real_buffer = ReplayBuffer(
        config.obs_dim,
        config.act_dim,
        len(dataset.observations),
    )
    real_buffer.initialize_with_dataset(dataset)
    fake_buffer = ReplayBuffer(
        config.obs_dim,
        config.act_dim,
        config.rollout_batch_size * config.rollout_length * config.model_retain_epochs,
    )

    trainer = MOPOTrainer(
        config,
        max_values,
        min_values,
        history_buffer,
        real_buffer,
        fake_buffer,
        logger,
        env,
        obs_mean, obs_std,
        test_history_buffer,
    )

    if config.load_dynamics_path:
        trainer.model.load(config.load_dynamics_path)
    else:
        if config.load_pt_dynamics_path:
            trainer.model.load_weights(config.load_pt_dynamics_path)

        trainer.train_model()

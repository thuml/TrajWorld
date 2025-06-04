import os
import sys
import time
from typing import Dict, Tuple, Any, Callable
from functools import partial
import collections
import random
import d4rl.gym_mujoco
import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from omegaconf import OmegaConf
import pickle

from dynamics.config import MOPOConfig
from data.data import get_dataset, ReplayBuffer
from dynamics.mlp_ensemble_dynamics import EnsembleDynamics
from dynamics.logger import Logger, make_log_dirs, snapshot_src
from policy.sac import SACPolicy
from architecture.mlp_ensemble import sample_actions
from dynamics.utils import Params

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


"""
python scripts/training/train_mlp.py env_name=hopper-expert-v2 log_root_dir=log_model_new dynamics_max_epochs_since_update=50 dynamics_max_epochs=300 seed=11 train_model_only=true exp_name=mlp_huge_ft_seed11 mlp_huge=true pt_path=pretrained_models/mlp_pt/mlp_dynamics990000.pkl
python scripts/training/train_mlp.py env_name=hopper-expert-v2 log_root_dir=log_model_new dynamics_max_epochs_since_update=50 dynamics_max_epochs=300 seed=11 train_model_only=true exp_name=mlp_huge_scratch_seed11 mlp_huge=true
"""

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

conf_dict = OmegaConf.from_cli()
config = MOPOConfig(**conf_dict)
config.dynamics_hidden_dims = tuple(config.dynamics_hidden_dims)
config.dynamics_weight_decay = tuple(config.dynamics_weight_decay)


@partial(jax.jit, static_argnames=(
    "rollout_length", 
    "actor_apply_fn",
    "dynamics_apply_fn", 
    "dynamisc_terminal_fn",
    "scaler_transform",
    "random_elite_idxs",
    "penalty_coef",
    "uncertainty_mode",
))
def _rollout_kernel(
    rng: jax.random.PRNGKey,
    init_obss: jnp.ndarray,
    rollout_length: int,
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    dynamics_apply_fn: Callable[..., Any],
    dynamics_params: Params,
    dynamisc_terminal_fn: Callable[..., Any],
    scaler_transform: Callable[..., Any],
    random_elite_idxs: Callable[..., Any],
    penalty_coef: float,
    uncertainty_mode: str,
) -> Tuple[Dict[str, jnp.ndarray], Dict]:

    num_transitions = 0
    rewards_arr = jnp.array([])
    raw_rewards_arr = jnp.array([])
    penalty_arr = jnp.array([])
    rollout_transitions = collections.defaultdict(list)

    # rollout
    observations = init_obss
    masks = None
    for _ in range(rollout_length):
        rng, actions = sample_actions(
            rng, actor_apply_fn, actor_params, observations, temperature=1.0)
        actions = jnp.clip(actions, -1, 1)
        actions = jax.lax.stop_gradient(actions)
        
        def step(
            rng: jax.random.PRNGKey,
            obs: jnp.ndarray,
            action: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
            obs_act = jnp.concatenate([obs, action], axis=-1)
            obs_act = scaler_transform(obs_act)
            mean, logvar, _, _ = dynamics_apply_fn(dynamics_params, obs_act)
            mean, logvar = jax.lax.stop_gradient(mean), jax.lax.stop_gradient(logvar)
            mean = mean.at[..., :-1].add(obs)
            std = jnp.sqrt(jnp.exp(logvar))

            rng, key = jax.random.split(rng)
            ensemble_samples = (mean + jax.random.normal(key, mean.shape) * std)

            num_models, batch_size, _ = ensemble_samples.shape
            rng, model_idxs = random_elite_idxs(rng, batch_size)
            samples = ensemble_samples[model_idxs, jnp.arange(batch_size)]

            next_obs = samples[..., :-1]
            reward = samples[..., -1:]
            terminal = dynamisc_terminal_fn(obs, action, next_obs)
            info = {}
            info["raw_reward"] = reward

            if penalty_coef:
                if uncertainty_mode == "aleatoric":
                    penalty = jnp.amax(jnp.linalg.norm(std, axis=2), axis=0)
                elif uncertainty_mode == "pairwise-diff":
                    next_obses_mean = mean[..., :-1]
                    next_obs_mean = jnp.mean(next_obses_mean, axis=0)
                    diff = next_obses_mean - next_obs_mean
                    penalty = jnp.amax(jnp.linalg.norm(diff, axis=2), axis=0)
                elif uncertainty_mode == "ensemble_std":
                    next_obses_mean = mean[..., :-1]
                    penalty = jnp.sqrt(next_obses_mean.var(0).mean(1))
                else:
                    raise ValueError
                penalty = jnp.expand_dims(penalty, 1).astype(jnp.float32)
                assert penalty.shape == reward.shape
                reward = reward - penalty_coef * penalty
                info["penalty"] = penalty

            return rng, next_obs, reward, terminal, info
        
        rng, next_observations, rewards, terminals, info = step(rng, observations, actions)
        
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

        observations = next_observations

    for k, v in rollout_transitions.items():
        rollout_transitions[k] = jnp.concatenate(v, axis=0)

    rollout_transitions["rewards"] = rollout_transitions["rewards"].squeeze(-1)
    rollout_transitions["dones"] = rollout_transitions["dones"].squeeze(-1)

    return rng, rollout_transitions, {
        "reward_mean": rewards_arr.mean(),
        "raw_reward_mean": raw_rewards_arr.mean(),
        "raw_reward_max": raw_rewards_arr.max(),
        "penalty_mean": penalty_arr.mean(),
        "penalty_std": penalty_arr.std()
    }


class MOPOTrainer(object):

    def __init__(self, config: MOPOConfig, real_buffer: ReplayBuffer, fake_buffer: ReplayBuffer, logger: Logger, eval_env, obs_mean, obs_std):
        self._env_name = config.env_name
        self._epoch = config.epoch
        self._step_per_epoch = config.step_per_epoch
        self._batch_size = config.batch_size
        self._real_ratio = config.real_ratio
        self._eval_episodes = config.eval_episodes
        self._rollout_freq = config.rollout_freq
        self._rollout_batch_size = config.rollout_batch_size
        self._rollout_length = config.rollout_length
        self._max_parallel_rollouts = None
        self._dynamics_max_epochs = config.dynamics_max_epochs
        self._dynamics_max_epochs_since_update = config.dynamics_max_epochs_since_update
        self._dynamics_update_freq = 0
        self._n_jitted_updates = config.n_jitted_updates
        self.config = config

        if self._n_jitted_updates is not None:
            assert self._rollout_freq % self._n_jitted_updates == 0
            assert self._step_per_epoch % self._n_jitted_updates == 0
            assert self._dynamics_update_freq % self._n_jitted_updates == 0

        self.rng = jax.random.PRNGKey(config.seed)
        self.policy = SACPolicy(config) #policy should have same dims as before

        if config.mlp_huge:
            config.dynamics_hidden_dims = (640, 640, 640, 640)
            config.obs_dim, config.act_dim = 90, 30
            print('Using Huge MLP Dynamics Model')

        if config.pt_path is not None:
            config.dynamics_hidden_dims = (640, 640, 640, 640)
            config.obs_dim, config.act_dim = 90, 30
            print('Finetuning Dynamics Model')

        self.model = EnsembleDynamics(config)

        if config.pt_path is not None:
            print('Loading Pretrained Dynamics Model')
            self.model.load(config.pt_path)

        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.logger = logger
        self.eval_env = eval_env
        self.obs_mean = obs_mean
        self.obs_std = obs_std

    def jitted_rollout(
        self,
        rng: jax.random.PRNGKey,
        init_obss: jnp.ndarray,
        rollout_length: int,
    ) -> Tuple[Dict[str, jnp.ndarray], Dict]:
        rng, rollout_transitions, rollout_info = _rollout_kernel(
            rng, init_obss, rollout_length, 
            self.policy.actor.apply_fn, self.policy.actor.params, 
            self.model.dynamics.apply_fn, self.model.dynamics.params, self.model.terminal_fn,
            self.model.scaler.transform, self.model.random_elite_idxs,
            self.model.penalty_coef, self.model.uncertainty_mode,
        )
        rollout_masks = ~rollout_transitions["masks"].flatten()
        rollout_info["num_transitions"] = rollout_masks.sum()
        del rollout_transitions["masks"]
        rollout_transitions = jax.tree.map(lambda x: np.asarray(x[rollout_masks]), rollout_transitions)
        return rng, rollout_transitions, rollout_info

    def rollout(
        self,
        rng: jax.random.PRNGKey,
        init_obss: jnp.ndarray,
        rollout_length: int,
    ) -> Tuple[Dict[str, jnp.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = jnp.array([])
        raw_rewards_arr = jnp.array([])
        penalty_arr = jnp.array([])
        rollout_transitions = collections.defaultdict(list)

        # rollout
        observations = init_obss
        masks = None
        for _ in range(rollout_length):
            rng, actions = self.policy.select_action(rng, observations)

            actions = jax.lax.stop_gradient(actions)
            rng, next_observations, rewards, terminals, info = self.model.step(rng, observations, actions)
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

            observations = next_observations

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
            "penalty_mean": penalty_arr.mean()
        }

    def learn(self, real_batch: Batch, fake_batch: Batch) -> Dict[str, float]:
        mix_batch = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), real_batch, fake_batch)
        return self.policy.update(mix_batch)

    def learn_n_times(self, real_batch: Batch, fake_batch: Batch) -> Dict[str, float]:
        real_batch = jax.tree.map(lambda x: x.reshape(self._n_jitted_updates, -1, *x.shape[1:]), real_batch)
        fake_batch = jax.tree.map(lambda x: x.reshape(self._n_jitted_updates, -1, *x.shape[1:]), fake_batch)
        mix_batch = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), real_batch, fake_batch)
        mix_batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), mix_batch)
        return self.policy.update_n_times(mix_batch, self._n_jitted_updates, self._batch_size)

    def train_model(self):
        self.model.train(self.real_buffer.sample_all(), self.logger, max_epochs=self._dynamics_max_epochs,
                         max_epochs_since_update=self._dynamics_max_epochs_since_update, epoch_start=self.config.epoch_start)

    def train_policy(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = collections.deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):
            start_time = time.time()
            pbar = tqdm.trange(self._step_per_epoch // (self._n_jitted_updates or 1), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar:
                if num_timesteps % self._rollout_freq == 0:
                    rollout_start_time = time.time()
                    init_obss = self.real_buffer.sample(self._rollout_batch_size).observations
                    self.rng, rollout_transitions, rollout_info = self.jitted_rollout(
                        self.rng, init_obss, self._rollout_length)

                    self.fake_buffer.insert_batch(**rollout_transitions)
                    self.logger.log(
                        "num rollout transitions: {}, reward mean: {:.4f}, time: {}".format(
                            rollout_info["num_transitions"], rollout_info["reward_mean"], time.time() -
                            rollout_start_time
                        )
                    )
                    for _key, _value in rollout_info.items():
                        self.logger.logkv_mean("rollout_info/" + _key, _value)

                if self._n_jitted_updates:
                    real_sample_size = int(self._batch_size * self._real_ratio)
                    fake_sample_size = self._batch_size - real_sample_size
                    real_batch = self.real_buffer.sample(batch_size=real_sample_size * self._n_jitted_updates)
                    fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size * self._n_jitted_updates)
                    loss = self.learn_n_times(real_batch, fake_batch)
                else:
                    real_sample_size = int(self._batch_size * self._real_ratio)
                    fake_sample_size = self._batch_size - real_sample_size
                    real_batch = self.real_buffer.sample(batch_size=real_sample_size)
                    fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                    loss = self.learn(real_batch, fake_batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean("policy_train/" + k, v)

                num_timesteps += self._n_jitted_updates or 1

                # update the dynamics if necessary
                if 0 < self._dynamics_update_freq and num_timesteps % self._dynamics_update_freq == 0:
                    assert False
                    dynamics_update_info = self.policy.update_dynamics(self.real_buffer)
                    for k, v in dynamics_update_info.items():
                        self.logger.logkv_mean(k, v)

            # evaluate current policy
            eval_info = self.evaluate()
            ep_reward_mean, ep_reward_std = np.mean(
                eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(
                eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])

            if 'maniskill' in self._env_name:
                ep_success_rate = np.mean(eval_info["eval/episode_success"])
                last_10_performance.append(ep_reward_mean)
                self.logger.logkv("eval/episode_reward", ep_reward_mean)
                self.logger.logkv("eval/episode_reward_std", ep_reward_std)
                self.logger.logkv("eval/episode_success_rate", ep_success_rate)
            else:
                norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
                norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
                last_10_performance.append(norm_ep_rew_mean)
                self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
                self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.logkv("time/policy_epoch", time.time() - start_time)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])

            # save checkpoint
            self.policy.save(os.path.join(self.logger.checkpoint_dir, "policy.pkl"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        self.policy.save(os.path.join(self.logger.checkpoint_dir, "policy.pkl"))
        self.model.save(self.logger.model_dir)
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def evaluate(self):
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        rng = jax.random.PRNGKey(0)
        while num_episodes < self._eval_episodes:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-5)
            rng, action = self.policy.select_action(rng, obs.reshape(1, -1), temperature=0.0)
            next_obs, reward, terminal, info = self.eval_env.step(np.array(action.flatten()))
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                if 'maniskill' in self._env_name:
                    eval_ep_info_buffer[-1]['episode_success'] = info['success']
                num_episodes += 1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()

        results = {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
        if 'maniskill' in self._env_name:
            results["eval/episode_success"] = [ep_info["episode_success"] for ep_info in eval_ep_info_buffer]
        return results


if __name__ == "__main__":
    log_dirs = make_log_dirs(config.env_name, config.algo, config.seed, vars(config),
                             record_params=["penalty_coef", "rollout_length"], root_dir=config.log_root_dir, exp_prefix=config.exp_name)

    with open(os.path.join(log_dirs, "cmd.sh"), "w") as f:
        config.cmd = "python " + " ".join(sys.argv)
        f.write("python " + " ".join(sys.argv))
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

    # seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    env.seed(config.seed)

    dataset, obs_mean, obs_std = get_dataset(
        env, config,
        dataset=pickle.load(open(config.dataset_path, 'rb')) if config.dataset_path is not None else None
    )
    config.act_dim = dataset.actions.shape[-1]
    config.obs_dim = dataset.observations.shape[-1]


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

    config.target_entropy = config.target_entropy if config.target_entropy else -config.act_dim
    config.actor_cosine_decay_steps = config.step_per_epoch * config.epoch
    trainer = MOPOTrainer(
        config,
        real_buffer,
        fake_buffer,
        logger,
        env,
        obs_mean,
        obs_std,
    )

    if config.load_dynamics_path:
        trainer.model.load(config.load_dynamics_path)
    else:
        trainer.train_model()

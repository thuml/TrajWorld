from typing import Dict, NamedTuple, Union, Optional

import d4rl
import gym
import jax
import numpy as np

from dynamics.config import MOPOConfig


def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    has_next_obs = True if 'next_observations' in dataset.keys() else False

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True
        # print([i for i in range(len(dataset['timeouts'])) if dataset['timeouts'][i] == 1])
        # !timeouts may have wrongs in the dataset

    episode_step = 0
    for i in range(N - 1):
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i + 1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0
            if not has_next_obs:
                continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


class Batch(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


def get_dataset(
    env: gym.Env, config: MOPOConfig, clip_to_eps: bool = True, eps: float = 1e-5, dataset: Optional[Dict[str, np.ndarray]] = None,
) -> Batch:
    if dataset is None:
        dataset = qlearning_dataset(env)

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    dataset = Batch(
        observations=np.array(dataset["observations"], dtype=np.float32),
        actions=np.array(dataset["actions"], dtype=np.float32),
        rewards=np.array(dataset["rewards"], dtype=np.float32),
        dones=np.array(dataset["terminals"], dtype=np.float32),
        next_observations=np.array(dataset["next_observations"], dtype=np.float32),
    )
    # shuffle data and select the first data_size samples
    data_size = min(config.data_size, len(dataset.observations))
    perm = np.random.permutation(len(dataset.observations))
    dataset = jax.tree.map(lambda x: x[perm], dataset)
    assert len(dataset.observations) >= data_size
    dataset = jax.tree.map(lambda x: x[:data_size], dataset)
    # normalize states
    if config.normalize_state:
        obs_mean = dataset.observations.mean(0)
        obs_std = dataset.observations.std(0)
        print("obs_mean", obs_mean)
        print("obs_std", obs_std)
        dataset = dataset._replace(
            observations=(dataset.observations - obs_mean) / (obs_std + 1e-5),
            next_observations=(dataset.next_observations - obs_mean) / (obs_std + 1e-5),
        )
    else:
        obs_mean = np.zeros_like(dataset.observations.mean(0))
        obs_std = np.ones_like(dataset.observations.std(0))
    return dataset, obs_mean, obs_std


class ReplayBuffer:

    def __init__(self, observation_dim: int,
                 action_dim: int, capacity: int):

        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, ), dtype=np.float32)
        self.dones = np.empty((capacity, ), dtype=np.float32)
        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Batch):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)
        assert self.capacity >= dataset_size, 'Dataset cannot be larger than the replay buffer capacity.'

        self.observations[:dataset_size] = dataset.observations
        self.actions[:dataset_size] = dataset.actions
        self.rewards[:dataset_size] = dataset.rewards
        self.dones[:dataset_size] = dataset.dones
        self.next_observations[:dataset_size] = dataset.next_observations

        self.insert_index = dataset_size
        self.size = dataset_size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     dones=self.dones[indx],
                     next_observations=self.next_observations[indx])

    def sample_all(self) -> Dict[str, np.ndarray]:
        return Batch(observations=self.observations[:self.size],
                     actions=self.actions[:self.size],
                     rewards=self.rewards[:self.size],
                     dones=self.dones[:self.size],
                     next_observations=self.next_observations[:self.size])

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, done: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.dones[self.insert_index] = done
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def insert_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_observations: np.ndarray
    ):
        batch_size = len(observations)
        insert_indx = np.arange(self.insert_index, self.insert_index + batch_size) % self.capacity

        self.observations[insert_indx] = observations
        self.actions[insert_indx] = actions
        self.rewards[insert_indx] = rewards
        self.dones[insert_indx] = dones
        self.next_observations[insert_indx] = next_observations

        self.insert_index = (self.insert_index + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

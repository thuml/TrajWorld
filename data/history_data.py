from typing import Dict, NamedTuple, Union, Tuple, Optional

import d4rl
import gym
import jax
import numpy as np
from tqdm import trange

from dynamics.config import MOPOConfig
from data.data import Batch

def symlog(x):
    return np.sign(x) * np.log(np.abs(x) + 1)


def history_augmented_qlearning_dataset(env, history_len, dataset=None, terminate_on_end=False, reward_clip_min: int = None, reward_clip_max: int = None, **kwargs):
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
        
    if reward_clip_min is not None or reward_clip_max is not None:
        print(f"Clipping rewards to [{reward_clip_min}, {reward_clip_max}]")
        print(f"Original rewards: {dataset['rewards'].min()}, {dataset['rewards'].max()}")
        dataset['rewards'] = np.clip(dataset['rewards'], reward_clip_min, reward_clip_max)
        print(f"Clipped rewards: {dataset['rewards'].min()}, {dataset['rewards'].max()}")

    has_next_obs = True if 'next_observations' in dataset.keys() else False

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    hist_ = []
    hist_mask_ = []

    obs_dim = dataset['observations'].shape[-1]
    act_dim = dataset['actions'].shape[-1]
    obs_hist = [np.zeros(obs_dim)] * history_len  # [T, d]
    rew_hist = [0.] * history_len  # [T]
    act_hist = [np.zeros(act_dim)] * history_len  # [T, d]
    hist_mask = [0.] * history_len  # [T]

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    last_reward = 0
    for i in trange(N - 1, desc='Loading history dataset'):
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
            if has_next_obs:
                obs_hist.pop(0)
                rew_hist.pop(0)
                act_hist.pop(0)
                hist_mask.pop(0)
                obs_hist.append(obs)
                rew_hist.append(last_reward)
                act_hist.append(action)
                hist_mask.append(1)
                if np.sum(hist_mask) > 1:
                    hist_.append(np.concatenate([np.array(obs_hist), np.array(rew_hist)[:, None], np.array(act_hist)], axis=-1))
                    hist_mask_.append(np.array(hist_mask))
            
            episode_step, last_reward = 0, 0
            obs_hist = [np.zeros(obs_dim)] * history_len  # [T, d]
            rew_hist = [0.] * history_len  # [T]
            act_hist = [np.zeros(act_dim)] * history_len  # [T, d]
            hist_mask = [0.] * history_len  # [T]
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

        obs_hist.pop(0)
        rew_hist.pop(0)
        act_hist.pop(0)
        hist_mask.pop(0)
        obs_hist.append(obs)
        rew_hist.append(last_reward)
        act_hist.append(action)
        hist_mask.append(1)
        if np.sum(hist_mask) > 1:
            hist_.append(np.concatenate([np.array(obs_hist), np.array(rew_hist)[:, None], np.array(act_hist)], axis=-1))
            hist_mask_.append(np.array(hist_mask))

        if done_bool or final_timestep or i==N-2:
            if has_next_obs:
                obs_hist.pop(0)
                rew_hist.pop(0)
                act_hist.pop(0)
                hist_mask.pop(0)
                obs_hist.append(new_obs)
                rew_hist.append(reward)
                act_hist.append(np.zeros_like(action))
                hist_mask.append(1)
                if np.sum(hist_mask) > 1:
                    hist_.append(np.concatenate([np.array(obs_hist), np.array(rew_hist)[:, None], np.array(act_hist)], axis=-1))
                    hist_mask_.append(np.array(hist_mask))
            
            last_reward = 0.
            obs_hist = [np.zeros(obs_dim)] * history_len  # [T, d]
            rew_hist = [0.] * history_len  # [T]
            act_hist = [np.zeros(act_dim)] * history_len  # [T, d]
            hist_mask = [0.] * history_len  # [T]
        else:
            last_reward = reward

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'histories': np.array(hist_),
        'history_masks': np.array(hist_mask_),
    }



def history_augmented_qlearning_dataset_legacy(env, history_len, dataset=None, terminate_on_end=False, reward_clip_min: int = None, reward_clip_max: int = None, **kwargs):
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
        
    if reward_clip_min is not None or reward_clip_max is not None:
        print(f"Clipping rewards to [{reward_clip_min}, {reward_clip_max}]")
        print(f"Original rewards: {dataset['rewards'].min()}, {dataset['rewards'].max()}")
        dataset['rewards'] = np.clip(dataset['rewards'], reward_clip_min, reward_clip_max)
        print(f"Clipped rewards: {dataset['rewards'].min()}, {dataset['rewards'].max()}")

    has_next_obs = True if 'next_observations' in dataset.keys() else False

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    hist_ = []
    hist_mask_ = []

    obs_dim = dataset['observations'].shape[-1]
    act_dim = dataset['actions'].shape[-1]
    obs_hist = [np.zeros(obs_dim)] * history_len  # [T, d]
    rew_hist = [0.] * history_len  # [T]
    act_hist = [np.zeros(act_dim)] * history_len  # [T, d]
    hist_mask = [0.] * history_len  # [T]

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    last_reward = 0
    for i in trange(N - 1, desc='Loading history dataset'):
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
            if has_next_obs:
                obs_hist.pop(0)
                rew_hist.pop(0)
                act_hist.pop(0)
                hist_mask.pop(0)
                obs_hist.append(new_obs)
                rew_hist.append(reward)
                act_hist.append(np.zeros_like(action))
                hist_mask.append(1)
                hist_.append(np.concatenate([np.array(obs_hist), np.array(rew_hist)[:, None], np.array(act_hist)], axis=-1))
                hist_mask_.append(np.array(hist_mask))
            
            episode_step, last_reward = 0, 0
            obs_hist = [np.zeros(obs_dim)] * history_len  # [T, d]
            rew_hist = [0.] * history_len  # [T]
            act_hist = [np.zeros(act_dim)] * history_len  # [T, d]
            hist_mask = [0.] * history_len  # [T]
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

        obs_hist.pop(0)
        rew_hist.pop(0)
        act_hist.pop(0)
        hist_mask.pop(0)
        obs_hist.append(obs)
        rew_hist.append(last_reward)
        act_hist.append(action)
        hist_mask.append(1)
        hist_.append(np.concatenate([np.array(obs_hist), np.array(rew_hist)[:, None], np.array(act_hist)], axis=-1))
        hist_mask_.append(np.array(hist_mask))

        if done_bool or final_timestep:
            if has_next_obs:
                obs_hist.pop(0)
                rew_hist.pop(0)
                act_hist.pop(0)
                hist_mask.pop(0)
                obs_hist.append(new_obs)
                rew_hist.append(reward)
                act_hist.append(np.zeros_like(action))
                hist_mask.append(1)
                hist_.append(np.concatenate([np.array(obs_hist), np.array(rew_hist)[:, None], np.array(act_hist)], axis=-1))
                hist_mask_.append(np.array(hist_mask))
            
            last_reward = 0.
            obs_hist = [np.zeros(obs_dim)] * history_len  # [T, d]
            rew_hist = [0.] * history_len  # [T]
            act_hist = [np.zeros(act_dim)] * history_len  # [T, d]
            hist_mask = [0.] * history_len  # [T]
        else:
            last_reward = reward

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'histories': np.array(hist_),
        'history_masks': np.array(hist_mask_),
    }

class HistoryBatch(NamedTuple):
    histories: np.ndarray
    history_masks: np.ndarray


def get_history_dataset(
    env: gym.Env, config: MOPOConfig, clip_to_eps: bool = True, eps: float = 1e-5, dataset: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[HistoryBatch, Batch, np.ndarray, np.ndarray]:
    dataset = history_augmented_qlearning_dataset(env, history_len=config.history_length, dataset=dataset, 
                                                  reward_clip_min=config.reward_clip_min, reward_clip_max=config.reward_clip_max)

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    step_dataset = Batch(
        observations=np.array(dataset["observations"], dtype=np.float32),
        actions=np.array(dataset["actions"], dtype=np.float32),
        rewards=np.array(dataset["rewards"], dtype=np.float32),
        dones=np.array(dataset["terminals"], dtype=np.float32),
        next_observations=np.array(dataset["next_observations"], dtype=np.float32),
    )
    hist_dataset = HistoryBatch(
        histories=np.array(dataset["histories"], dtype=np.float32),
        history_masks=np.array(dataset["history_masks"], dtype=np.float32),
    )
    print("Dataset size: ", len(step_dataset.observations))
    print("History dataset size: ", len(hist_dataset.histories))
    max_values = np.concatenate([
        np.max(np.concatenate([step_dataset.observations, step_dataset.next_observations], axis=0), axis=0),
        np.max(step_dataset.rewards, axis=0, keepdims=True),
        np.max(step_dataset.actions, axis=0),
    ])
    min_values = np.concatenate([
        np.min(np.concatenate([step_dataset.observations, step_dataset.next_observations], axis=0), axis=0),
        np.min(step_dataset.rewards, axis=0, keepdims=True),
        np.min(step_dataset.actions, axis=0),
    ])
    
    # normalize states
    if config.normalize_state:
        obs_mean = step_dataset.observations.mean(0)
        obs_std = step_dataset.observations.std(0)
        print("obs_mean", obs_mean)
        print("obs_std", obs_std)
        step_dataset = step_dataset._replace(
            observations=(step_dataset.observations - obs_mean) / (obs_std + 1e-5),
            next_observations=(step_dataset.next_observations - obs_mean) / (obs_std + 1e-5),
        )
        if config.normalize_reward:
            rew_mean = step_dataset.rewards.mean()
            rew_std = step_dataset.rewards.std()
            print("rew_mean", rew_mean)
            print("rew_std", rew_std)
            step_dataset = step_dataset._replace(
                rewards=(step_dataset.rewards - rew_mean) / (rew_std + 1e-5),
            )
        else:
            rew_mean = 0.0
            rew_std = 1.0
    else:
        obs_mean = np.zeros_like(step_dataset.observations.mean(0))
        obs_std = np.ones_like(step_dataset.observations.std(0))

    if config.normalize_history:
        hist_mean = np.concatenate([obs_mean, [rew_mean], np.zeros_like(step_dataset.actions.mean(0))])
        hist_std = np.concatenate([obs_std, [rew_std], np.ones_like(step_dataset.actions.std(0))])
        hist_dataset = hist_dataset._replace(
            histories=(hist_dataset.histories - hist_mean) / (hist_std + 1e-5),
        )
        max_values = (max_values - hist_mean) / (hist_std + 1e-5)
        min_values = (min_values - hist_mean) / (hist_std + 1e-5)
    
    return hist_dataset, step_dataset, max_values, min_values, obs_mean, obs_std


class HistoryReplayBuffer:

    def __init__(self, observation_dim: int,
                 action_dim: int, capacity: int,
                 history_len: int):

        self.histories = np.empty((capacity, history_len, observation_dim + 1 + action_dim), dtype=np.float32)
        self.history_masks = np.empty((capacity, history_len), dtype=np.float32)
        self.size = 0

        self.insert_index = 0
        self.capacity = capacity
        self.history_len = history_len

    def initialize_with_dataset(self, dataset: HistoryBatch):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.histories)
        assert self.capacity >= dataset_size, 'Dataset cannot be larger than the replay buffer capacity.'

        self.histories[:dataset_size] = dataset.histories
        self.history_masks[:dataset_size] = dataset.history_masks

        self.insert_index = dataset_size
        self.size = dataset_size

    def sample(self, batch_size: int) -> HistoryBatch:
        indx = np.random.randint(self.size, size=batch_size)
        return HistoryBatch(histories=self.histories[indx],
                            history_masks=self.history_masks[indx])

    def sample_all(self) -> Dict[str, np.ndarray]:
        return HistoryBatch(histories=self.histories[:self.size],
                            history_masks=self.history_masks[:self.size])

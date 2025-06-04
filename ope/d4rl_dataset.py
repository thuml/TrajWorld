# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for loading data."""
import numpy as np
# import torch
import jax
import jax.numpy as jnp
from tqdm import trange


class D4rlDataset(object):
    """Dataset class for policy evaluation."""

    def __init__(self,
                 d4rl_env,
                 normalize_states=False,
                 normalize_rewards=False,
                 eps=1e-5,
                 noise_scale=0.0,
                 bootstrap=False,
                 seed=0):
        """Processes data from D4RL environment.

        Args:
          d4rl_env: gym.Env corresponding to D4RL environment.
          normalize_states: whether to normalize the states.
          normalize_rewards: whether to normalize the rewards.
          eps: Epsilon used for normalization.
          noise_scale: Data augmentation noise scale.
          bootstrap: Whether to generated bootstrapped weights.
        """
        self.rng = jax.random.PRNGKey(seed)
        dataset = dict(
            trajectories=dict(
                states=[],
                actions=[],
                next_states=[],
                rewards=[],
                masks=[]))
        d4rl_dataset = d4rl_env.get_dataset()
        dataset_length = len(d4rl_dataset['actions'])
        new_trajectory = True
        bar = trange(dataset_length, desc='load d4rl dataset')
        for idx in bar:
            if new_trajectory:
                trajectory = dict(
                    states=[], actions=[], next_states=[], rewards=[], masks=[])

            trajectory['states'].append(d4rl_dataset['observations'][idx])
            trajectory['actions'].append(d4rl_dataset['actions'][idx])
            trajectory['rewards'].append(d4rl_dataset['rewards'][idx])
            trajectory['masks'].append(1.0 - d4rl_dataset['terminals'][idx])
            if not new_trajectory:
                trajectory['next_states'].append(d4rl_dataset['observations'][idx])

            end_trajectory = (d4rl_dataset['terminals'][idx] or
                              d4rl_dataset['timeouts'][idx])
            if end_trajectory:
                trajectory['next_states'].append(d4rl_dataset['observations'][idx])
                if d4rl_dataset['timeouts'][idx] and not d4rl_dataset['terminals'][idx]:
                    for key in trajectory:
                        del trajectory[key][-1]
                if trajectory['actions']:
                    for k, v in trajectory.items():
                        assert len(v) == len(trajectory['actions'])
                        dataset['trajectories'][k].append(np.array(v, dtype=np.float32))
                    bar.set_postfix({
                        'traj_id': len(dataset['trajectories']['actions']),
                        'traj_len': len(trajectory['actions'])})

            new_trajectory = end_trajectory

        dataset['trajectories']['steps'] = [
            np.arange(len(state_trajectory))
            for state_trajectory in dataset['trajectories']['states']
        ]

        dataset['initial_states'] = np.stack([
            state_trajectory[0]
            for state_trajectory in dataset['trajectories']['states']
        ])
        dataset['initial_actions'] = np.stack([
            action_trajectory[0]
            for action_trajectory in dataset['trajectories']['actions']
        ])

        num_trajectories = len(dataset['trajectories']['states'])
        if bootstrap:
            dataset['initial_weights'] = np.random.multinomial(
                num_trajectories, [1.0 / num_trajectories] * num_trajectories,
                1).astype(np.float32)[0]
        else:
            dataset['initial_weights'] = np.ones(num_trajectories, dtype=np.float32)

        dataset['trajectories']['weights'] = []
        for i in range(len(dataset['trajectories']['masks'])):
            dataset['trajectories']['weights'].append(
                np.ones_like(dataset['trajectories']['masks'][i]) *
                dataset['initial_weights'][i])

        for k, v in dataset['trajectories'].items():
            if 'initial' not in k:
                dataset[k] = np.concatenate(dataset['trajectories'][k], axis=0)

        self.states = dataset['states']
        self.actions = dataset['actions']
        self.next_states = dataset['next_states']
        self.masks = dataset['masks']
        self.weights = dataset['weights']
        self.rewards = dataset['rewards']
        self.steps = dataset['steps']

        self.size = dataset['states'].shape[0]

        self.initial_states = dataset['initial_states']
        self.initial_weights = dataset['initial_weights']
        self.initial_actions = dataset['initial_actions']

        self.eps = eps
        self.model_filename = None

        if normalize_states:
            self.state_mean = np.mean(self.states, axis=0)
            self.state_std = np.maximum(np.std(self.states, axis=0), self.eps)

            self.initial_states = self.normalize_states(self.initial_states)
            self.states = self.normalize_states(self.states)
            self.next_states = self.normalize_states(self.next_states)
        else:
            self.state_mean = 0.0
            self.state_std = 1.0

        if normalize_rewards:
            self.reward_mean = np.mean(self.rewards)
            if np.min(self.masks) == 0.0:
                self.reward_mean = 0.0
            self.reward_std = max(np.std(self.rewards), self.eps)

            self.rewards = self.normalize_rewards(self.rewards)
        else:
            self.reward_mean = 0.0
            self.reward_std = 1.0

    def normalize_states(self, states):
        return (states - self.state_mean) / self.state_std

    def unnormalize_states(self, states):
        if isinstance(states, np.ndarray) or isinstance(states, jnp.ndarray):
            return states * self.state_std + self.state_mean
        else:
            raise NotImplementedError

    def normalize_rewards(self, rewards):
        return (rewards - self.reward_mean) / self.reward_std

    def unnormalize_rewards(self, rewards):
        return rewards * self.reward_std + self.reward_mean

    def get_batch(self, ind):
        return (
            jnp.array(self.states[ind]),
            jnp.array(self.actions[ind]),
            jnp.array(self.next_states[ind]),
            jnp.array(self.rewards[ind]).reshape(-1, 1),
            jnp.array(self.masks[ind]).reshape(-1, 1),
            jnp.array(self.weights[ind]).reshape(-1, 1),
            jnp.array(self.steps[ind])
        )

    def sample(self, batch_size):
        self.rng, key = jax.random.split(self.rng)
        ind = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.size)
        return self.get_batch(ind)

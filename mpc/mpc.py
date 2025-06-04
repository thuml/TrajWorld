import csv
import re

import math
from jax import random, lax
import jax
import jax.numpy as jnp
import gym
import argparse
import os
import d4rl.gym_mujoco
import d4rl
import json
import time
from tqdm import trange
from omegaconf import OmegaConf
from typing import Dict, Tuple, Any, Callable, Optional, Sequence, Union, List
from functools import partial
import flax
from copy import deepcopy
import numpy as np

from architecture.trm_tdm import TDMTransformer
from ope.logger import Logger
from dynamics.config import MOPOConfig
from dynamics.tdm_dynamics import TDM_Dynamics

from ope import dope_policies, utils
from ope.d4rl_dataset import D4rlDataset

from dynamics.trajworld_dynamics import TrajWorldDynamics, transform_from_probs, transform_to_probs
from dynamics.mlp_ensemble_dynamics import EnsembleDynamics
from architecture.trm_traj import TrajWorldTransformer

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
Params = flax.core.FrozenDict[str, Any]

"""
python mpc/mpc.py --algo trajworld --env walker2d-medium-replay-v2 --group 5 --clear_kv_cache_every 10 --trm_lookback_window 10 --action_proposal_id 3 --std 0.1
"""

def get_list_dirs(group):
    if group == 5:
        return [
            "/home/NAS/rl_data/log_final/walker2d-medium-replay-v2/mergeall_ft_baseline_smalllr_6_layer_seed83&seed_83&timestamp_25-0110-033120/model/trm_dynamics150.pkl"
        ]
    else:
        raise NotImplementedError

def generate_normal_noise(key, shape, std=2.0):
    noise = jax.random.normal(key, shape) * std
    return noise

def estimate_returns_with_ensemble(
        rng,
        model: EnsembleDynamics, initial_states, get_target_actions, discount,
        min_reward, max_reward, min_state, max_state, clip=True, horizon=2000,
        mlp_huge=False
):
    returns = 0
    masks = jnp.ones((initial_states.shape[0], 1))
    states = initial_states

    noise = generate_normal_noise(rng, (initial_states.shape[0], horizon, action_dim), std=args.std)
    init_action = get_target_actions(states)

    for i in range(horizon):
        actions = get_target_actions(states)
        actions = actions + noise[:, i] # added here
        rng, next_states, rewards, terminal, info = model.step(rng, states, actions, mlp_huge)
        if clip:
            rewards = jnp.clip(rewards, min_reward, max_reward)
        returns += (discount ** i) * masks * rewards

        masks = masks * (1 - terminal)
        states = next_states
        if min_state is not None and max_state is not None:
            states = jnp.clip(states, min_state, max_state)
        if jnp.all(masks == 0):
            break

    actions_idx = jnp.argmax(returns)
    action = init_action[actions_idx] + noise[actions_idx, 0]
    return rng, action

def estimate_returns_with_ensemble_us(
        rng,
        model: EnsembleDynamics, initial_states, discount,
        min_reward, max_reward, min_state, max_state, clip=True, horizon=2000,
        mlp_huge=False, actor_weights=None,
):
    returns = 0
    masks = jnp.ones((initial_states.shape[0], 1))
    states = initial_states

    if actor_weights is not None: # action proposal
        actor_weights = deepcopy(actor_weights)
        actor_nonlinearity = actor_weights.pop('nonlinearity')
        actor_apply_fn = dope_policies.get_jax_policy_actions
        rng, actions = actor_apply_fn(rng, actor_weights, actor_nonlinearity, states)
        rng, subkey = jax.random.split(rng)
        actions = actions[:, None, :]
        actions = actions + generate_normal_noise(rng, (initial_states.shape[0], horizon, action_dim), std=args.std)
        actions = jnp.clip(actions, -1.0, 1.0)
    else:
        rng, subkey = jax.random.split(rng)
        actions = generate_normal_noise(rng, (initial_states.shape[0], horizon, action_dim), std=math.sqrt(2.0))
        actions = jnp.clip(actions, -1.0, 1.0)

    for i in range(horizon):
        action = actions[:, i]
        rng, next_states, rewards, terminal, info = model.step(rng, states, action, mlp_huge)
        if clip:
            rewards = jnp.clip(rewards, min_reward, max_reward)
        returns += masks * rewards

        masks = masks * (1 - terminal)
        states = next_states
        if min_state is not None and max_state is not None:
            states = jnp.clip(states, min_state, max_state)
        if jnp.all(masks == 0):
            break

    actions_idx = jnp.argmax(returns)
    action = actions[actions_idx, 0]
    return action


@partial(jax.jit, static_argnames=(
        "obs_dim",
        "act_dim",
        "trm_lookback_window",
        "rollout_length",
        "clip",
        "dynamics_empty_cache",
        "dynamics_terminal_fn",
        "dynamics_apply_fn",
))
def _rollout_kernel_us(
        rng: jax.random.PRNGKey,
        obs_dim: int,
        act_dim: int,
        trm_lookback_window: int,
        init_histories: jnp.ndarray,
        init_history_masks: jnp.ndarray,
        rollout_length: int,
        clip: bool,
        min_reward: jnp.ndarray,
        max_reward: jnp.ndarray,
        min_state: jnp.ndarray,
        max_state: jnp.ndarray,
        actions: jnp.ndarray,
        dynamics_min_values: jnp.ndarray,
        dynamics_max_values: jnp.ndarray,
        dynamics_support: jnp.ndarray,
        dynamics_sigma: float,
        dynamics_obs_act_indicator: jnp.ndarray,
        dynamics_empty_cache: Callable[..., Any],
        dynamics_terminal_fn: Callable[..., Any],
        dynamics_apply_fn: Callable[..., Any],
        dynamics_params: Params,
) -> Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def step(
            rng: jax.random.PRNGKey,
            inputs: jnp.ndarray,
            padding_mask: jnp.ndarray,
            caches: Optional[List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]],
    ):
        """
        history: [B, T, M]
        padding_mask: [B, T]
        """
        obs = inputs[:, -1, :obs_dim]
        action = inputs[:, -1, obs_dim + 1:]

        inputs = jnp.clip(inputs, dynamics_min_values, dynamics_max_values)
        inputs = transform_to_probs(inputs, dynamics_support, dynamics_sigma)

        obs_act_indicator = jnp.tile(dynamics_obs_act_indicator, (inputs.shape[0], inputs.shape[1], 1))

        rng, key = jax.random.split(rng)

        pred, updated_caches = dynamics_apply_fn(dynamics_params, inputs, obs_act_indicator,
                                                 padding_mask, caches=caches, training=False,
                                                 rngs={'dropout': key},  # TODO: need fix?
                                                 method=TrajWorldTransformer.call_kv_cache,
                                                 )

        pred = pred[:, -1, :obs_dim + 1]
        pred_prob = jax.nn.softmax(pred)

        samples = transform_from_probs(pred_prob, dynamics_support[:obs_dim + 1])

        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = dynamics_terminal_fn(obs, action, next_obs)
        info = {}

        return rng, next_obs, reward, terminal, info, updated_caches

    histories, history_masks = init_histories[:, -trm_lookback_window:], init_history_masks[:, -trm_lookback_window:]
    all_dim = obs_dim + act_dim + 1
    batch_size = init_histories.shape[0]
    caches = dynamics_empty_cache(batch_size=batch_size * all_dim)

    all_rewards = []
    all_terminals = []
    for j in range(rollout_length):
        histories = histories.at[:, -1, obs_dim + 1:].set(actions[:, j])
        if j == 0:
            rng, next_states, rewards, terminal, info, caches = step(
                rng, histories[:, -trm_lookback_window:], history_masks[:, -trm_lookback_window:], caches=caches)
        else:
            rng, next_states, rewards, terminal, info, caches = step(
                rng, histories[:, -1:], history_masks[:, -1:], caches=caches)

        if clip:
            rewards = jnp.clip(rewards, min_reward, max_reward)

        if min_state is not None and max_state is not None:
            next_states = jnp.clip(next_states, min_state, max_state)

        next_column = jnp.concatenate([next_states, rewards, jnp.zeros((batch_size, act_dim))], axis=-1)
        histories = jnp.concatenate([histories, next_column[:, None]], axis=1)
        next_column = jnp.ones((batch_size, 1))
        history_masks = jnp.concatenate([history_masks, next_column], axis=1)

        all_rewards.append(rewards)
        all_terminals.append(terminal)

    return rng, histories, history_masks, jnp.concat(all_rewards, axis=-1), jnp.concat(all_terminals, axis=-1)

@partial(jax.jit, static_argnames=(
        "dynamics_apply_fn",
))
def step_tt(
        rng: jax.random.PRNGKey,
        inputs: jnp.ndarray,
        padding_mask: jnp.ndarray,
        dynamics_min_values: jnp.ndarray,
        dynamics_max_values: jnp.ndarray,
        dynamics_support: jnp.ndarray,
        dynamics_sigma: float,
        dynamics_obs_act_indicator: jnp.ndarray,
        dynamics_apply_fn: Callable[..., Any],
        dynamics_params: Params,
        caches: Optional[List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]],
):
    """
    history: [B, T, M]
    padding_mask: [B, T]
    """
    inputs = jnp.clip(inputs, dynamics_min_values, dynamics_max_values)
    inputs = transform_to_probs(inputs, dynamics_support, dynamics_sigma)
    obs_act_indicator = jnp.tile(dynamics_obs_act_indicator, (inputs.shape[0], inputs.shape[1], 1))

    rng, key = jax.random.split(rng)

    pred, updated_caches = dynamics_apply_fn(dynamics_params, inputs, obs_act_indicator,
                                             padding_mask, caches=caches, training=False,
                                             rngs={'dropout': key},  # TODO: need fix?
                                             method=TDMTransformer.call_kv_cache,
                                             )
    pred = pred[:, -1]
    pred_prob = jax.nn.softmax(pred)

    dynamics_support_out = jnp.concat([dynamics_support[1:], dynamics_support[:1]], axis=0)
    samples = transform_from_probs(pred_prob, dynamics_support_out)

    return rng, samples, updated_caches

def rollout_step_tt(rng, histories, history_masks, action_in, obs_dim, all_dim,
                 dynamics_terminal_fn, dynamics_min_values, dynamics_max_values,
                 dynamics_support, dynamics_sigma, dynamics_obs_act_indicator,
                 dynamics_apply_fn, dynamics_params, updated_caches):
    histories = histories.at[:, -1, obs_dim + 1:].set(action_in)

    for k in range(obs_dim + 1):
        if k == 0:
            rng, samples, updated_caches = step_tt(
                rng, histories, history_masks, dynamics_min_values, dynamics_max_values,
                dynamics_support, dynamics_sigma, dynamics_obs_act_indicator, dynamics_apply_fn, dynamics_params, updated_caches)
            pred = samples[:, -1:]
            padded_pred = jnp.concat([pred, jnp.zeros((samples.shape[0], all_dim - 1))], axis=-1)
            padded_pred = padded_pred[:, None, :]
            histories = jnp.concat([histories, padded_pred], axis=1)
            history_masks = jnp.concat([history_masks, jnp.ones((samples.shape[0], 1))], axis=1)
        elif k < obs_dim:
            rng, samples, _ = step_tt(
                rng, histories[:, -1:], history_masks[:, -1:], dynamics_min_values, dynamics_max_values,
                dynamics_support, dynamics_sigma, dynamics_obs_act_indicator, dynamics_apply_fn, dynamics_params, updated_caches)
            pred = samples[:, None, k - 1:k]
            histories = histories.at[:, -1:, k:k + 1].set(pred)
        else:
            # only reward is needed
            rng, samples, _ = step_tt(
                rng, histories[:, -1:], history_masks[:, -1:], dynamics_min_values, dynamics_max_values,
                dynamics_support, dynamics_sigma, dynamics_obs_act_indicator, dynamics_apply_fn, dynamics_params, updated_caches)
            rewards = samples[:, k - 1:k]
            histories = histories.at[:, -1, k:k + 1].set(rewards)

            next_obs = histories[:, -1, :obs_dim]
            obs = histories[:, -2, :obs_dim]
            action = histories[:, -2, obs_dim + 1:]
            terminal = dynamics_terminal_fn(obs, action, next_obs)
    return rng, next_obs, rewards, terminal, updated_caches


def _rollout_kernel_tt_us(
        rng: jax.random.PRNGKey,
        obs_dim: int,
        act_dim: int,
        trm_lookback_window: int,
        init_histories: jnp.ndarray,
        init_history_masks: jnp.ndarray,
        rollout_length: int,
        clip: bool,
        min_reward: jnp.ndarray,
        max_reward: jnp.ndarray,
        min_state: jnp.ndarray,
        max_state: jnp.ndarray,
        actions: jnp.ndarray,
        dynamics_min_values: jnp.ndarray,
        dynamics_max_values: jnp.ndarray,
        dynamics_support: jnp.ndarray,
        dynamics_sigma: float,
        dynamics_obs_act_indicator: jnp.ndarray,
        dynamics_empty_cache: Callable[..., Any],
        dynamics_terminal_fn: Callable[..., Any],
        dynamics_apply_fn: Callable[..., Any],
        dynamics_params: Params,
) -> Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:


    histories, history_masks = init_histories[:, -trm_lookback_window:
                               ], init_history_masks[:, -trm_lookback_window:]
    batch_size = init_histories.shape[0]

    all_dim = obs_dim + 1 + act_dim

    batch_size = histories.shape[0]

    updated_caches = dynamics_empty_cache(batch_size=batch_size)

    all_rewards = []
    all_terminals = []
    for j in range(rollout_length):
        if j == 0:
            rng, next_obs, rewards, terminal, updated_caches = (
                rollout_step_tt(rng, histories, history_masks, actions[:, j], obs_dim, all_dim, dynamics_terminal_fn
                             , dynamics_min_values, dynamics_max_values, dynamics_support, dynamics_sigma, dynamics_obs_act_indicator
                             , dynamics_apply_fn, dynamics_params, updated_caches))
        else:
            rng, next_obs, rewards, terminal, updated_caches = (
                rollout_step_tt(rng, histories[:, -1:], history_masks[:, -1:], actions[:, j], obs_dim, all_dim, dynamics_terminal_fn
                             , dynamics_min_values, dynamics_max_values, dynamics_support, dynamics_sigma, dynamics_obs_act_indicator
                             , dynamics_apply_fn, dynamics_params, updated_caches))

        if clip:
            rewards = jnp.clip(rewards, min_reward, max_reward)

        next_column = jnp.concatenate([next_obs, rewards, jnp.zeros((batch_size, act_dim))], axis=-1)
        histories = jnp.concatenate([histories, next_column[:, None]], axis=1)
        next_column = jnp.ones((batch_size, 1))
        history_masks = jnp.concatenate([history_masks, next_column], axis=1)

        all_rewards.append(rewards)
        all_terminals.append(terminal)

    return rng, histories, history_masks, jnp.concat(all_rewards, axis=-1), jnp.concat(all_terminals, axis=-1)


def estimate_returns_with_transformer_us_jit(
        rng,
        model: TrajWorldDynamics, initial_states, histories, history_masks, # actions,
        min_reward, max_reward, min_state, max_state, clip, horizon,
        clear_kv_cache_every, trm_lookback_window, actor_weights=None,
):
    returns = 0
    masks = jnp.ones((initial_states.shape[0], 1))
    states = initial_states

    batch_size, state_dim, action_dim = states.shape[0], model.config.obs_dim, model.config.act_dim

    if actor_weights is not None: # action proposal
        actor_weights = deepcopy(actor_weights)
        actor_nonlinearity = actor_weights.pop('nonlinearity')
        actor_apply_fn = dope_policies.get_jax_policy_actions
        rng, actions = actor_apply_fn(rng, actor_weights, actor_nonlinearity, states)
        rng, subkey = jax.random.split(rng)
        actions = actions[:, None, :]
        actions = actions + generate_normal_noise(rng, (batch_size, horizon, action_dim), std=args.std)
        actions = jnp.clip(actions, -1.0, 1.0)
    else:
        rng, subkey = jax.random.split(rng)
        actions = generate_normal_noise(rng, (batch_size, horizon, action_dim), std=math.sqrt(2.0))
        actions = jnp.clip(actions, -1.0, 1.0)

    for i in range(0, horizon, clear_kv_cache_every):
        rng, histories, history_masks, rewards, terminals = _rollout_kernel_us(
            rng,
            model.config.obs_dim, model.config.act_dim,
            trm_lookback_window, histories[:, -trm_lookback_window:], history_masks[:, -trm_lookback_window:],
            clear_kv_cache_every,
            clip, min_reward, max_reward, min_state, max_state,
            actions[:, i:i + clear_kv_cache_every],
            model.min_values, model.max_values, model.support, model.sigma, model.obs_act_indicator,
            model.empty_cache, model.terminal_fn, model.trm.apply_fn, model.trm.params
        )
        for j in range(clear_kv_cache_every):
            returns += masks * rewards[:, j:j + 1]
            masks = masks * (1 - terminals[:, j:j + 1])

        if jnp.all(masks == 0):
            break

    actions_idx = jnp.argmax(returns)
    action = actions[actions_idx, 0]
    return action

def estimate_returns_with_transformer_tt_us_jit(
        rng,
        model: TrajWorldDynamics, initial_states, histories, history_masks, # actions,
        min_reward, max_reward, min_state, max_state, clip, horizon,
        clear_kv_cache_every, trm_lookback_window, actor_weights=None,
):
    returns = 0
    masks = jnp.ones((initial_states.shape[0], 1))
    states = initial_states

    batch_size, state_dim, action_dim = states.shape[0], model.config.obs_dim, model.config.act_dim

    if actor_weights is not None: # action proposal
        actor_weights = deepcopy(actor_weights)
        actor_nonlinearity = actor_weights.pop('nonlinearity')
        actor_apply_fn = dope_policies.get_jax_policy_actions
        rng, actions = actor_apply_fn(rng, actor_weights, actor_nonlinearity, states)
        rng, subkey = jax.random.split(rng)
        actions = actions[:, None, :]
        actions = actions + generate_normal_noise(rng, (batch_size, horizon, action_dim), std=args.std)
        actions = jnp.clip(actions, -1.0, 1.0)
    else:
        rng, subkey = jax.random.split(rng)
        actions = generate_normal_noise(rng, (batch_size, horizon, action_dim), std=math.sqrt(2.0))
        actions = jnp.clip(actions, -1.0, 1.0)

    for i in range(0, horizon, clear_kv_cache_every):
        rng, histories, history_masks, rewards, terminals = _rollout_kernel_tt_us(
            rng,
            model.config.obs_dim, model.config.act_dim,
            trm_lookback_window, histories[:, -trm_lookback_window:], history_masks[:, -trm_lookback_window:],
            clear_kv_cache_every,
            clip, min_reward, max_reward, min_state, max_state,
            actions[:, i:i + clear_kv_cache_every],
            model.min_values, model.max_values, model.support, model.sigma, model.obs_act_indicator,
            model.empty_cache, model.terminal_fn, model.trm.apply_fn, model.trm.params,
        )

        # all_histories_for_render.append(histories[0, -clear_kv_cache_every:])

        for j in range(clear_kv_cache_every):
            returns += masks * rewards[:, j:j + 1]
            masks = masks * (1 - terminals[:, j:j + 1])

        if jnp.all(masks == 0):
            break

    actions_idx = jnp.argmax(returns)
    action = actions[actions_idx, 0]
    return action

@partial(jax.jit, static_argnames=(
        "obs_dim",
        "act_dim",
        "trm_lookback_window",
        "rollout_length",
        "clip",
        "actor_apply_fn",
        "actor_nonlinearity",
        "dynamics_empty_cache",
        "dynamics_terminal_fn",
        "dynamics_apply_fn",
))
def _rollout_kernel(
        rng: jax.random.PRNGKey,
        obs_dim: int,
        act_dim: int,
        trm_lookback_window: int,
        init_histories: jnp.ndarray,
        init_history_masks: jnp.ndarray,
        rollout_length: int,
        clip: bool,
        min_reward: jnp.ndarray,
        max_reward: jnp.ndarray,
        min_state: jnp.ndarray,
        max_state: jnp.ndarray,
        actor_apply_fn: Callable[..., Any],
        actor_weights: dict[str, Union[jnp.ndarray, str]],
        actor_nonlinearity: str,
        dynamics_min_values: jnp.ndarray,
        dynamics_max_values: jnp.ndarray,
        dynamics_support: jnp.ndarray,
        dynamics_sigma: float,
        dynamics_obs_act_indicator: jnp.ndarray,
        dynamics_empty_cache: Callable[..., Any],
        dynamics_terminal_fn: Callable[..., Any],
        dynamics_apply_fn: Callable[..., Any],
        dynamics_params: Params,
        noise: jnp.ndarray,
) -> Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def step(
            rng: jax.random.PRNGKey,
            inputs: jnp.ndarray,
            padding_mask: jnp.ndarray,
            caches: Optional[List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]],
    ):
        """
        history: [B, T, M]
        padding_mask: [B, T]
        """
        obs = inputs[:, -1, :obs_dim]
        action = inputs[:, -1, obs_dim + 1:]

        inputs = jnp.clip(inputs, dynamics_min_values, dynamics_max_values)
        inputs = transform_to_probs(inputs, dynamics_support, dynamics_sigma)
        obs_act_indicator = jnp.tile(dynamics_obs_act_indicator, (inputs.shape[0], inputs.shape[1], 1))

        rng, key = jax.random.split(rng)

        pred, updated_caches = dynamics_apply_fn(dynamics_params, inputs, obs_act_indicator,
                                                 padding_mask, caches=caches, training=False,
                                                 rngs={'dropout': key},  # TODO: need fix?
                                                 method=TrajWorldTransformer.call_kv_cache
                                                 )

        pred = pred[:, -1, :obs_dim + 1]
        pred_prob = jax.nn.softmax(pred)

        samples = transform_from_probs(pred_prob, dynamics_support[:obs_dim + 1])
        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = dynamics_terminal_fn(obs, action, next_obs)
        info = {}

        return rng, next_obs, reward, terminal, info, updated_caches

    histories, history_masks = init_histories[:, -trm_lookback_window:], init_history_masks[:, -trm_lookback_window:]
    all_dim = obs_dim + act_dim + 1
    batch_size = init_histories.shape[0]
    caches = dynamics_empty_cache(batch_size=batch_size * all_dim)

    all_rewards = []
    all_terminals = []
    for j in range(rollout_length):
        states = histories[:, -1, :obs_dim]
        rng, actions = actor_apply_fn(rng, actor_weights, actor_nonlinearity, states)
        actions = actions + noise[:, j] # added here
        histories = histories.at[:, -1, state_dim + 1:].set(actions)
        if j == 0:
            rng, next_states, rewards, terminal, info, caches = step(
                rng, histories[:, -trm_lookback_window:], history_masks[:, -trm_lookback_window:], caches=caches)
        else:
            rng, next_states, rewards, terminal, info, caches = step(
                rng, histories[:, -1:], history_masks[:, -1:], caches=caches)

        if clip:
            rewards = jnp.clip(rewards, min_reward, max_reward)

        if min_state is not None and max_state is not None:
            next_states = jnp.clip(next_states, min_state, max_state)

        next_column = jnp.concatenate([next_states, rewards, jnp.zeros((batch_size, action_dim))], axis=-1)
        histories = jnp.concatenate([histories, next_column[:, None]], axis=1)
        next_column = jnp.ones((batch_size, 1))
        history_masks = jnp.concatenate([history_masks, next_column], axis=1)

        all_rewards.append(rewards)
        all_terminals.append(terminal)

    return rng, histories, history_masks, jnp.concat(all_rewards, axis=-1), jnp.concat(all_terminals, axis=-1)


def _rollout_kernel_tt(
        rng: jax.random.PRNGKey,
        obs_dim: int,
        act_dim: int,
        trm_lookback_window: int,
        init_histories: jnp.ndarray,
        init_history_masks: jnp.ndarray,
        rollout_length: int,
        clip: bool,
        min_reward: jnp.ndarray,
        max_reward: jnp.ndarray,
        min_state: jnp.ndarray,
        max_state: jnp.ndarray,
        actor_apply_fn: Callable[..., Any],
        actor_weights: dict[str, Union[jnp.ndarray, str]],
        actor_nonlinearity: str,
        dynamics_min_values: jnp.ndarray,
        dynamics_max_values: jnp.ndarray,
        dynamics_support: jnp.ndarray,
        dynamics_sigma: float,
        dynamics_obs_act_indicator: jnp.ndarray,
        dynamics_empty_cache: Callable[..., Any],
        dynamics_terminal_fn: Callable[..., Any],
        dynamics_apply_fn: Callable[..., Any],
        dynamics_params: Params,
        noise: jnp.ndarray,
) -> Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    histories, history_masks = init_histories[:, -trm_lookback_window:], init_history_masks[:, -trm_lookback_window:]
    batch_size = init_histories.shape[0]

    all_dim = obs_dim + 1 + act_dim

    updated_caches = dynamics_empty_cache(batch_size=batch_size)
    all_rewards = []
    all_terminals = []
    for j in range(rollout_length):
        states = histories[:, -1, :obs_dim]
        rng, actions = actor_apply_fn(rng, actor_weights, actor_nonlinearity, states)
        actions = actions + noise[:, j]  # added here

        if j == 0:
            rng, next_obs, rewards, terminal, updated_caches = (
                rollout_step_tt(rng, histories, history_masks, actions, obs_dim, all_dim, dynamics_terminal_fn
                                , dynamics_min_values, dynamics_max_values, dynamics_support, dynamics_sigma,
                                dynamics_obs_act_indicator
                                , dynamics_apply_fn, dynamics_params, updated_caches))
        else:
            rng, next_obs, rewards, terminal, updated_caches = (
                rollout_step_tt(rng, histories[:, -1:], history_masks[:, -1:], actions, obs_dim, all_dim,
                                dynamics_terminal_fn
                                , dynamics_min_values, dynamics_max_values, dynamics_support, dynamics_sigma,
                                dynamics_obs_act_indicator
                                , dynamics_apply_fn, dynamics_params, updated_caches))

        if clip:
            rewards = jnp.clip(rewards, min_reward, max_reward)

        next_column = jnp.concatenate([next_obs, rewards, jnp.zeros((batch_size, act_dim))], axis=-1)
        histories = jnp.concatenate([histories, next_column[:, None]], axis=1)
        next_column = jnp.ones((batch_size, 1))
        history_masks = jnp.concatenate([history_masks, next_column], axis=1)

        all_rewards.append(rewards)
        all_terminals.append(terminal)

    return rng, histories, history_masks, jnp.concat(all_rewards, axis=-1), jnp.concat(all_terminals, axis=-1)


def estimate_returns_with_transformer_jit(
        rng,
        model: TrajWorldDynamics, initial_states, actor_weights, discount,
        min_reward, max_reward, min_state, max_state, clip, horizon,
        clear_kv_cache_every, trm_lookback_window,
):
    returns = 0
    masks = jnp.ones((initial_states.shape[0], 1))
    states = initial_states

    batch_size, state_dim, action_dim = states.shape[0], model.config.obs_dim, model.config.act_dim

    histories = jnp.concat([states[:, None], jnp.zeros((batch_size, 1, 1 + action_dim))], axis=-1)
    # history_masks = jnp.ones((batch_size, 1))
    histories = jnp.concat(
        [jnp.zeros((batch_size, trm_lookback_window - 1, state_dim + 1 + action_dim)), histories], axis=1)
    history_masks = jnp.concat([jnp.zeros((batch_size, trm_lookback_window - 1)), jnp.ones((batch_size, 1))], axis=-1)

    actor_weights = deepcopy(actor_weights)
    actor_nonlinearity = actor_weights.pop('nonlinearity')
    actor_output_transformation = actor_weights.pop('output_distribution')

    noise = generate_normal_noise(rng, (batch_size, horizon, action_dim), std=args.std)
    _, init_action = dope_policies.get_jax_policy_actions(rng, actor_weights, actor_nonlinearity, states)

    for i in range(0, horizon, clear_kv_cache_every):
        rng, histories, history_masks, rewards, terminals = _rollout_kernel(
            rng,
            model.config.obs_dim, model.config.act_dim,
            trm_lookback_window, histories[:, -trm_lookback_window:], history_masks[:, -trm_lookback_window:],
            clear_kv_cache_every,
            clip, min_reward, max_reward, min_state, max_state,
            dope_policies.get_jax_policy_actions,
            actor_weights, actor_nonlinearity,
            model.min_values, model.max_values, model.support, model.sigma, model.obs_act_indicator,
            model.empty_cache, model.terminal_fn, model.trm.apply_fn, model.trm.params, noise
        )

        # all_histories_for_render.append(histories[0, -clear_kv_cache_every:])

        for j in range(clear_kv_cache_every):
            returns += (discount ** (i + j)) * masks * rewards[:, j:j + 1]
            masks = masks * (1 - terminals[:, j:j + 1])

        if jnp.all(masks == 0):
            break

    actions_idx = jnp.argmax(returns)
    action = init_action[actions_idx] + noise[actions_idx, 0]
    return rng, action


def estimate_returns_with_transformer_tt_jit(
        rng,
        model: TrajWorldDynamics, initial_states, actor_weights, discount,
        min_reward, max_reward, min_state, max_state, clip, horizon,
        clear_kv_cache_every, trm_lookback_window,
):
    returns = 0
    masks = jnp.ones((initial_states.shape[0], 1))
    states = initial_states

    batch_size, state_dim, action_dim = states.shape[0], model.config.obs_dim, model.config.act_dim

    histories = jnp.concat([states[:, None], jnp.zeros((batch_size, 1, 1 + action_dim))], axis=-1)
    # history_masks = jnp.ones((batch_size, 1))
    histories = jnp.concat(
        [jnp.zeros((batch_size, trm_lookback_window - 1, state_dim + 1 + action_dim)), histories], axis=1)
    history_masks = jnp.concat([jnp.zeros((batch_size, trm_lookback_window - 1)), jnp.ones((batch_size, 1))], axis=-1)

    actor_weights = deepcopy(actor_weights)
    actor_nonlinearity = actor_weights.pop('nonlinearity')
    actor_output_transformation = actor_weights.pop('output_distribution')

    noise = generate_normal_noise(rng, (batch_size, horizon, action_dim), std=args.std)
    _, init_action = dope_policies.get_jax_policy_actions(rng, actor_weights, actor_nonlinearity, states)

    for i in range(0, horizon, clear_kv_cache_every):
        rng, histories, history_masks, rewards, terminals = _rollout_kernel_tt(
            rng,
            model.config.obs_dim, model.config.act_dim,
            trm_lookback_window, histories[:, -trm_lookback_window:], history_masks[:, -trm_lookback_window:],
            clear_kv_cache_every,
            clip, min_reward, max_reward, min_state, max_state,
            dope_policies.get_jax_policy_actions,
            actor_weights, actor_nonlinearity,
            model.min_values, model.max_values, model.support, model.sigma, model.obs_act_indicator,
            model.empty_cache, model.terminal_fn, model.trm.apply_fn, model.trm.params, noise
        )

        for j in range(clear_kv_cache_every):
            returns += (discount ** (i + j)) * masks * rewards[:, j:j + 1]
            masks = masks * (1 - terminals[:, j:j + 1])

        if jnp.all(masks == 0):
            break

    actions_idx = jnp.argmax(returns)
    action = init_action[actions_idx] + noise[actions_idx, 0]
    return rng, action

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='ens', type=str, choices=['trajworld', 'ens', 'tdm', 'random', 'actor'])
    parser.add_argument("--env", default="halfcheetah-expert-v2")
    parser.add_argument('--task', default='d4rl')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--discount', default=0.995, type=float)
    parser.add_argument('--normalize_states', default=False, action='store_true')
    parser.add_argument('--normalize_rewards', default=False, action='store_true')
    parser.add_argument('--horizon', default=25, type=int)
    # MPC
    parser.add_argument('--Nsamples', default=128, type=int)
    parser.add_argument('--action_proposal_id', default=-1, type=int)
    parser.add_argument('--receding_horizon', default=False, action='store_true')
    parser.add_argument('--std', default=0.05, type=float)
    # MC estimate
    parser.add_argument('--mc_estimate_target_policy', default=False, action='store_true')
    parser.add_argument('--num_mc_episodes', default=100, type=int)
    # Work dir
    parser.add_argument('--model_path', default='path/to/pretrained/architecture', type=str)
    # Transformer
    parser.add_argument('--trm_lookback_window', default=10, type=int)
    parser.add_argument('--clear_kv_cache_every', default=10, type=int)
    parser.add_argument('--history_length', default=20, type=int)
    parser.add_argument("--group", default=-1, type=int)
    parser.add_argument("--n_blocks", default=6, type=int)
    parser.add_argument("--mlp_huge", default=False, action='store_true')
    # Evaluation
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--max_steps', default=1000, type=int)
    parser.add_argument('--episode_start', default=0, type=int)

    args = parser.parse_args()

    conf_dict = OmegaConf.from_cli()
    config = MOPOConfig(**conf_dict)
    config.penalty_coef = 0.0  # as we are evaluating, the reward should be raw_rewards instead of penalized rewards
    config.env_name = args.env
    config.history_length = args.history_length
    config.n_blocks = args.n_blocks
    config.mlp_huge = args.mlp_huge

    env = gym.make(args.env)
    # render_env = gym.make(args.env)
    # render_env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_space = env.action_space
    max_action = float(env.action_space.high[0])

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    utils.set_seed_everywhere(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    """Get dataset"""
    behavior_dataset = D4rlDataset(
        env,
        normalize_states=args.normalize_states,
        normalize_rewards=args.normalize_rewards,
        seed=args.seed)
    min_reward = jnp.min(behavior_dataset.rewards)
    max_reward = jnp.max(behavior_dataset.rewards)
    min_state = jnp.min(behavior_dataset.states, 0)
    max_state = jnp.max(behavior_dataset.states, 0)
    config.obs_dim = behavior_dataset.states.shape[-1]
    config.act_dim = behavior_dataset.actions.shape[-1]
    print("Dataset Size:", behavior_dataset.states.shape[0])

    if config.mlp_huge:
        config.dynamics_hidden_dims = (640, 640, 640, 640)
        config.obs_dim, config.act_dim = 90, 30
        print('Using Huge MLP Dynamics Model')

    """Get action proposal"""
    if args.action_proposal_id != -1:
        print("Getting action proposal")
        agent_name = args.env.split('-')[0]
        policies = [f"{agent_name}_online_{i}" for i in range(11)]
        returns = []
        actors = []
        for target_policy in policies:
            # logger = Logger(work_dir, use_tb=True)
            """Get target policy"""
            if args.task == 'd4rl':
                actor, policy_returns = dope_policies.get_target_policy(target_policy, args)
            else:
                raise NotImplementedError
            returns.append(policy_returns)
            actors.append(actor)
        """Get action proposal"""
        returns = jnp.array(returns)
        sorted_indices = jnp.argsort(returns)
        mid_point = sorted_indices[args.action_proposal_id]
        actor = actors[mid_point]
        actor_proposal = actors[mid_point].weights
    else:
        actor = None
        actor_proposal = None

    print("---------------------------------------")
    print(f"Algo: {args.algo}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    model_list = []

    model_name_list = get_list_dirs(args.group)

    print(f"{len(model_name_list)} models to evaluate")

    for model_name in model_name_list:
        model_list.append(model_name)

    if len(model_list) == 0:
        model_list = [None]

    last_tag = None

    for model_path in model_list:

        header_lh = ['seed', 'episode', 'method', 'training_env', 'horizon', 'n_sample', 'std', 'action_proposal_id','reward']

        out_csv = f"mpc_pt_new.csv"

        method = args.algo

        """Get architecture"""
        if args.algo == 'trajworld':
            dummy_max_values = jnp.max(behavior_dataset.states, axis=0)  # will later be replaced by loaded architecture
            dummy_min_values = jnp.min(behavior_dataset.states, axis=0)
            model = TrajWorldDynamics(config, dummy_max_values, dummy_min_values)
            model.load(model_path)
        elif args.algo == 'ens':
            model = EnsembleDynamics(config)
            model.load(model_path)
            model.scaler.load_scaler(model_path)
        elif args.algo == 'tdm':
            dummy_max_values = jnp.max(behavior_dataset.states, axis=0)  # will later be replaced by loaded architecture
            dummy_min_values = jnp.min(behavior_dataset.states, axis=0)
            model = TDM_Dynamics(config, dummy_max_values, dummy_min_values)
            model.load(model_path)
        elif args.algo == 'random' or args.algo == 'actor':
            model = None
        else:
            raise NotImplementedError

        if model is not None:
            checkpoint_num = model_path.split('/')[-1][12:-4] if len(model_path.split('/')[-1]) > 16 else 0

            if last_tag is None or last_tag != model_path.split('/')[-3]:

                if last_tag is not None:
                    # we have a new global logger, close the old one
                    global_logger._sw.close()

                last_tag = model_path.split('/')[-3]
                print(f"New tag: {last_tag}")

                global_dir = f"global_new/{model_path.split('/')[-4]}-{model_path.split('/')[-3]}-{time.time()}"
                os.makedirs(global_dir, exist_ok=True)
                os.makedirs(os.path.join(global_dir, 'src'), exist_ok=True)
                global_logger = Logger(global_dir, use_tb=True)
                with open(os.path.join(global_dir, "model_path.txt"), "a") as f:
                    f.write("architecture path:" + model_path + "\n")

                with open(os.path.join(global_dir, 'args.json'), 'w') as f:
                    json.dump(vars(args), f, sort_keys=True, indent=4)

        """Evaluate architecture with MPC"""
        total_reward_list = []

        for episode in range(args.eval_episodes):
            rng, sub_key = jax.random.split(rng)

            terminal = False
            total_reward = 0
            step = 0

            # Initial histories
            obs = env.reset()
            states = jnp.tile(obs, (args.Nsamples, 1))
            histories = jnp.concat([states[:, None], jnp.zeros((args.Nsamples, 1, 1 + action_dim))], axis=-1)
            histories = jnp.concat([jnp.zeros((args.Nsamples, args.trm_lookback_window - 1, state_dim + 1 + action_dim)), histories], axis=1)
            history_masks = jnp.concat([jnp.zeros((args.Nsamples, args.trm_lookback_window - 1)), jnp.ones((args.Nsamples, 1))], axis=-1)

            # add evaluate episodes
            pbar = trange(args.max_steps)
            pbar.set_description(f"Episode {episode}")

            for step in pbar:
                if args.algo == 'random':
                    rng, sub_key = jax.random.split(rng)
                    action = jax.random.normal(sub_key, (action_dim,)) * jnp.sqrt(2)
                    action = jnp.clip(action, -1.0, 1.0)
                elif args.algo == 'trajworld':
                    # Get actions with MPC
                    if args.action_proposal_id != -1:
                        rng, action = estimate_returns_with_transformer_jit(
                            rng,
                            model, states, actor_proposal, args.discount,
                            min_reward, max_reward, min_state, max_state, True, args.horizon,
                            args.clear_kv_cache_every, args.trm_lookback_window
                        )
                    else:
                        action = estimate_returns_with_transformer_us_jit(
                            rng,
                            model, states, histories, history_masks,
                            min_reward, max_reward, min_state, max_state, True, args.horizon,
                            args.clear_kv_cache_every, args.trm_lookback_window, actor_proposal
                        )
                elif args.algo == 'tdm':
                    if args.action_proposal_id != -1:
                        rng, action = estimate_returns_with_transformer_tt_jit(
                            rng,
                            model, states, actor_proposal, args.discount,
                            min_reward, max_reward, min_state, max_state, True, args.horizon,
                            args.clear_kv_cache_every, args.trm_lookback_window
                        )
                    else:
                        # Get actions with MPC
                        action = estimate_returns_with_transformer_tt_us_jit(
                            rng,
                            model, states, histories, history_masks,
                            min_reward, max_reward, min_state, max_state, True, args.horizon,
                            args.clear_kv_cache_every, args.trm_lookback_window, actor_proposal
                        )
                elif args.algo == 'ens':
                    if args.action_proposal_id != -1:
                        rng, action = estimate_returns_with_ensemble(
                            rng,
                            model, states, actor.get_target_actions, args.discount,
                            min_reward, max_reward, min_state, max_state, True, args.horizon,
                            args.mlp_huge
                        )
                    else:
                        action = estimate_returns_with_ensemble_us(
                            rng,
                            model, states, args.discount,
                            min_reward, max_reward, min_state, max_state, True, args.horizon,
                            args.mlp_huge, actor_proposal
                        )
                elif args.algo == 'actor':
                    actor_weights = deepcopy(actor_proposal)
                    actor_nonlinearity = actor_weights.pop('nonlinearity')
                    actor_apply_fn = dope_policies.get_jax_policy_actions
                    rng, action = actor_apply_fn(rng, actor_weights, actor_nonlinearity, states)
                    rng, subkey = jax.random.split(rng)
                    action = jnp.clip(action[0], -1.0, 1.0)
                # Step environment
                next_obs, reward, terminal, info = env.step(np.array(action.flatten()))

                # record
                states = jnp.tile(next_obs, (args.Nsamples, 1))
                histories = histories.at[:, -1, state_dim + 1:].set(action)
                new_step = jnp.concatenate([next_obs, jnp.zeros((1 + action_dim))], axis=-1)
                new_step = new_step.at[state_dim: state_dim + 1].set(reward)
                new_step = jnp.tile(new_step, (args.Nsamples, 1))[:, None]
                histories = jnp.concatenate([histories, new_step], axis=1)[:, -args.trm_lookback_window:]
                history_masks = jnp.concatenate([history_masks, jnp.ones((args.Nsamples, 1))], axis=1)[:,
                                -args.trm_lookback_window:]
                total_reward += reward

                step += 1
                pbar.set_postfix({'total_reward': total_reward})

                if terminal:
                    break

            total_reward_list.append(total_reward)

            print(f"Episode {episode} total reward: {total_reward}")

        total_reward_list = np.array(total_reward_list)
        print(f"Average reward: {np.mean(total_reward_list)}")
        print(f"Std: {np.std(total_reward_list)}")
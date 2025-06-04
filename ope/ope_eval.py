import glob
import sys

import jax
import jax.numpy as jnp
import gym
import argparse
import os
import d4rl
import json
import time
from tqdm import trange
from coolname import generate_slug
from omegaconf import OmegaConf
from typing import Dict, Tuple, Any, Callable, Optional, Sequence, Union, List
from functools import partial
import flax
from copy import deepcopy
import numpy as np
import d4rl.gym_mujoco

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
python ope/ope_eval.py --algo trajworld --env halfcheetah-expert-v2 --clear_kv_cache_every 10 --trm_lookback_window 10 --group 0 --n_blocks 6
python ope/ope_eval.py --algo trajworld --env walker2d-random-v2 --clear_kv_cache_every 10 --trm_lookback_window 10 --group 23 --n_blocks 6  --max_initial_states 1000
python ope/ope_eval.py --algo ens --env walker2d-expert-v2 --group 30 --max_initial_states 1000 --mlp_huge
python ope/ope_eval.py --algo tdm --env walker2d-random-v2 --group 60 --max_initial_states 100 --clear_kv_cache_every 1 --trm_lookback_window 10 --n_blocks 6 --embed_dim 384
"""

def get_list_dirs(group):
    if group == 0:
        return [
            "/home/NAS/rl_data/log_final/halfcheetah-expert-v2/scratch_smalllr_6_layer_seed11&seed_11&timestamp_25-0108-231808/model/trm_dynamics300.pkl",
            "/home/NAS/rl_data/log_final/halfcheetah-expert-v2/scratch_smalllr_6_layer_seed183&seed_183&timestamp_25-0108-231801/model/trm_dynamics300.pkl",
            "/home/NAS/rl_data/log_final/halfcheetah-expert-v2/scratch_smalllr_6_layer_seed83&seed_83&timestamp_25-0108-231803/model/trm_dynamics300.pkl",
        ]
    else:
        raise NotImplementedError

def estimate_returns_with_ensemble(
        rng,
        model: EnsembleDynamics, initial_states, get_target_actions, discount,
        min_reward, max_reward, min_state, max_state, clip=True, horizon=2000,
        mlp_huge=False
):
    returns = 0
    masks = jnp.ones((initial_states.shape[0], 1))
    states = initial_states
    bar = trange(horizon, desc='Estimating Returns with Ensemble')
    for i in bar:
        actions = get_target_actions(states)
        rng, next_states, rewards, terminal, info = model.step(rng, states, actions, mlp_huge)
        if clip:
            rewards = jnp.clip(rewards, min_reward, max_reward)
        returns += (discount ** i) * masks * rewards
        bar.set_postfix({'estimated return': jnp.mean(returns), 'estimated reward': jnp.mean(masks * rewards)})

        masks = masks * (1 - terminal)
        states = next_states
        if min_state is not None and max_state is not None:
            states = jnp.clip(states, min_state, max_state)
        if jnp.all(masks == 0):
            break

    return rng, jnp.mean(returns).item()

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
        # inputs = transform_to_onehot(inputs, self.support, self.sigma)
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
) -> Tuple[jax.random.PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    @jax.jit
    def step(
            rng: jax.random.PRNGKey,
            inputs: jnp.ndarray,
            padding_mask: jnp.ndarray,
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

        pred= dynamics_apply_fn(dynamics_params, inputs, obs_act_indicator,
                                                 padding_mask, training=False,
                                                 rngs={'dropout': key},  # TODO: need fix?
                                                 )
        pred = pred[:, -1]
        pred_prob = jax.nn.softmax(pred)

        dynamics_support_out = jnp.concat([dynamics_support[1:], dynamics_support[:1]], axis=0)
        samples = transform_from_probs(pred_prob, dynamics_support_out)

        return rng, samples

    histories, history_masks = init_histories[:, -trm_lookback_window:], init_history_masks[:, -trm_lookback_window:]
    batch_size = init_histories.shape[0]

    all_dim = obs_dim + 1 + act_dim

    all_rewards = []
    all_terminals = []
    for j in range(rollout_length):
        states = histories[:, -1, :obs_dim]
        rng, actions = actor_apply_fn(rng, actor_weights, actor_nonlinearity, states)
        histories = histories.at[:, -1, state_dim + 1:].set(actions)

        for k in range(obs_dim + 1):
            if k  == 0:
                rng, samples = step(
                    rng, histories, history_masks)
                pred = samples[:, -1:]
                padded_pred = jnp.concat([pred, jnp.zeros((samples.shape[0], all_dim - 1))], axis=-1)
                padded_pred = padded_pred[:, None, :]
                histories = jnp.concat([histories, padded_pred], axis=1)
                history_masks = jnp.concat([history_masks, jnp.ones((samples.shape[0], 1))], axis=1)
            elif k < obs_dim:
                rng, samples = step(
                    rng, histories, history_masks)
                pred = samples[:, None, k-1:k]
                histories = histories.at[:, -1:, k:k+1].set(pred)
            else:
                # only reward is needed
                rng, samples = step(
                    rng, histories, history_masks)
                rewards = samples[:, k-1:k]
                histories = histories.at[:, -1, k:k+1].set(rewards)

                next_obs = histories[:, -1, :obs_dim]
                obs = histories[:, -2, :obs_dim]
                action = histories[:, -2, obs_dim + 1:]
                terminal = dynamics_terminal_fn(obs, action, next_obs)
                info = {}

        if clip:
            rewards = jnp.clip(rewards, min_reward, max_reward)

        if min_state is not None and max_state is not None:
            next_states = jnp.clip(next_obs, min_state, max_state)

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
    bar = trange(horizon, desc='Estimating Returns with Transformer')

    batch_size, state_dim, action_dim = states.shape[0], model.config.obs_dim, model.config.act_dim

    histories = jnp.concat([states[:, None], jnp.zeros((batch_size, 1, 1 + action_dim))], axis=-1)
    # history_masks = jnp.ones((batch_size, 1))
    histories = jnp.concat(
        [jnp.zeros((batch_size, trm_lookback_window - 1, state_dim + 1 + action_dim)), histories], axis=1)
    history_masks = jnp.concat([jnp.zeros((batch_size, trm_lookback_window - 1)), jnp.ones((batch_size, 1))], axis=-1)

    actor_weights = deepcopy(actor_weights)
    actor_nonlinearity = actor_weights.pop('nonlinearity')
    actor_output_transformation = actor_weights.pop('output_distribution')

    for i in range(0, horizon, clear_kv_cache_every):
        rng, histories, history_masks, rewards, terminals = _rollout_kernel(
            rng,
            model.config.obs_dim, model.config.act_dim,
            trm_lookback_window, histories[:, -trm_lookback_window:], history_masks[:, -trm_lookback_window:],
            clear_kv_cache_every,
            clip, min_reward, max_reward, min_state, max_state,
            dope_policies.get_jax_policy_actions,
            actor_weights, actor_nonlinearity,
            model.min_values, model.max_values, model.support, model.sigma, model.obs_act_indicator,  # fixme
            model.empty_cache, model.terminal_fn, model.trm.apply_fn, model.trm.params,
        )

        # all_histories_for_render.append(histories[0, -clear_kv_cache_every:])

        for j in range(clear_kv_cache_every):
            returns += (discount ** (i + j)) * masks * rewards[:, j:j + 1]
            masks = masks * (1 - terminals[:, j:j + 1])

        bar.update(clear_kv_cache_every)
        bar.set_postfix({'estimated return': jnp.mean(returns),
                         'estimated reward': jnp.mean(rewards), 'masks': jnp.mean(masks)})
        if jnp.all(masks == 0):
            break

    return rng, jnp.mean(returns)

def estimate_returns_with_transformer_tt_jit(
        rng,
        model: TrajWorldDynamics, initial_states, actor_weights, discount,
        min_reward, max_reward, min_state, max_state, clip, horizon,
        clear_kv_cache_every, trm_lookback_window,
):
    returns = 0
    masks = jnp.ones((initial_states.shape[0], 1))
    states = initial_states
    bar = trange(horizon, desc='Estimating Returns with Transformer')

    batch_size, state_dim, action_dim = states.shape[0], model.config.obs_dim, model.config.act_dim

    histories = jnp.concat([states[:, None], jnp.zeros((batch_size, 1, 1 + action_dim))], axis=-1)
    # history_masks = jnp.ones((batch_size, 1))
    histories = jnp.concat(
        [jnp.zeros((batch_size, trm_lookback_window - 1, state_dim + 1 + action_dim)), histories], axis=1)
    history_masks = jnp.concat([jnp.zeros((batch_size, trm_lookback_window - 1)), jnp.ones((batch_size, 1))], axis=-1)

    actor_weights = deepcopy(actor_weights)
    actor_nonlinearity = actor_weights.pop('nonlinearity')
    actor_output_transformation = actor_weights.pop('output_distribution')

    for i in range(0, horizon, clear_kv_cache_every):
        rng, histories, history_masks, rewards, terminals = _rollout_kernel_tt(
            rng,
            model.config.obs_dim, model.config.act_dim,
            trm_lookback_window, histories[:, -trm_lookback_window:], history_masks[:, -trm_lookback_window:],
            clear_kv_cache_every,
            clip, min_reward, max_reward, min_state, max_state,
            dope_policies.get_jax_policy_actions,
            actor_weights, actor_nonlinearity,
            model.min_values, model.max_values, model.support, model.sigma, model.obs_act_indicator,  # fixme
            model.empty_cache, model.terminal_fn, model.trm.apply_fn, model.trm.params,
        )

        # all_histories_for_render.append(histories[0, -clear_kv_cache_every:])

        for j in range(clear_kv_cache_every):
            returns += (discount ** (i + j)) * masks * rewards[:, j:j + 1]
            masks = masks * (1 - terminals[:, j:j + 1])

        bar.update(clear_kv_cache_every)
        bar.set_postfix({'estimated return': jnp.mean(returns),
                         'estimated reward': jnp.mean(rewards), 'masks': jnp.mean(masks)})
        if jnp.all(masks == 0):
            break

    return rng, jnp.mean(returns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='ens', type=str, choices=['trajworld', 'ens', 'tdm'])
    parser.add_argument("--env", default="halfcheetah-medium-v2")
    parser.add_argument('--task', default='d4rl')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--discount', default=0.995, type=float)
    parser.add_argument('--normalize_states', default=False, action='store_true')
    parser.add_argument('--normalize_rewards', default=False, action='store_true')
    parser.add_argument('--horizon', default=2000, type=int)
    # MC estimate
    parser.add_argument('--mc_estimate_target_policy', default=False, action='store_true')
    parser.add_argument('--num_mc_episodes', default=100, type=int)
    # Work dir
    parser.add_argument('--notes', default=None, type=str)
    parser.add_argument('--model_path', default='path/to/pretrained/architecture', type=str)
    # Transformer
    parser.add_argument('--trm_lookback_window', default=None, type=int)
    parser.add_argument('--clear_kv_cache_every', default=10, type=int)
    parser.add_argument('--max_initial_states', default=None, type=int)
    parser.add_argument('--history_length', default=20, type=int)
    parser.add_argument("--group", default=0, type=int)
    parser.add_argument("--n_blocks", default=6, type=int)
    parser.add_argument("--embed_dim", default=256, type=int)
    parser.add_argument("--mlp_huge", default=False, action='store_true')

    args = parser.parse_args()
    args.cooldir = generate_slug(2)

    conf_dict = OmegaConf.from_cli()
    config = MOPOConfig(**conf_dict)
    config.penalty_coef = 0.0  # as we are evaluating, the reward should be raw_rewards instead of penalized rewards
    config.env_name = args.env
    config.history_length = args.history_length
    config.n_blocks = args.n_blocks
    config.mlp_huge = args.mlp_huge
    config.embed_dim = args.embed_dim
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
    # render_env.seed(args.seed)
    # render_env.action_space.seed(args.seed)

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


    unnormalized_initial_states = behavior_dataset.unnormalize_states(behavior_dataset.initial_states)
    if args.max_initial_states is not None:
        unnormalized_initial_states = unnormalized_initial_states[:args.max_initial_states]

    print("---------------------------------------")
    print(f"Algo: {args.algo}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    model_list = []

    model_name_list = get_list_dirs(args.group)

    print(f"{len(model_name_list)} models to evaluate")

    for model_name in model_name_list:
        model_list.append(model_name)

    if len(model_list) == 0:
        model_list = [args.model_path]

    last_tag = None

    for model_path in model_list:
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
        else:
            raise NotImplementedError

        agent_name = args.env.split('-')[0]
        policies = [f"{agent_name}_online_{i}" for i in range(11)]
        checkpoint_num = model_path.split('/')[-1][12:-4] if len(model_path.split('/')[-1]) > 16 else 0

        task_data = {
            'policy_idx': [],
            'true_returns': [],
            'pred_returns': []
        }

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


        for target_policy in policies:

            task_data['policy_idx'].append(int(target_policy.split('_')[-1]))

            # logger = Logger(work_dir, use_tb=True)
            """Get target policy"""
            if args.task == 'd4rl':
                actor, policy_returns = dope_policies.get_target_policy(target_policy, args)
            else:
                raise NotImplementedError
            # logger.log('eval/true_returns', policy_returns, step=0)
            print(f"True Returns: {policy_returns}")

            task_data['true_returns'].append(policy_returns)

            if args.algo == 'trajworld':
                rng, pred_returns = estimate_returns_with_transformer_jit(
                    rng, model, unnormalized_initial_states, actor.weights,
                    args.discount, min_reward, max_reward, min_state, max_state, clip=True,
                    trm_lookback_window=args.trm_lookback_window, clear_kv_cache_every=args.clear_kv_cache_every,
                    horizon=args.horizon)
            elif args.algo == 'ens':
                rng, pred_returns = estimate_returns_with_ensemble(rng, model,
                                                                   unnormalized_initial_states,
                                                                   actor.get_target_actions,
                                                                   args.discount,
                                                                   min_reward=min_reward, max_reward=max_reward,
                                                                   min_state=min_state, max_state=max_state,
                                                                   clip=True,
                                                                   horizon=args.horizon,
                                                                   mlp_huge=args.mlp_huge)
            elif args.algo == 'tdm':
                rng, pred_returns = estimate_returns_with_transformer_tt_jit(
                    rng, model, unnormalized_initial_states, actor.weights,
                    args.discount, min_reward, max_reward, min_state, max_state, clip=True,
                    trm_lookback_window=args.trm_lookback_window, clear_kv_cache_every=args.clear_kv_cache_every,
                    horizon=args.horizon)
            else:
                raise NotImplementedError

            print(f"Pred Returns: {pred_returns}")

            task_data['pred_returns'].append(pred_returns)

        true_values = np.array(task_data['true_returns'])
        pred_values = np.array(task_data['pred_returns'])
        policy_idxs = np.array(task_data['policy_idx'])
        print(f"Number of policies: {len(policy_idxs)}")

        # Normalize true and predicted values
        value_min, value_max = true_values.min(), true_values.max()
        norm_true_values = (true_values - value_min) / (value_max - value_min)
        norm_pred_values = (pred_values - value_min) / (value_max - value_min)

        # Calculate absolute errors
        abs_error = np.abs(true_values - pred_values)
        norm_abs_error = np.abs(norm_true_values - norm_pred_values)
        abs_error_mean = abs_error.mean()
        abs_error_max = abs_error.max()
        abs_error_min = abs_error.min()
        norm_abs_error_mean = norm_abs_error.mean()

        # Rank correlation
        rank_correlation = np.corrcoef(true_values, pred_values)[0, 1]

        # Calculate Regret@1 and Regret@5
        top_1_idx = np.argsort(norm_pred_values)[-1]
        top_5_idx = np.argsort(norm_pred_values)[-5:]

        regret_1 = norm_true_values.max() - norm_true_values[top_1_idx]
        regret_5 = norm_true_values.max() - norm_true_values[top_5_idx].max()

        # True and estimated ranking comparison
        true_ranking = np.argsort(-norm_true_values)  # Descending order
        estimated_ranking = np.argsort(-norm_pred_values)  # Descending order

        # Store metrics
        task_metrics = {
            'AbsError Mean': abs_error_mean,
            'Norm. AbsError Mean': norm_abs_error_mean,
            'RankCorrelation': rank_correlation,
            'Regret@1': regret_1,
            'Regret@5': regret_5
        }

        ranking_results = {
            'True Ranking': policy_idxs[true_ranking],
            'Estimated Ranking': policy_idxs[estimated_ranking]
        }

        output_file = f"{global_dir}/{checkpoint_num}.txt"

        global_logger.log('eval/abs_error_mean', abs_error_mean, step=checkpoint_num)
        global_logger.log('eval/norm_abs_error_mean', norm_abs_error_mean, step=checkpoint_num)
        global_logger.log('eval/rank_correlation', rank_correlation, step=checkpoint_num)
        global_logger.log('eval/regret_1', regret_1, step=checkpoint_num)
        global_logger.log('eval/regret_5', regret_5, step=checkpoint_num)

        with open(output_file, 'a') as f:
            for true_value in true_values:
                f.write(f"{true_value} ")
            f.write("\n")
            # f.write(pred_values)
            for pred_value in pred_values:
                f.write(f"{pred_value} ")
            f.write("\n")
            for metric, value in task_metrics.items():
                f.write(f"  {metric}: {value}\n")
            f.write(f"  True Ranking: {ranking_results['True Ranking']}\n")
            f.write(f"  Estimated Ranking: {ranking_results['Estimated Ranking']}\n")
            f.write("\n")
        print(f"Results saved to {output_file}")


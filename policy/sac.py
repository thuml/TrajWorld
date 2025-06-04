from functools import partial
from typing import Any, Dict, Tuple, NamedTuple
import numpy as np
import pickle

import distrax
import jax
import optax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from data.data import Batch
from dynamics.utils import update_by_loss_grad, InfoDict, Params, update_by_loss_grad_rl
from dynamics.config import MOPOConfig
from policy.net import DoubleCritic, NormalTanhPolicy, Temperature, sample_actions


class TargetQConfig(NamedTuple):
    sample_num_qs: int
    target_type: str


class BCConfig(NamedTuple):
    bc_weight: float
    bc_ratio: float


def compute_target(key: jax.random.PRNGKey, qs: jnp.ndarray, sample_num_qs: int, target_type: str):
    batch_size = qs.shape[1]
    q_indices = jax.vmap(lambda key, num_items: jax.random.choice(key, num_items, (sample_num_qs,), replace=False),
                         in_axes=(0, None))(jax.random.split(key, batch_size), qs.shape[0])
    sample_qs = jnp.stack([qs[q_indices[:, i], jnp.arange(batch_size)] for i in range(sample_num_qs)], axis=0)
    if target_type == 'min':
        target_q = sample_qs.min(0)
    elif target_type == 'mean':
        target_q = sample_qs.mean(0)
    else:
        raise ValueError(f"Unknown target type: {target_type}")
    return target_q


def update_critic(key: jax.random.PRNGKey, actor: TrainState, critic: TrainState, target_critic: TrainState,
                  temp: TrainState, batch: Batch, discount: float,
                  target_config: TargetQConfig) -> Tuple[TrainState, InfoDict]:
    key, sample_key, q_key = jax.random.split(key, 3)
    base_dist: distrax.Distribution = actor.apply_fn(actor.params, batch.next_observations)['base_dist']
    next_actions, next_log_probs = base_dist.sample_and_log_prob(seed=sample_key)
    next_actions = jnp.tanh(next_actions)
    next_log_probs = next_log_probs - jnp.log((1 - jnp.square(next_actions)) + 1e-6).sum(-1)

    next_qs = target_critic.apply_fn(target_critic.params, batch.next_observations, next_actions)
    next_q = compute_target(q_key, next_qs, target_config.sample_num_qs, target_config.target_type)

    target_q = batch.rewards + discount * (1 - batch.dones) * next_q
    target_q -= discount * (1 - batch.dones) * temp.apply_fn(jax.lax.stop_gradient(temp.params)) * next_log_probs
    target_q = jax.lax.stop_gradient(target_q)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        qs = critic.apply_fn(critic_params, batch.observations, batch.actions)
        critic_loss = ((qs - target_q)**2).sum(0).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'qs': qs.mean(),
            'target_q': target_q.mean(),
            'target_q_std': target_q.std(),
            'next_actions_max': jnp.abs(next_actions).max(-1).mean(),
        }

    new_critic, info = update_by_loss_grad_rl(critic, critic_loss_fn, True, return_grad_norm=True)
    return new_critic, info


def update_target(
    model: TrainState, target_model: TrainState, tau: float
) -> TrainState:
    new_target_params = jax.tree.map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


def update_actor(key: jax.random.PRNGKey, actor: TrainState, critic: TrainState, temp: TrainState,
                 batch: Batch, bc_config: BCConfig, target_config: TargetQConfig) -> Tuple[TrainState, InfoDict]:
    key, dropout_key, sample_key, q_key = jax.random.split(key, 4)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actor_out = actor.apply_fn(actor_params, batch.observations, training=True, rngs={'dropout': dropout_key})
        base_dist: distrax.Distribution = actor_out['base_dist']
        actions, log_probs = base_dist.sample_and_log_prob(seed=sample_key)
        actions = jnp.tanh(actions)
        log_probs = log_probs - jnp.log((1 - jnp.square(actions)) + 1e-6).sum(-1)

        qs = critic.apply_fn(jax.lax.stop_gradient(critic.params), batch.observations, actions)
        q = compute_target(q_key, qs, target_config.sample_num_qs, target_config.target_type)

        q = -log_probs * temp.apply_fn(jax.lax.stop_gradient(temp.params)) + q
        actor_loss = -q.mean()
        if bc_config.bc_weight > 0 and bc_config.bc_ratio > 0:
            dist: distrax.Distribution = actor_out['dist']
            bc_log_probs = dist.log_prob(jnp.clip(batch.actions, -1 + 1e-5, 1 - 1e-5))
            bc_size = int(bc_log_probs.shape[0] * bc_config.bc_ratio)
            bc_log_probs = bc_log_probs[:bc_size]
            # actor_loss -= bc_config.bc_weight * bc_log_probs.mean()
            # actor_loss = -bc_log_probs.mean()
            mean_abs_q = jax.lax.stop_gradient(jnp.abs(q[bc_size:]).mean())
            # actor_loss = -q[bc_size:].mean() / mean_abs_q - bc_config.bc_weight * bc_log_probs.mean()
            actor_loss = -q[bc_size:].mean() - bc_config.bc_weight * bc_log_probs.mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'bc_log_probs': bc_log_probs.mean() if bc_config.bc_weight > 0 else 0,
            'bc_weight': bc_config.bc_weight,
        }

    new_actor, info = update_by_loss_grad_rl(actor, actor_loss_fn, True)
    return new_actor, info


def update_temperature(temp: TrainState, entropy: float,
                       target_entropy: float) -> Tuple[TrainState, InfoDict]:

    def temperature_loss_fn(temp_params: Params):
        temperature = temp.apply_fn(temp_params)
        temp_loss = jnp.log(temperature) * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    new_temp, info = update_by_loss_grad_rl(temp, temperature_loss_fn, True)

    return new_temp, info


@partial(jax.jit, static_argnames=('config',))
def update_sac(
    config: MOPOConfig,
    rng: jax.random.PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    temp: TrainState,
    batch: Batch,
) -> Tuple[jax.random.PRNGKey, TrainState, TrainState, TrainState, TrainState, InfoDict]:
    target_config = TargetQConfig(config.sample_num_qs, config.target_q_type)
    bc_config = BCConfig(config.bc_weight, config.real_ratio)

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor, critic, target_critic, temp,
                                            batch,
                                            config.discount, target_config)

    new_target_critic = update_target(new_critic, target_critic, config.tau)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch, bc_config, target_config)
    if config.auto_alpha:
        new_temp, alpha_info = update_temperature(temp, actor_info['entropy'],
                                                  config.target_entropy)
    else:
        new_temp, alpha_info = temp, {}

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


@partial(jax.jit, static_argnames=('config', 'n_times_update', 'micro_batch_size'))
def update_sac_n_times(
    config: MOPOConfig,
    rng: jax.random.PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    temp: TrainState,
    batch: Batch,
    n_times_update: int,
    micro_batch_size: int
):
    for i in range(n_times_update):
        micro_batch = jax.tree.map(lambda x: x[i * micro_batch_size: (i + 1) * micro_batch_size], batch)
        rng, actor, critic, target_critic, temp, info = update_sac(
            config, rng, actor, critic, target_critic, temp,
            micro_batch)
    return rng, actor, critic, target_critic, temp, info


class SACPolicy(object):

    def __init__(self, config: MOPOConfig):
        self.config = config

        self.rng = jax.random.PRNGKey(config.seed)
        observations = jnp.zeros((1, config.obs_dim))
        actions = jnp.zeros((1, config.act_dim))

        self.rng, actor_key, critic_key, temp_key = jax.random.split(self.rng, 4)
        actor_model = NormalTanhPolicy(config.hidden_dims, config.act_dim,
                                       add_layer_norm=config.add_layer_norm, dropout_rate=config.actor_dropout)
        if config.actor_cosine_decay_steps:
            scheduler = optax.cosine_decay_schedule(
                init_value=config.actor_lr, decay_steps=config.actor_cosine_decay_steps, alpha=config.actor_lr_decay_alpha)
            actor_tx = optax.inject_hyperparams(optax.adam)(learning_rate=scheduler)
        else:
            actor_tx = optax.adam(config.actor_lr)
        self.actor = TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_model.init(actor_key, observations),
            tx=actor_tx,
        )
        critic_model = DoubleCritic(config.hidden_dims, add_layer_norm=config.add_layer_norm or config.add_critic_layer_norm, num_qs=config.num_qs)
        self.critic = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_model.init(critic_key, observations, actions),
            tx=optax.chain(
                optax.clip_by_global_norm(config.critic_clip_grads),
                optax.adam(config.critic_lr),
            )
        )
        self.target_critic = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_model.init(critic_key, observations, actions),
            tx=optax.adam(config.critic_lr),
        )
        if config.auto_alpha:
            temp_model = Temperature(1.0)
        else:
            temp_model = Temperature(max(config.alpha, 1e-8))
        self.temp = TrainState.create(
            apply_fn=temp_model.apply,
            params=temp_model.init(temp_key),
            tx=optax.adam(config.alpha_lr),
        )

    def select_action(
        self,
        rng: jax.random.PRNGKey,
        observations: np.ndarray,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        rng, actions = sample_actions(
            rng, self.actor.apply_fn,
            self.actor.params, observations,
            temperature)
        return rng, jnp.clip(actions, -1, 1)  # TODO: make this a parameter

    def select_action_eval(
        self,
        rng: jax.random.PRNGKey,
        observations: np.ndarray,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        rng, actions = sample_actions(
            rng, self.actor.apply_fn,
            self.actor.params, observations,
            temperature)
        # return rng, jnp.clip(actions, -1, 1)  # TODO: make this a parameter
        actions = jnp.clip(actions, -1, 1)
        value = self.critic.apply_fn(self.critic.params, observations, actions)
        print(f"Value: {value.flatten()}")
        return rng, actions

    def update(self, batch) -> InfoDict:
        self.rng, self.actor, self.critic, self.target_critic, self.temp, info = update_sac(
            self.config,
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch)
        if self.config.actor_cosine_decay_steps:
            info['actor_lr'] = self.actor.opt_state.hyperparams['learning_rate']
        return info

    def update_n_times(self, batch, update_times: int, micro_batch_size: int) -> InfoDict:
        self.rng, self.actor, self.critic, self.target_critic, self.temp, info = update_sac_n_times(
            self.config,
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, update_times, micro_batch_size)
        if self.config.actor_cosine_decay_steps:
            info['actor_lr'] = self.actor.opt_state.hyperparams['learning_rate']
        return info

    def save(self, save_path: str) -> None:
        pickle.dump({
            "actor": self.actor.params,
            "critic": self.critic.params,
            "target_critic": self.target_critic.params,
            "temp": self.temp.params,
        }, open(save_path, 'wb'))

    def load(self, load_path: str) -> None:
        params = pickle.load(open(load_path, 'rb'))
        self.actor = self.actor.replace(params=params["actor"])
        self.critic = self.critic.replace(params=params["critic"])
        self.target_critic = self.target_critic.replace(params=params["target_critic"])
        self.temp = self.temp.replace(params=params["temp"])

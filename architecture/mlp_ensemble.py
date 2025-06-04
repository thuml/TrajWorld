from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Union, List

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer
import numpy as np

from dynamics.utils import Params


def symmetric_uniform(scale=1e-2, dtype=jnp.float_) -> Initializer:
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return jax.random.uniform(key, shape, dtype) * jnp.array(scale, dtype) * 2 - jnp.array(scale, dtype)
    return init


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    add_layer_norm: bool = False
    layer_norm_final: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(
                size,
                kernel_init=symmetric_uniform(scale=1 / jnp.sqrt(x.shape[-1])),
                bias_init=symmetric_uniform(scale=1 / jnp.sqrt(x.shape[-1])),
            )(x)
            if self.add_layer_norm:
                if self.layer_norm_final or i + 1 < len(self.hidden_dims):
                    x = nn.LayerNorm()(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    add_layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations,
                     add_layer_norm=self.add_layer_norm, layer_norm_final=False)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2
    add_layer_norm: bool = False

    @nn.compact
    def __call__(self, states, actions):

        VmapCritic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        activations=self.activations,
                        add_layer_norm=self.add_layer_norm)(states, actions)
        return qs


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def tanh_normal_log_prob(squashed_action: jnp.ndarray, base_dist: distrax.Distribution, raw_action: jnp.ndarray) -> jnp.ndarray:
    log_prob = base_dist.log_prob(raw_action)
    eps = 1e-6
    log_prob = log_prob - jnp.log((1 - jnp.square(squashed_action)) + eps).sum(-1)
    return log_prob


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    init_mean: Optional[jnp.ndarray] = None
    add_layer_norm: bool = False

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate,
                      add_layer_norm=self.add_layer_norm,
                      layer_norm_final=True)(observations, training=training)

        means = nn.Dense(
            self.action_dim,
            kernel_init=symmetric_uniform(scale=1 / jnp.sqrt(outputs.shape[-1])),
            bias_init=symmetric_uniform(scale=1 / jnp.sqrt(outputs.shape[-1])),
        )(outputs)
        if self.init_mean is not None:
            means += self.init_mean

        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim,
                kernel_init=symmetric_uniform(scale=1 / jnp.sqrt(outputs.shape[-1])),
                bias_init=symmetric_uniform(scale=1 / jnp.sqrt(outputs.shape[-1])),
            )(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = distrax.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash_distribution:
            dist = distrax.Transformed(distribution=base_dist,
                                       bijector=distrax.Block(distrax.Tanh(), ndims=1))
        else:
            dist = base_dist
        return {
            'base_mode': means,
            'mode': jnp.tanh(means),
            'dist': dist,
            'base_dist': base_dist,
        }


@partial(jax.jit, static_argnames=('actor_apply_fn', 'distribution'))
def sample_actions(
    rng: jax.random.PRNGKey,
    actor_apply_fn: Callable[..., Any],
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
    distribution: str = 'log_prob'
) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
    if distribution == 'det':
        return rng, actor_apply_fn(actor_params, observations, temperature)['mode']
    else:
        dist = actor_apply_fn(actor_params, observations, temperature)['dist']
        rng, key = jax.random.split(rng)
        return rng, dist.sample(seed=key)


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


def soft_clamp(
    x: jnp.ndarray,
    _min: Optional[jnp.ndarray] = None,
    _max: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - nn.softplus(_max - x)
    if _min is not None:
        x = _min + nn.softplus(x - _min)
    return x


class EnsembleLinear(nn.Module):
    input_dim: int
    output_dim: int
    num_ensemble: int

    def setup(self):
        # TODO: refactor with vmap like Double critic
        self.weight = self.param('weight', nn.initializers.truncated_normal(stddev=1 / (2 * self.input_dim**0.5)),
                                 (self.num_ensemble, self.input_dim, self.output_dim))
        self.bias = self.param('bias', nn.initializers.zeros, (self.num_ensemble, 1, self.output_dim))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            x = jnp.einsum('ij,bjk->bik', x, weight)
        elif len(x.shape) == 3:
            x = jnp.einsum('bij,bjk->bik', x, weight)
        elif len(x.shape) == 4:
            x = jnp.einsum('bdij,bjk->bdik', x, weight)
        if len(x.shape) == 4:
            bias = jnp.expand_dims(bias, 2)
        x = x + bias

        return x

    @staticmethod
    def get_decay_loss(params, weight_decay: float):
        decay_loss = weight_decay * (0.5 * ((params['weight']**2).sum()))
        return decay_loss


class EnsembleDynamicsModel(nn.Module):
    obs_dim: int
    action_dim: int
    hidden_dims: Union[List[int], Tuple[int]]
    num_ensemble: int = 7
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
    with_reward: bool = True

    def setup(self):
        hidden_dims = [self.obs_dim + self.action_dim] + list(self.hidden_dims)
        module_list = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            module_list.append(EnsembleLinear(in_dim, out_dim, self.num_ensemble))
        self.backbones = module_list

        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * (self.obs_dim + self.with_reward),
            self.num_ensemble,
        )

        self.max_logvar = self.param('max_logvar', nn.initializers.constant(0.5), (self.obs_dim + self.with_reward,))
        self.min_logvar = self.param('min_logvar', nn.initializers.constant(-10.0), (self.obs_dim + self.with_reward,))

    def __call__(self, obs_action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mean, logvar = jnp.split(self.output_layer(output), 2, axis=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        max_logvar = self.max_logvar
        min_logvar = self.min_logvar
        return mean, logvar, max_logvar, min_logvar

    @staticmethod
    def get_decay_loss(params, weight_decay: Sequence[float]):
        decay_loss = 0
        for i, decay in enumerate(weight_decay[:-1]):
            decay_loss += EnsembleLinear.get_decay_loss(params[f'backbones_{i}'], decay)
        decay_loss += EnsembleLinear.get_decay_loss(params['output_layer'], weight_decay[-1])
        return decay_loss

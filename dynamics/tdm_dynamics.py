import os
import time
from functools import partial
from typing import Any, Dict, Optional, List, Tuple
import pickle
from copy import deepcopy
from tqdm import trange

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from scipy.spatial import KDTree

from dynamics.config import MOPOConfig
from data.history_data import HistoryBatch
from env.termination_fns import get_termination_fn
from dynamics.utils import update_by_loss_grad, InfoDict, Params
from dynamics.logger import Logger

from architecture.trm_tdm import TDMTransformer

@partial(jax.jit, static_argnames=('input_discrete', 'target_discrete'))
def step_trm_for_pretrain(
        rng: jax.random.PRNGKey,
        trm: TrainState,
        inputs: jnp.ndarray,
        obs_act_indicator: jnp.ndarray,
        padding_masks: jnp.ndarray,
        variate_masks: jnp.ndarray,
        obs_select: jnp.ndarray,
        support: jnp.ndarray,
        sigma: jnp.ndarray,
        input_discrete: str = 'gauss',
        target_discrete: str = 'onehot',
):
    rng, key = jax.random.split(rng)
    key, variate_key = jax.random.split(key)

    inputs_probs = transform(input_discrete, inputs, support, sigma)

    pred = trm.apply_fn(trm.params, inputs_probs, obs_act_indicator, padding_masks, variate_masks,
                        training=False, variate_key=variate_key, rngs={'dropout': key},
                        method=TDMTransformer.call_variate_mask)
    pred_prob = jax.nn.softmax(pred)
    pred_values = transform_from_probs(pred_prob, support)[:, -1:]
    return rng, pred_values

@jax.jit
def transform_to_probs(target: jax.Array, support: jax.Array, sigma: jax.Array) -> jax.Array:
    # *HL-Gauss
    cdf_evals = jax.scipy.special.erf((support - target[..., None]) / (jnp.sqrt(2) * sigma[..., None]))
    z = cdf_evals[..., -1] - cdf_evals[..., 0]
    bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
    return bin_probs / (z[..., None] + 1e-6)


@jax.jit
def transform_to_onehot(target: jax.Array, support: jax.Array, sigma: jax.Array) -> jax.Array:
    # *One-Hot
    min_values, max_values, uniform_bin = support[..., 0], support[..., -1], support.shape[-1] - 1
    target = jnp.clip((target - min_values) / (max_values - min_values + 1e-8), 0, 1)
    target = jnp.floor(target * uniform_bin).astype(jnp.int32).clip(0, uniform_bin - 1)
    return jax.nn.one_hot(target, uniform_bin).astype(jnp.float32)


def transform(type: str, target: jax.Array, support: jax.Array, sigma: jax.Array):
    if type == 'onehot':
        return transform_to_onehot(target, support, sigma)
    elif type == 'gauss':
        return transform_to_probs(target, support, sigma)
    else:
        raise NotImplementedError


@jax.jit
def transform_from_probs(probs: jax.Array, support: jax.Array) -> jax.Array:
    centers = (support[..., :-1] + support[..., 1:]) / 2
    return (probs * centers).sum(-1)


@jax.jit
def transform_from_probs_sample(probs: jax.Array, support: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    key1, key2 = jax.random.split(key)
    centers = (support[..., :-1] + support[..., 1:]) / 2
    sample = jax.random.categorical(key1, jnp.log(probs + 1e-8), axis=-1)
    sample = jax.nn.one_hot(sample, probs.shape[-1])  # convert to onehot
    # return (sample * centers).sum(-1)
    lower_bound, upper_bound = (sample * support[..., :-1]).sum(-1), (sample * support[..., 1:]).sum(-1)
    return jax.random.uniform(key2, shape=sample.shape[:-1]) * (upper_bound - lower_bound) + lower_bound


@partial(jax.jit, static_argnames=('obs_dim', 'weighted_loss', 'input_discrete', 'target_discrete'))
def update_trm(
        rng: jax.random.PRNGKey,
        trm: TrainState,
        inputs: jnp.ndarray,
        obs_act_indicator: jnp.ndarray,
        padding_masks: jnp.ndarray,
        obs_dim: int,
        support: jnp.ndarray,
        sigma: jnp.ndarray,
        weighted_loss: bool,
        input_discrete: str = 'gauss',
        target_discrete: str = 'onehot',
):
    """
    inputs [B, T, M]
    padding_masks: [B, T]
    """
    rng, key = jax.random.split(rng)
    key, variate_key = jax.random.split(key)
    targets = transform(target_discrete, inputs, support, sigma)
    targets = targets.reshape((targets.shape[0], targets.shape[1] * targets.shape[2], targets.shape[3]))
    inputs = transform(input_discrete, inputs, support, sigma)
    # shift padding_masks to the right by 1
    padding_masks = jnp.concat([padding_masks[..., 0][..., None], padding_masks], axis=1)
    obs_select = 1.0 - obs_act_indicator

    def loss_fn(params: Params) -> jnp.ndarray:
        pred = trm.apply_fn(params, inputs, obs_act_indicator, padding_masks, training=True, variate_key=variate_key,
                            rngs={'dropout': key})
        B, T, M, D = pred.shape
        pred = pred.reshape((pred.shape[0], pred.shape[1] * pred.shape[2], pred.shape[3]))
        pred, target = pred[:, :-1], targets[:, 1:]
        loss = optax.softmax_cross_entropy(pred, target)
        loss = jnp.concat([jnp.zeros((loss.shape[0], 1)), loss], axis=1)
        loss = loss.reshape(B, T, M)
        loss = (loss * obs_select).sum(-1) / (obs_select.sum(-1) + 1e-6)
        loss = (padding_masks[:, :-1] * loss).sum() / (padding_masks[:, :-1].sum() + 1e-6)
        return loss, {
            "loss": loss,
        }

    new_trm, info = update_by_loss_grad(trm, loss_fn, has_aux=True)
    return rng, new_trm, info


@partial(jax.jit, static_argnames=(
'obs_dim', 'n_times_update', 'micro_batch_size', 'weighted_loss', 'input_discrete', 'target_discrete'))
def update_trm_n_times(
        rng: jax.random.PRNGKey,
        trm: TrainState,
        inputs: jnp.ndarray,
        obs_act_indicator: jnp.ndarray,
        padding_masks: jnp.ndarray,
        obs_dim: int,
        n_times_update: int,
        micro_batch_size: int,
        support: jnp.ndarray,
        sigma: jnp.ndarray,
        weighted_loss: bool,
        input_discrete: str = 'gauss',
        target_discrete: str = 'onehot',
):
    for i in range(n_times_update):
        micro_inputs = inputs[i * micro_batch_size: (i + 1) * micro_batch_size]
        micro_masks = padding_masks[i * micro_batch_size: (i + 1) * micro_batch_size]
        micro_obs_act_indicator = obs_act_indicator[i * micro_batch_size: (i + 1) * micro_batch_size]

        rng, trm, info = update_trm(
            rng, trm, micro_inputs, micro_obs_act_indicator, micro_masks, obs_dim, support, sigma,
            weighted_loss, input_discrete, target_discrete)
    return rng, trm, info


@partial(jax.jit, static_argnames=('obs_dim', 'weighted_loss', 'input_discrete', 'target_discrete'))
def eval_trm(
        rng: jax.random.PRNGKey,
        trm: TrainState,
        inputs: jnp.ndarray,
        obs_act_indicator: jnp.ndarray,
        padding_masks: jnp.ndarray,
        obs_dim: int,
        support: jnp.ndarray,
        sigma: jnp.ndarray,
        weighted_loss: bool,
        input_discrete: str = 'gauss',
        target_discrete: str = 'onehot',
):
    rng, key = jax.random.split(rng)
    key, variate_key = jax.random.split(key)

    inputs_probs = transform(input_discrete, inputs, support, sigma)
    targets_probs = transform(target_discrete, inputs, support, sigma)
    targets_probs = targets_probs.reshape((targets_probs.shape[0], targets_probs.shape[1] * targets_probs.shape[2], targets_probs.shape[3]))

    params = trm.params
    pred = trm.apply_fn(params, inputs_probs, obs_act_indicator,
                        padding_masks, training=False, variate_key=variate_key, rngs={'dropout': key})

    B, T, M, D = pred.shape
    pred_prob = jax.nn.softmax(pred)
    pred_prob = pred_prob.reshape((pred_prob.shape[0], pred_prob.shape[1] * pred_prob.shape[2], pred_prob.shape[3]))[:, :-1]
    pred_prob = jnp.concat([jnp.zeros((pred_prob.shape[0], 1, pred_prob.shape[2])), pred_prob], axis=1).reshape(B, T, M, D)
    pred_values = transform_from_probs(pred_prob, support)
    pred_values = pred_values.reshape((pred_values.shape[0], pred_values.shape[1] * pred_values.shape[2]))[:, 1:]
    target_values = inputs.reshape((inputs.shape[0], inputs.shape[1] * inputs.shape[2]))[:, 1:]

    pred = pred.reshape((pred.shape[0], pred.shape[1] * pred.shape[2], pred.shape[3]))
    loss = optax.softmax_cross_entropy(pred[:, :-1], targets_probs[:, 1:])
    loss = jnp.concat([jnp.zeros((loss.shape[0], 1)), loss], axis=1)
    loss = loss.reshape(B, T, M)
    loss = loss[:, :, :obs_dim + 1]
    if weighted_loss:
        weight = jnp.maximum(support[:, -1] - support[:, 0], 0.1)[:obs_dim + 1]
        weight = weight / (weight.sum() + 1e-6)
        loss = (weight * loss).sum(-1)
    else:
        loss = loss.mean(-1)
    loss = (padding_masks[:, -2: -1] * loss[:, -1:]).sum() / padding_masks[:, -2: -1].sum()

    mse = jnp.square(pred_values - target_values)
    mse = jnp.concat([jnp.zeros((mse.shape[0], 1)), mse], axis=1)
    mse = mse.reshape(B, T, M)
    mse = mse[:, :, :obs_dim + 1]
    obs_mse = mse[:, :, :obs_dim]
    rew_mse = mse[:, :, obs_dim]
    mse = mse.mean(-1)
    mse = (padding_masks[:, -2: -1] * mse[:, -1:]).sum() / (padding_masks[:, -2: -1].sum() + 1e-6)

    obs_mse = (padding_masks[:, -2: -1] * obs_mse.mean(-1)[:, -1:]).sum() / padding_masks[:, -2: -1].sum()
    rew_mse = (padding_masks[:, -2: -1] * rew_mse[:, -1:]).sum() / padding_masks[:, -2: -1].sum()

    abs_err = jnp.abs(pred_values - target_values)
    abs_err = jnp.concat([jnp.zeros((abs_err.shape[0], 1)), abs_err], axis=1)
    abs_err = abs_err.reshape(B, T, M)[:, :, :obs_dim + 1]
    obs_abs_err = abs_err[:, :, :obs_dim]
    abs_err = (padding_masks[:, -2: -1] * abs_err.mean(-1)[:, -1:]).sum() / padding_masks[:, -2: -1].sum()
    obs_abs_err = (padding_masks[:, -2: -1] * obs_abs_err.mean(-1)[:, -1:]).sum() / padding_masks[:, -2: -1].sum()

    return rng, {"loss": loss,
                 "mse": mse, "obs_mse": obs_mse, "rew_mse": rew_mse,
                 "abs_err": abs_err, "obs_abs_err": obs_abs_err}


@partial(jax.jit, static_argnames=('input_discrete', 'target_discrete'))
def update_trm_for_pretrain1(
        rng: jax.random.PRNGKey,
        trm: TrainState,
        inputs: jnp.ndarray,
        obs_act_indicator: jnp.ndarray,
        padding_masks: jnp.ndarray,
        variate_masks: jnp.ndarray,
        obs_select: jnp.ndarray,
        support: jnp.ndarray,
        sigma: jnp.ndarray,
        input_discrete: str = 'gauss',
        target_discrete: str = 'onehot'
):
    """
    inputs [B, T, M]
    padding_masks: [B, T]
    """
    rng, key = jax.random.split(rng)
    key, variate_key = jax.random.split(key)
    targets = transform(target_discrete, inputs, support, sigma)
    targets = targets.reshape((targets.shape[0], targets.shape[1] * targets.shape[2], targets.shape[3])) # for 1d tsm
    inputs = transform(input_discrete, inputs, support, sigma)
    # shift padding_masks to the right by 1
    padding_masks = jnp.concat([padding_masks[..., 0][..., None], padding_masks], axis=1)

    def loss_fn(params: Params) -> jnp.ndarray:
        pred = trm.apply_fn(params, inputs, obs_act_indicator, padding_masks, variate_masks,
                            training=True, variate_key=variate_key, rngs={'dropout': key},
                            method=TDMTransformer.call_variate_mask)
        B, T, M, D = pred.shape
        pred = pred.reshape((pred.shape[0], pred.shape[1] * pred.shape[2], pred.shape[3]))
        pred, target = pred[:, :-1], targets[:, 1:]
        loss = optax.softmax_cross_entropy(pred, target)
        loss = jnp.concat([jnp.zeros((loss.shape[0], 1)), loss], axis=1)
        loss = loss.reshape(B, T, M)
        loss = (loss * obs_select).sum(-1) / (obs_select.sum(-1) + 1e-6)
        loss = (padding_masks[:, :-1] * loss).sum() / (padding_masks[:, :-1].sum() + 1e-6)
        return loss, {
            "loss": loss,
        }

    new_trm, info = update_by_loss_grad(trm, loss_fn, has_aux=True)
    return rng, new_trm, info


@partial(jax.jit, static_argnames=('input_discrete', 'target_discrete'))
def update_trm_for_pretrain2(
        rng: jax.random.PRNGKey,
        trm: TrainState,
        inputs: jnp.ndarray,
        obs_act_indicator: jnp.ndarray,
        padding_masks: jnp.ndarray,
        variate_masks: jnp.ndarray,
        obs_select: jnp.ndarray,
        support: jnp.ndarray,
        sigma: jnp.ndarray,
        input_discrete: str = 'gauss',
        target_discrete: str = 'onehot'
):
    """
    inputs [B, T, M]
    padding_masks: [B, T]
    """
    rng, key = jax.random.split(rng)
    key, variate_key = jax.random.split(key)
    targets = transform(target_discrete, inputs, support, sigma)
    targets = targets.reshape((targets.shape[0], targets.shape[1] * targets.shape[2], targets.shape[3]))
    inputs = transform(input_discrete, inputs, support, sigma)
    # shift padding_masks to the right by 1
    padding_masks = jnp.concat([padding_masks[..., 0][..., None], padding_masks], axis=1)

    def loss_fn(params: Params) -> jnp.ndarray:
        pred = trm.apply_fn(params, inputs, obs_act_indicator, padding_masks, variate_masks,
                            training=True, variate_key=variate_key, rngs={'dropout': key},
                            method=TDMTransformer.call_variate_mask)
        B, T, M, D = pred.shape
        pred = pred.reshape((pred.shape[0], pred.shape[1] * pred.shape[2], pred.shape[3]))
        pred, target = pred[:, :-1], targets[:, 1:]
        loss = optax.softmax_cross_entropy(pred, target)
        loss = jnp.concat([jnp.zeros((loss.shape[0], 1)), loss], axis=1)
        loss = loss.reshape(B, T, M)
        loss = (loss * obs_select).sum(-1) / (obs_select.sum(-1) + 1e-6)
        loss = (padding_masks[:, :-1] * loss).sum() / (padding_masks[:, :-1].sum() + 1e-6)
        return loss, {
            "loss": loss,
        }

    new_trm, info = update_by_loss_grad(trm, loss_fn, has_aux=True)
    return rng, new_trm, info


@partial(jax.jit, static_argnames=('input_discrete', 'target_discrete'))
def update_trm_for_pretrain3(
        rng: jax.random.PRNGKey,
        trm: TrainState,
        inputs: jnp.ndarray,
        obs_act_indicator: jnp.ndarray,
        padding_masks: jnp.ndarray,
        variate_masks: jnp.ndarray,
        obs_select: jnp.ndarray,
        support: jnp.ndarray,
        sigma: jnp.ndarray,
        input_discrete: str = 'gauss',
        target_discrete: str = 'onehot'
):
    """
    inputs [B, T, M]
    padding_masks: [B, T]
    """
    rng, key = jax.random.split(rng)
    key, variate_key = jax.random.split(key)
    targets = transform(target_discrete, inputs, support, sigma)
    targets = targets.reshape((targets.shape[0], targets.shape[1] * targets.shape[2], targets.shape[3]))
    inputs = transform(input_discrete, inputs, support, sigma)
    # shift padding_masks to the right by 1
    padding_masks = jnp.concat([padding_masks[..., 0][..., None], padding_masks], axis=1)

    def loss_fn(params: Params) -> jnp.ndarray:
        pred = trm.apply_fn(params, inputs, obs_act_indicator, padding_masks, variate_masks,
                            training=True, variate_key=variate_key, rngs={'dropout': key},
                            method=TDMTransformer.call_variate_mask)
        B, T, M, D = pred.shape
        pred = pred.reshape((pred.shape[0], pred.shape[1] * pred.shape[2], pred.shape[3]))
        pred, target = pred[:, :-1], targets[:, 1:]
        loss = optax.softmax_cross_entropy(pred, target)
        loss = jnp.concat([jnp.zeros((loss.shape[0], 1)), loss], axis=1)
        loss = loss.reshape(B, T, M)
        loss = (loss * obs_select).sum(-1) / (obs_select.sum(-1) + 1e-6)
        loss = (padding_masks[:, :-1] * loss).sum() / (padding_masks[:, :-1].sum() + 1e-6)
        return loss, {
            "loss": loss,
        }

    new_trm, info = update_by_loss_grad(trm, loss_fn, has_aux=True)
    return rng, new_trm, info


@partial(jax.jit, static_argnames=('input_discrete', 'target_discrete'))
def eval_trm_for_pretrain(
        rng: jax.random.PRNGKey,
        trm: TrainState,
        inputs: jnp.ndarray,
        obs_act_indicator: jnp.ndarray,
        padding_masks: jnp.ndarray,
        variate_masks: jnp.ndarray,
        obs_select: jnp.ndarray,
        support: jnp.ndarray,
        sigma: jnp.ndarray,
        input_discrete: str = 'gauss',
        target_discrete: str = 'onehot'
):
    rng, key = jax.random.split(rng)
    key, variate_key = jax.random.split(key)

    inputs_probs = transform(input_discrete, inputs, support, sigma)
    targets_probs = transform(target_discrete, inputs, support, sigma)
    targets_probs = targets_probs.reshape((targets_probs.shape[0], targets_probs.shape[1] * targets_probs.shape[2], targets_probs.shape[3]))

    pred = trm.apply_fn(trm.params, inputs_probs, obs_act_indicator, padding_masks, variate_masks,
                        training=False, variate_key=variate_key, rngs={'dropout': key},
                        method=TDMTransformer.call_variate_mask)
    B, T, M, D = pred.shape
    pred_prob = jax.nn.softmax(pred)
    pred_values = transform_from_probs(pred_prob, support)
    pred_values = pred_values.reshape((pred_values.shape[0], pred_values.shape[1] * pred_values.shape[2]))[:, :-1]
    target_values = inputs.reshape((inputs.shape[0], inputs.shape[1] * inputs.shape[2]))[:, 1:]

    mse = jnp.square(pred_values - target_values)
    mse = jnp.concat([jnp.zeros((mse.shape[0], 1)), mse], axis=1)
    mse = mse.reshape(B, T, M)
    mse = (mse * obs_select).sum(-1) / (obs_select.sum(-1) + 1e-6)
    mse = (padding_masks[:, -2: -1] * mse[:, -1:]).sum() / (padding_masks[:, -2: -1].sum() + 1e-6)

    pred = pred.reshape((pred.shape[0], pred.shape[1] * pred.shape[2], pred.shape[3]))
    loss = optax.softmax_cross_entropy(pred[:, :-1], targets_probs[:, 1:])
    loss = jnp.concat([jnp.zeros((loss.shape[0], 1)), loss], axis=1)
    loss = loss.reshape(B, T, M)
    loss = (loss * obs_select).sum(-1) / (obs_select.sum(-1) + 1e-6)
    loss = (padding_masks[:, -2: -1] * loss[:, -1:]).sum() / (padding_masks[:, -2: -1].sum() + 1e-6)

    return rng, {"loss": loss, "mse": mse}


class TDM_Dynamics(object):
    def __init__(self, config: MOPOConfig, max_values: np.ndarray, min_values: np.ndarray,
                 knn_data: Optional[np.ndarray] = None):
        self.config = config
        self.rng = jax.random.PRNGKey(config.seed)

        dummy_inputs = jnp.zeros((1, config.history_length, config.obs_dim + 1 +
                                  config.act_dim, config.uniform_bin), jnp.int32)
        dummy_pad_mask = jnp.ones((1, config.history_length))
        self.obs_act_indicator = jnp.concatenate([
            jnp.zeros((1, 1, config.obs_dim + 1), jnp.int32),
            jnp.ones((1, 1, config.act_dim), jnp.int32),
        ], axis=-1)  # [1, 1, M]

        max_timestep = config.history_length * 512
        rng = jax.random.PRNGKey(config.seed)
        rng, init_key = jax.random.split(rng, 2)
        trm_model = TDMTransformer(
            vocab_size=config.uniform_bin,
            n_blocks=config.n_blocks,
            h_dim=config.embed_dim,
            n_heads=config.n_heads,
            drop_p=config.dropout_p,
            max_timestep=max_timestep,
            use_variate_embed=config.trm_variate_embed,
            shuffle_variate=config.trm_shuffle_variate,
            mask_ratio=config.trm_mask_ratio,
        )
        scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.trm_lr,
            warmup_steps=config.trm_warmup_steps,
            decay_steps=config.trm_max_steps,
        )

        init_key, variate_key = jax.random.split(init_key)
        self.trm = TrainState.create(
            apply_fn=trm_model.apply,
            params=trm_model.init(init_key, dummy_inputs, self.obs_act_indicator,
                                  dummy_pad_mask, training=False, variate_key=variate_key),
            tx=optax.chain(
                optax.clip_by_global_norm(config.trm_clip_grads),
                optax.scale_by_adam(),
                optax.add_decayed_weights(config.trm_weight_decay),
                optax.scale_by_schedule(scheduler),
                optax.scale(-1),
            ),
        )
        self.empty_cache = trm_model.get_empty_cache

        self.terminal_fn = get_termination_fn(task=config.env_name)
        self.uncertainty_mode = config.penalty_mode or "entropy"
        self.uncertainty_temp = config.uncertainty_temp
        self.penalty_coef = config.penalty_coef

        self.max_values = max_values + 1e-5
        self.min_values = min_values - 1e-5
        self.support = jnp.linspace(self.min_values, self.max_values,
                                    config.uniform_bin + 1, dtype=jnp.float32).transpose()
        self.relative_sigma = 0.75  # TODO: magic number
        self.sigma = (self.max_values - self.min_values) / config.uniform_bin * self.relative_sigma

        if knn_data is not None:
            self.kd_tree = KDTree(knn_data)

    def get_params(self):
        return self.trm.params

    def step(
            self,
            rng: jax.random.PRNGKey,
            inputs: jnp.ndarray,
            padding_mask: jnp.ndarray,
            caches: Optional[List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]] = None,
    ):
        """
        history: [B, T, M]
        padding_mask: [B, T]
        """
        obs = inputs[:, -1, :self.config.obs_dim]
        action = inputs[:, -1, self.config.obs_dim + 1:]

        inputs = jnp.clip(inputs, self.min_values, self.max_values)
        inputs = transform_to_probs(inputs, self.support, self.sigma)
        # inputs = transform_to_onehot(inputs, self.support, self.sigma)
        obs_act_indicator = np.tile(self.obs_act_indicator, (inputs.shape[0], inputs.shape[1], 1))

        rng, key = jax.random.split(rng)
        if caches is None:
            pred = self.trm.apply_fn(self.get_params(), inputs, obs_act_indicator,
                                     padding_mask, training=False, rngs={'dropout': key})
        else:
            pred, updated_caches = self.trm.apply_fn(self.get_params(), inputs, obs_act_indicator,
                                                     padding_mask, caches=caches, training=False,

                                                     # rngs={'dropout': key}  # TODO: need fix?
                                                     method=TDMTransformer.call_kv_cache
                                                     )
        pred = pred[:, -1, :self.config.obs_dim + 1]
        pred_prob = jax.nn.softmax(pred)
        samples = transform_from_probs(pred_prob, self.support[:self.config.obs_dim + 1])

        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward
        pred_prob = jax.nn.softmax(pred / self.uncertainty_temp)
        info["confidence"] = pred_prob.max(-1)

        if self.penalty_coef:
            if self.uncertainty_mode == "entropy":
                ent = -jnp.sum(pred_prob * jnp.log(pred_prob + 1e-8), axis=-1)
                ent = jnp.maximum(ent + jnp.log(0.5), 0)
                penalty = ent.mean(-1)
            elif self.uncertainty_mode == "entropy_max":
                ent = -jnp.sum(pred_prob * jnp.log(pred_prob + 1e-8), axis=-1)
                ent = jnp.maximum(ent + jnp.log(0.5), 0)
                penalty = ent.max(-1)
            elif self.uncertainty_mode == "entropy_max_noclip":
                ent = -jnp.sum(pred_prob * jnp.log(pred_prob + 1e-8), axis=-1)
                penalty = ent.max(-1)
            elif self.uncertainty_mode == "confidence":
                penalty = 1 - pred_prob.max(-1).mean(-1)
            elif self.uncertainty_mode == "pred_std":
                raise NotImplementedError
            elif self.uncertainty_mode == "knn":
                obs_act = jnp.concatenate([obs, action], axis=-1)
                _, idx = self.kd_tree.query(obs_act, k=1, workers=-1)
                neighbor = self.kd_tree.data[idx]
                dist = jnp.linalg.norm(neighbor - obs_act, axis=-1)
                penalty = dist
            else:
                raise NotImplementedError
            penalty = jnp.expand_dims(penalty, 1).astype(jnp.float32)
            assert penalty.shape == reward.shape
            reward = reward - self.penalty_coef * penalty
            info["penalty"] = penalty

        if caches is None:
            return rng, next_obs, reward, terminal, info
        else:
            return rng, next_obs, reward, terminal, info, updated_caches

    def train(
            self,
            data: HistoryBatch,
            logger: Logger,
            max_epochs: Optional[int] = None,
            max_epochs_since_update: int = 5,
            holdout_eps_len: Optional[int] = None,
            holdout_size: int = 1000,
            test_data: Optional[HistoryBatch] = None,
            epoch_start: int = 0,
    ) -> None:
        inputs = data.histories
        masks = data.history_masks

        # data split
        data_size = inputs.shape[0]
        train_size = data_size - holdout_size
        train_holdout_size = data_size
        if holdout_eps_len:
            eps_len = holdout_eps_len
            indices = np.arange(data_size // eps_len)
            np.random.shuffle(indices)
            train_indices = np.array([i * eps_len + j for i in indices[:train_size // eps_len] for j in range(eps_len)])
            holdout_indices = np.array(
                [i * eps_len + j for i in indices[train_size // eps_len: train_holdout_size // eps_len]
                 for j in range(eps_len)])
        else:
            indices = np.arange(data_size)
            np.random.shuffle(indices)
            train_indices = indices[:train_size]
            holdout_indices = indices[train_size:train_holdout_size]
        print("holdout_indices", holdout_indices[:10])

        train_inputs, train_masks = inputs[train_indices], masks[train_indices]
        holdout_inputs, holdout_masks = inputs[holdout_indices], masks[holdout_indices]
        if test_data is not None:
            test_inputs, test_masks = test_data.histories, test_data.history_masks
            test_indices = np.arange(test_inputs.shape[0])
            # set independent test set
            rand_state = np.random.get_state()
            np.random.seed(0)
            np.random.shuffle(test_indices)
            np.random.set_state(rand_state)

            test_indices = test_indices[:5000]
            test_inputs, test_masks = test_inputs[test_indices], test_masks[test_indices]

        holdout_info = self.validate(holdout_inputs, holdout_masks, self.config.trm_batch_size)
        for k, v in holdout_info.items():
            logger.logkv(f"dynamics_eval/holdout_{k}", v)

        # train
        epoch = epoch_start
        if epoch_start != 0:
            print(f"Resuming training from epoch {epoch_start}")
        cnt = 0
        logger.log("Training dynamics:")
        start_time = time.time()
        best_hold_mse = 1e10
        while True:
            epoch += 1

            train_indices = np.arange(train_inputs.shape[0])
            np.random.shuffle(train_indices)
            info = self.learn(train_inputs[train_indices], train_masks[train_indices],
                              self.config.trm_batch_size, self.config.trm_epoch_steps,)

            for key, value in info.items():
                logger.logkv("dynamics_train/" + key, value)
            holdout_info = self.validate(holdout_inputs, holdout_masks, self.config.trm_batch_size,)
            holdout_mse = holdout_info["loss"]
            for k, v in holdout_info.items():
                logger.logkv(f"dynamics_eval/holdout_{k}", v)
            if test_data is not None:
                test_info = self.validate(test_inputs, test_masks, self.config.trm_batch_size,)
                for k, v in test_info.items():
                    logger.logkv(f"dynamics_eval/test_{k}", v)
            logger.logkv("time/model_epoch", time.time() - start_time)
            start_time = time.time()
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            self.update_save()
            self.save(logger.model_dir, suffix=epoch)

            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        self.load_save()
        self.save(logger.model_dir)
        logger.log(f"holdout mse: {best_hold_mse}")

    def learn(
            self,
            inputs: np.ndarray,
            masks: np.ndarray,
            batch_size: int = 256,
            max_steps: Optional[int] = None,
    ) -> float:
        train_size = inputs.shape[0]
        trm_jitted_updates = self.config.trm_jitted_updates or 1
        batch_size = batch_size * trm_jitted_updates
        train_iters = train_size // batch_size  # drop last batch
        if max_steps is not None:
            train_iters = min(train_iters, max_steps // trm_jitted_updates)
        obs_act_indicator = None
        info_sum = None

        bar = trange(train_iters, desc="training dynamics")
        for batch_num in bar:
            inputs_batch = inputs[batch_num * batch_size: (batch_num + 1) * batch_size]
            masks_batch = masks[batch_num * batch_size: (batch_num + 1) * batch_size]
            if obs_act_indicator is None or inputs_batch.shape[0] != obs_act_indicator.shape[0]:
                obs_act_indicator = jnp.tile(self.obs_act_indicator, (inputs_batch.shape[0], inputs_batch.shape[1], 1))

            if trm_jitted_updates > 1:
                self.rng, self.trm, info = update_trm_n_times(
                    self.rng, self.trm,
                    inputs_batch, obs_act_indicator, masks_batch,
                    self.config.obs_dim, trm_jitted_updates, batch_size // trm_jitted_updates,
                    self.support, self.sigma,
                    self.config.trm_weighted_ce_loss,
                    input_discrete=self.config.trm_input_discrete,
                    target_discrete=self.config.trm_target_discrete,
                )
            else:
                self.rng, self.trm, info = update_trm(
                    self.rng, self.trm,
                    inputs_batch, obs_act_indicator, masks_batch,
                    self.config.obs_dim, self.support, self.sigma,
                    self.config.trm_weighted_ce_loss,
                    input_discrete=self.config.trm_input_discrete,
                    target_discrete=self.config.trm_target_discrete,
                )
            info_sum = info if info_sum is None else jax.tree.map(lambda a, b: a + b, info_sum, info)
            bar.set_postfix({k: v for k, v in info.items()})

        return jax.tree.map(lambda x: x / train_iters, info_sum)

    def validate(
            self,
            inputs: np.ndarray,
            masks: np.ndarray,
            batch_size: int = 256,
    ) -> List[float]:
        eval_size = inputs.shape[0]
        eval_iters = int(np.ceil(eval_size / batch_size))
        obs_act_indicator = None
        info_sum = None

        for batch_num in range(eval_iters):
            inputs_batch = inputs[batch_num * batch_size: (batch_num + 1) * batch_size]
            masks_batch = masks[batch_num * batch_size: (batch_num + 1) * batch_size]
            if obs_act_indicator is None or inputs_batch.shape[0] != obs_act_indicator.shape[0]:
                obs_act_indicator = np.tile(self.obs_act_indicator, (inputs_batch.shape[0], inputs_batch.shape[1], 1))

            self.rng, info = eval_trm(
                self.rng, self.trm,
                inputs_batch, obs_act_indicator, masks_batch,
                self.config.obs_dim, self.support, self.sigma,
                self.config.trm_weighted_ce_loss,
                input_discrete=self.config.trm_input_discrete,
                target_discrete=self.config.trm_target_discrete,
            )
            info_sum = info if info_sum is None else jax.tree.map(lambda a, b: a + b, info_sum, info)

        return jax.tree.map(lambda x: x / eval_iters, info_sum)

    def train_with_dataloader(
            self,
            # train_loader: DataLoader,
            # val_loader: DataLoader,
            train_loader,
            val_loader,
            logger: Logger,
            start_updated_iters: int = 0,
    ) -> None:
        # train
        logger.log("Training dynamics:")
        start_time = time.time()
        max_iters, val_iters, val_interval, log_interval = \
            self.config.pt_max_iters, \
                self.config.pt_val_iters, \
                self.config.pt_val_interval, \
                self.config.pt_log_interval

        def preprocess_batch(batch):
            history, obs_act_indicator, history_mask = batch['history'], batch['obs_act_indicator'], batch[
                'history_mask']
            act_dim = obs_act_indicator[0, 0].sum().item()
            obs_dim = obs_act_indicator.shape[-1] - act_dim - 1

            max_num_variates = ((history.shape[-1] - 1) // 31 + 1) * 31
            max_values = jnp.ones(max_num_variates)
            min_values = jnp.zeros(max_num_variates)
            support = jnp.linspace(min_values, max_values, self.config.uniform_bin +
                                   1, dtype=jnp.float32).transpose()
            sigma = (max_values - min_values) / self.config.uniform_bin * 0.75  # TODO: magic number

            padding_variate_dim = max_num_variates - obs_dim - 1 - act_dim
            history = jnp.concatenate([history, jnp.zeros(history.shape[:-1] + (padding_variate_dim,))], axis=-1)
            obs_select = jnp.concat([1.0 - obs_act_indicator,
                                     jnp.zeros(obs_act_indicator.shape[:-1] + (padding_variate_dim,), dtype=jnp.int8)],
                                    axis=-1)
            obs_act_indicator = jnp.concat(
                [obs_act_indicator, jnp.zeros(obs_act_indicator.shape[:-1] + (padding_variate_dim,), dtype=jnp.int8)],
                axis=-1)
            variate_masks = jnp.concatenate([jnp.ones((history.shape[0], obs_dim + 1 + act_dim)),
                                             jnp.zeros((history.shape[0], padding_variate_dim))], axis=-1)

            return history, obs_act_indicator, history_mask, variate_masks, obs_select, support, sigma

        updated_iters = start_updated_iters
        bar = trange(max_iters - updated_iters, desc="training dynamics")
        for batch in train_loader:
            history, obs_act_indicator, history_mask, variate_masks, obs_select, support, sigma = preprocess_batch(
                batch)
            if batch['history'].shape[-1] <= 31:
                self.rng, self.trm, info = update_trm_for_pretrain1(
                    self.rng, self.trm,
                    history, obs_act_indicator, history_mask, variate_masks, obs_select, support, sigma,
                    input_discrete=self.config.trm_input_discrete,
                    target_discrete=self.config.trm_target_discrete,
                )
            elif batch['history'].shape[-1] <= 62:
                self.rng, self.trm, info = update_trm_for_pretrain2(
                    self.rng, self.trm,
                    history, obs_act_indicator, history_mask, variate_masks, obs_select, support, sigma,
                    input_discrete=self.config.trm_input_discrete,
                    target_discrete=self.config.trm_target_discrete,
                )
            else:
                self.rng, self.trm, info = update_trm_for_pretrain3(
                    self.rng, self.trm,
                    history, obs_act_indicator, history_mask, variate_masks, obs_select, support, sigma,
                    input_discrete=self.config.trm_input_discrete,
                    target_discrete=self.config.trm_target_discrete,
                )

            if updated_iters % log_interval == 0:
                for key, value in info.items():
                    logger.logkv("dynamics_train/" + key, value)

            if updated_iters % val_interval == 0:
                def validate(val_name, one_val_loader):
                    info_sum = None
                    for i, val_batch in enumerate(one_val_loader):
                        history, obs_act_indicator, history_mask, variate_masks, obs_select, support, sigma = preprocess_batch(
                            val_batch)
                        self.rng, info = eval_trm_for_pretrain(
                            self.rng, self.trm,
                            history, obs_act_indicator, history_mask, variate_masks, obs_select, support, sigma,
                            input_discrete=self.config.trm_input_discrete,
                            target_discrete=self.config.trm_target_discrete,
                        )
                        info_sum = info if info_sum is None else jax.tree.map(
                            lambda a, b: a + b, info_sum, info)
                        # bar.update(1)
                        if i >= val_iters:
                            break
                    holdout_info = jax.tree.map(lambda x: x / val_iters, info_sum)

                    for k, v in holdout_info.items():
                        logger.logkv(f"dynamics_eval/{val_name}holdout_{k}", v)
                    logger.logkv("time/model_iters", time.time() - start_time)

                if type(val_loader) == dict:
                    for k, v in val_loader.items():
                        validate(k + '_', v)
                else:
                    validate('', val_loader)

                start_time = time.time()
                self.save(logger.model_dir, suffix=updated_iters)

            logger.set_timestep(updated_iters)
            logger.dumpkvs(exclude=["policy_training_progress"])

            updated_iters += 1
            bar.update(1)
            bar.set_postfix(loss=info["loss"])
            if updated_iters >= max_iters:
                break

    def load_save(self):
        self.trm = self.trm.replace(params=self.saved_params)

    def update_save(self):
        self.saved_params = deepcopy(self.trm.params)

    def save(self, save_path: str, suffix='') -> None:
        params = self.trm.params

        pickle.dump({"params": params, "max_values": self.max_values, "min_values": self.min_values},
                    open(os.path.join(save_path, f"trm_dynamics{suffix}.pkl"), 'wb'))

    def load(self, load_path: str) -> None:
        if load_path.endswith('.pkl'):
            ckpt = pickle.load(open(load_path, 'rb'))
        else:
            ckpt = pickle.load(open(os.path.join(load_path, "trm_dynamics.pkl"), 'rb'))
        self.trm = self.trm.replace(params=ckpt["params"])
        self.max_values = ckpt["max_values"]
        self.min_values = ckpt["min_values"]
        self.support = jnp.linspace(self.min_values, self.max_values,
                                    self.config.uniform_bin + 1, dtype=jnp.float32).transpose()
        self.relative_sigma = 0.75  # TODO: magic number
        self.sigma = (self.max_values - self.min_values) / self.config.uniform_bin * self.relative_sigma

    def load_weights(self, load_path: str) -> None:
        if load_path.endswith('.pkl'):
            ckpt = pickle.load(open(load_path, 'rb'))
        else:
            ckpt = pickle.load(open(os.path.join(load_path, "trm_dynamics.pkl"), 'rb'))
        self.trm = self.trm.replace(params=ckpt["params"])

    def load_max_min_values(self, max_values, min_values) -> None:
        self.max_values = max_values
        self.min_values = min_values
        self.support = jnp.linspace(self.min_values, self.max_values,
                                    self.config.uniform_bin + 1, dtype=jnp.float32).transpose()
        self.relative_sigma = 0.75
        self.sigma = (self.max_values - self.min_values) / self.config.uniform_bin * self.relative_sigma
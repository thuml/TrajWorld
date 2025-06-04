import os
import time
from functools import partial
from typing import Any, Dict, Optional, List, Tuple, Sequence
import pickle

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from tqdm import trange

from dynamics.config import MOPOConfig
from data.data import Batch
from architecture.mlp_ensemble import EnsembleDynamicsModel
from env.termination_fns import get_termination_fn
from dynamics.utils import StandardScaler, update_by_loss_grad_mlp, InfoDict, Params
from dynamics.logger import Logger

def eval_model_for_pretrain(
        dynamics: TrainState,
        inputs_batch,
        targets_batch,
        training_target_masks,
):
    mean, logvar, max_logvar, min_logvar = dynamics.apply_fn(dynamics.params, inputs_batch)
    mse = jnp.square(mean - targets_batch) * training_target_masks
    return {"eval_mse": mse.mean(), "eval_obs_mse": mse[..., :-1].mean(), "eval_rew_mse": mse[..., -1].mean()}

@partial(jax.jit, static_argnames=('logvar_loss_coef', 'dynamics_weight_decay'))
def update_model_for_pretrain(
        dynamics: TrainState,
        inputs_batch,
        targets_batch,
        training_target_masks,
        logvar_loss_coef: float,
        dynamics_weight_decay: Sequence[float],
) -> Tuple[TrainState, InfoDict]:

    def model_loss_fn(model_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        mean, logvar, max_logvar, min_logvar = dynamics.apply_fn(model_params, inputs_batch)
        inv_var = jnp.exp(-logvar) * training_target_masks

        logvar = logvar * training_target_masks
        train_mse = jnp.square(mean - targets_batch) * training_target_masks
        n_ensemble, batch_size, _ = train_mse.shape

        mse_loss_inv = (train_mse * inv_var).sum(axis=(1, 2)) / jnp.sum(training_target_masks) / batch_size
        var_loss = logvar.sum(axis=(1, 2)) / jnp.sum(training_target_masks) / batch_size
        loss = mse_loss_inv.sum() + var_loss.sum()
        decay_loss = EnsembleDynamicsModel.get_decay_loss(model_params['params'], dynamics_weight_decay)
        loss = loss + decay_loss
        logvar_reg = logvar_loss_coef * (max_logvar*training_target_masks).sum() - logvar_loss_coef*(training_target_masks * min_logvar).sum()
        loss = loss + logvar_reg
        return loss, {
            "model_loss": loss.mean(),
            "train_mse": train_mse.sum() / jnp.sum(training_target_masks) / batch_size / n_ensemble,
            "train_obs_mse": train_mse[..., :-1].sum() / (jnp.sum(training_target_masks) - 1) / batch_size / n_ensemble,
            "train_rew_mse": train_mse[..., -1].mean(),
            "mse_loss_inv": mse_loss_inv.sum(),
            "var_loss": var_loss.sum(),
            "decay_loss": decay_loss,
            "logvar_reg": logvar_reg,
            "max_logvar": max_logvar.mean(),
            "min_logvar": min_logvar.mean(),
        }

    new_dynamics, info = update_by_loss_grad_mlp(dynamics, model_loss_fn, True)
    return new_dynamics, info

@partial(jax.jit, static_argnames=('logvar_loss_coef', 'dynamics_weight_decay'))
def update_model(
    dynamics: TrainState,
    inputs_batch: np.ndarray, targets_batch: np.ndarray,
    logvar_loss_coef: float,
    dynamics_weight_decay: Sequence[float],
    training_target_masks: np.ndarray,
) -> Tuple[TrainState, InfoDict]:

    def model_loss_fn(model_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        mean, logvar, max_logvar, min_logvar = dynamics.apply_fn(model_params, inputs_batch)
        inv_var = jnp.exp(-logvar) * training_target_masks

        logvar = logvar * training_target_masks
        train_mse = jnp.square(mean - targets_batch) * training_target_masks
        n_ensemble, batch_size, _ = train_mse.shape

        mse_loss_inv = (train_mse * inv_var).sum(axis=(1, 2)) / np.sum(training_target_masks) / batch_size
        var_loss = logvar.sum(axis=(1, 2)) / np.sum(training_target_masks) / batch_size
        loss = mse_loss_inv.sum() + var_loss.sum()
        decay_loss = EnsembleDynamicsModel.get_decay_loss(model_params['params'], dynamics_weight_decay)
        loss = loss + decay_loss
        logvar_reg = logvar_loss_coef * (max_logvar * training_target_masks).sum() - logvar_loss_coef * (
                    training_target_masks * min_logvar).sum()
        loss = loss + logvar_reg
        return loss, {
            "model_loss": loss.mean(),
            "train_mse": train_mse.sum() / np.sum(training_target_masks) / batch_size / n_ensemble,
            "train_obs_mse": train_mse[..., :-1].sum() / (np.sum(training_target_masks) - 1) / batch_size / n_ensemble,
            "train_rew_mse": train_mse[..., -1].mean(),
            "mse_loss_inv": mse_loss_inv.sum(),
            "var_loss": var_loss.sum(),
            "decay_loss": decay_loss,
            "logvar_reg": logvar_reg,
            "max_logvar": max_logvar.mean(),
            "min_logvar": min_logvar.mean(),
        }

    new_dynamics, info = update_by_loss_grad_mlp(dynamics, model_loss_fn, True)
    return new_dynamics, info


class EnsembleDynamics(object):

    def __init__(self, config: MOPOConfig):
        self.config = config
        self.num_ensemble = config.n_ensemble
        self.num_elites = config.n_elites
        self.dynamics_weight_decay = config.dynamics_weight_decay

        observations = jnp.zeros((1, config.obs_dim))
        actions = jnp.zeros((1, config.act_dim))

        rng = jax.random.PRNGKey(config.seed)
        rng, dynamics_key = jax.random.split(rng, 2)
        dynamics_model = EnsembleDynamicsModel(
            obs_dim=config.obs_dim,
            action_dim=config.act_dim,
            hidden_dims=config.dynamics_hidden_dims,
            num_ensemble=config.n_ensemble,
        )
        self.dynamics = TrainState.create(
            apply_fn=dynamics_model.apply,
            params=dynamics_model.init(dynamics_key, jnp.concatenate([observations, actions], axis=-1)),
            tx=optax.adam(config.dynamics_lr),
        )
        self.saved_dynamics = TrainState.create(
            apply_fn=dynamics_model.apply,
            params=dynamics_model.init(dynamics_key, jnp.concatenate([observations, actions], axis=-1)),
            tx=optax.adam(config.dynamics_lr),
        )
        self.elites = jnp.array(list(range(0, self.num_elites)))

        self.terminal_fn = get_termination_fn(task=config.env_name)
        self.uncertainty_mode = "aleatoric"
        self.penalty_coef = config.penalty_coef
        self.scaler = StandardScaler()


    def validate(self, inputs: np.ndarray, targets: np.ndarray, training_target_masks: np.ndarray):
        mean, _, _, _ = self.dynamics.apply_fn(self.dynamics.params, inputs)
        mse = ((mean - targets) ** 2) * training_target_masks
        n_ensemble, batch_size, _ = mse.shape
        abserr = jnp.abs(mean - targets) * training_target_masks
        loss = mse.sum(axis=(1, 2)) / np.sum(training_target_masks) / batch_size
        val_loss = list(loss)
        val_info = {
            "holdout_obs_mse": mse[..., :-1].sum() / (np.sum(training_target_masks) - 1) / batch_size / n_ensemble,
            "holdout_rew_mse": mse[..., -1].mean(),
            "obs_mse_all":mse[..., :-1].sum(axis=(1, 2)) / (np.sum(training_target_masks) - 1) / batch_size,
            "obs_abs_err_all":abserr[..., :-1].sum(axis=(1, 2)) / (np.sum(training_target_masks) - 1) / batch_size,
            "abs_err_all":abserr.sum(axis=(1, 2)) / np.sum(training_target_masks) / batch_size,
        }
        return val_loss, val_info

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.elites = jnp.array(indexes)

    def random_elite_idxs(self, rng, batch_size: int) -> np.ndarray:
        rng, key = jax.random.split(rng)
        idxs = jax.random.choice(key, self.elites, (batch_size,))
        return rng, idxs

    def step(
        self,
        rng: jax.random.PRNGKey,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        mlp_huge: bool = False,
    ):
        if mlp_huge:
            obs_dim = obs.shape[-1]
            act_dim = action.shape[-1]

            max_obs_variates = 90 # max: 78
            max_act_variates = 30 # max: 21
            obs_padding_dim = max_obs_variates - obs_dim
            act_padding_dim = max_act_variates - act_dim

            obs = jnp.concatenate([obs, jnp.zeros((obs.shape[0], obs_padding_dim))], axis=-1)
            action = jnp.concatenate([action, jnp.zeros((obs.shape[0], act_padding_dim))], axis=-1)
            obs_act = jnp.concatenate([obs, action], axis=-1)
        else:
            obs_act = jnp.concatenate([obs, action], axis=-1)
            obs_dim = obs.shape[-1]
        obs_act = self.scaler.transform(obs_act)
        mean, logvar, _, _ = self.dynamics.apply_fn(self.dynamics.params, obs_act)
        mean, logvar = jax.lax.stop_gradient(mean), jax.lax.stop_gradient(logvar)
        mean = mean.at[..., :-1].add(obs)
        std = jnp.sqrt(jnp.exp(logvar))

        rng, key = jax.random.split(rng)
        ensemble_samples = (mean + jax.random.normal(key, mean.shape) * std)
        # ensemble_samples = mean

        num_models, batch_size, _ = ensemble_samples.shape
        rng, model_idxs = self.random_elite_idxs(rng, batch_size)
        samples = ensemble_samples[model_idxs, jnp.arange(batch_size)]

        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward

        if self.penalty_coef:
            if self.uncertainty_mode == "aleatoric":
                penalty = jnp.amax(jnp.linalg.norm(std, axis=2), axis=0)
            elif self.uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1]
                next_obs_mean = jnp.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = jnp.amax(jnp.linalg.norm(diff, axis=2), axis=0)
            elif self.uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = jnp.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            penalty = jnp.expand_dims(penalty, 1).astype(jnp.float32)
            assert penalty.shape == reward.shape
            reward = reward - self.penalty_coef * penalty
            info["penalty"] = penalty

        if mlp_huge:
            next_obs = next_obs[:, :obs_dim]

        return rng, next_obs, reward, terminal, info

    def format_samples_for_training(self, data: Batch) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obss = data.observations
        actions = data.actions
        next_obss = data.next_observations
        rewards = np.expand_dims(data.rewards, axis=-1)
        if self.config.pt_path is not None or self.config.mlp_huge:
            max_obs_variates = 90  # max: 78
            max_act_variates = 30  # max: 21
            obs_dim = obss.shape[-1]
            act_dim = actions.shape[-1]
            obs_padding_dim = max_obs_variates - obs_dim
            act_padding_dim = max_act_variates - act_dim

            input_obs_padded = np.concatenate([obss, jnp.zeros((obss.shape[0], obs_padding_dim))], axis=-1)
            input_act_padded = np.concatenate([actions, jnp.zeros((obss.shape[0], act_padding_dim))], axis=-1)
            next_obss_padded = np.concatenate([next_obss, jnp.zeros((obss.shape[0], obs_padding_dim))], axis=-1)
            delta_obss = next_obss_padded - input_obs_padded
            inputs = np.concatenate([input_obs_padded, input_act_padded], axis=-1)
            targets = np.concatenate([delta_obss, rewards], axis=-1)
            training_target_masks = np.concatenate([np.ones((obs_dim)), np.zeros((obs_padding_dim)), np.ones(1)], axis=-1)
        else:
            delta_obss = next_obss - obss
            inputs = np.concatenate((obss, actions), axis=-1)
            targets = np.concatenate((delta_obss, rewards), axis=-1)
            training_target_masks = np.ones((obss.shape[-1] + 1))
        return inputs, targets, training_target_masks

    def train(
        self,
        data: Batch,
        logger: Logger,
        max_epochs: Optional[int] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01,
        epoch_start: int = 0,
    ) -> None:
        inputs, targets, training_target_masks = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = self.config.holdout_size
        train_size = data_size - holdout_size
        # change holdout index to make it consistent with trm
        eps_len = self.config.holdout_eps_len
        indices = np.arange(data_size // eps_len)
        np.random.shuffle(indices)
        train_indices = np.array([i * eps_len + j for i in indices[:train_size // eps_len] for j in range(eps_len)])
        holdout_indices = np.array(
            [i * eps_len + j for i in indices[train_size // eps_len: data_size // eps_len]
             for j in range(eps_len)])
        # np.random.shuffle(indices)
        # train_indices = indices[:train_size]
        # holdout_indices = indices[train_size:]
        train_inputs, train_targets = inputs[train_indices], targets[train_indices]
        holdout_inputs, holdout_targets = inputs[holdout_indices], targets[holdout_indices]
        print("holdout_indices", holdout_indices[:10])

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for _ in range(self.num_ensemble)]
        train_size = train_inputs.shape[0]
        print("train_size", train_size)

        data_idxes = np.random.randint(train_size, size=[self.num_ensemble, train_size])

        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = epoch_start
        if epoch_start != 0:
            print(f"Resuming training from epoch {epoch_start}")
        cnt = 0
        logger.log("Training dynamics:")
        start_time = time.time()
        while True:
            epoch += 1
            info = self.learn(train_inputs[data_idxes], train_targets[data_idxes], training_target_masks, batch_size, logvar_loss_coef)
            for key, value in info.items():
                logger.logkv("dynamics_train/" + key, value)
            new_holdout_losses, info = self.validate(holdout_inputs, holdout_targets, training_target_masks)
            holdout_loss = (np.sort(new_holdout_losses)[:self.num_elites]).mean()
            logger.logkv("dynamics_eval/holdout_mse", holdout_loss)
            for k, v in info.items():
                if v.shape == (self.num_ensemble,):
                    continue
                logger.logkv("dynamics_eval/" + k, v)
            logger.logkv("time/model_epoch", time.time() - start_time)
            start_time = time.time()
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss

            self.save(logger.model_dir, suffix=epoch)

            if len(indexes) > 0:
                self.update_save(indexes)
                cnt = 0
            else:
                cnt += 1

            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        indexes = self.select_elites(holdout_losses)
        self.set_elites(indexes)
        self.load_save()
        self.save(logger.model_dir)
        logger.log("elites:{} , holdout loss: {}".format(
            indexes, (np.sort(holdout_losses)[:self.num_elites]).mean()))

    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        training_target_masks: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01,
    ) -> float:
        train_size = inputs.shape[1]
        train_iters = int(np.ceil(train_size / batch_size))
        info_sum = None

        for batch_num in range(train_iters):
            inputs_batch = inputs[:, batch_num * batch_size: (batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size: (batch_num + 1) * batch_size]

            self.dynamics, info = update_model(
                self.dynamics,
                inputs_batch, targets_batch,
                logvar_loss_coef, self.dynamics_weight_decay,
                training_target_masks
            )
            info_sum = info if info_sum is None else jax.tree.map(lambda a, b: a + b, info_sum, info)
        return jax.tree.map(lambda x: x / train_iters, info_sum)

    # def preprocess_batch(self, batch):
    #     history, obs_act_indicator, history_mask = batch['history'], batch['obs_act_indicator'], batch[
    #         'history_mask']
    #     assert history.shape[1] == 2, "history shape should be (batch_size, 2, obs_dim + act_dim + 1) for mlp dynamics"
    #     act_dim = obs_act_indicator[0, 0].sum().item()
    #     obs_dim = obs_act_indicator.shape[-1] - act_dim - 1
    #
    #     max_obs_variates = 90  # max: 78
    #     max_act_variates = 30  # max: 21
    #     obs_padding_dim = max_obs_variates - obs_dim
    #     act_padding_dim = max_act_variates - act_dim
    #
    #     input_obs = history[:, 0, :obs_dim]
    #     input_act = history[:, 0, obs_dim + 1:obs_dim + 1 + act_dim]
    #     input_obs_padded = jnp.concatenate([input_obs, jnp.zeros((history.shape[0], obs_padding_dim))], axis=-1)
    #     input_act_padded = jnp.concatenate([input_act, jnp.zeros((history.shape[0], act_padding_dim))], axis=-1)
    #     input_batch = jnp.concatenate([input_obs_padded, input_act_padded], axis=-1)
    #     input_batch = input_batch * 2 - 1  # normalize to [-1, 1]
    #     input_batch = input_batch.reshape(self.config.n_ensemble, self.config.batch_size, -1)
    #
    #     # target is not scaled in scratch
    #     delta_target_obs = (history[:, 1, :obs_dim] - input_obs) * 10.0
    #     target_rew = history[:, 1, obs_dim:obs_dim + 1]
    #     target_obs_padded = jnp.concatenate([delta_target_obs, jnp.zeros((history.shape[0], obs_padding_dim))], axis=-1)
    #     target_batch = jnp.concatenate([target_obs_padded, target_rew], axis=-1)
    #     target_batch = target_batch.reshape(self.config.n_ensemble, self.config.batch_size, -1)
    #
    #     training_target_masks = np.concatenate([np.ones((obs_dim)),
    #                                             np.zeros((obs_padding_dim)), np.ones(1), ], axis=-1)
    #     return input_batch, target_batch, training_target_masks

    def validate_pt(self, one_val_loader):
        info_sum = None
        mse =0.0
        loss = 0.0
        naive_mse = 0.0
        abs = 0.0
        naive_abs = 0.0
        abs_obs = 0.0
        naive_abs_obs = 0.0
        all_size = 5 # only for pendulum
        one_val_loader = one_val_loader["pendulum"] if "pendulum" in one_val_loader else one_val_loader["two_pole"]
        # for i, val_batch in enumerate(one_val_loader):
        #     eval_input_batch, eval_target_batch, eval_target_masks = self.preprocess_batch(val_batch)
        #     info = eval_model_for_pretrain(
        #         self.dynamics,
        #         eval_input_batch, eval_target_batch, eval_target_masks,
        #     )
        #     mse += info["mse"]
        #     loss += info["loss"]
        #     # naive_mse += info["naive_mse"]
        #     # abs += info["abs"]
        #     # naive_abs += info["naive_abs"]
        #     # abs_obs += info["abs_obs"]
        #     # naive_abs_obs += info["naive_abs_obs"]
        #
        #     if i > all_size:
        #         break
        mse /= all_size
        loss /= all_size
        # naive_mse /= all_size
        # abs /= all_size
        # naive_abs /= all_size
        # abs_obs /= all_size
        # naive_abs_obs /= all_size
        episode1 = np.load("/workspace/yinshaofeng/JAX-CORL/anylearn_two_pole/episode_0000026.npz")
        episode2 = np.load("/workspace/yinshaofeng/JAX-CORL/anylearn_two_pole/episode_0000027.npz")
        episode3 = np.load("/workspace/yinshaofeng/JAX-CORL/anylearn_two_pole/episode_0000026.npz")
        episode4 = np.load("/workspace/yinshaofeng/JAX-CORL/anylearn_two_pole/episode_0000027.npz")
        episode5 = np.load("/workspace/yinshaofeng/JAX-CORL/anylearn_two_pole/episode_0000026.npz")
        episode6 = np.load("/workspace/yinshaofeng/JAX-CORL/anylearn_two_pole/episode_0000027.npz")
        obs1 = episode1["observation"][:-1]
        obs2 = episode2["observation"][:-1]
        obs3 = obs1 # [:, ::-1]
        obs4 = obs2 # [:, ::-1]
        idx = np.arange(obs3.shape[1])
        while idx[0] == 0:
            np.random.shuffle(idx)
        inverse_idx = np.argsort(idx)
        obs5 = obs1 # [:, idx]
        obs6 = obs2 # [:, idx]
        act1 = episode1["action"][1:]
        act2 = episode2["action"][1:]
        act3 = episode3["action"][1:]
        act4 = episode4["action"][1:]
        act5 = episode5["action"][1:]
        act6 = episode6["action"][1:]
        r1 = episode1["reward"][1:]
        r2 = episode2["reward"][1:]
        r3 = episode3["reward"][1:]
        r4 = episode4["reward"][1:]
        r5 = episode5["reward"][1:]
        r6 = episode6["reward"][1:]
        hist1 = np.concatenate([obs1, r1, act1], axis=-1)
        hist2 = np.concatenate([obs2, r2, act2], axis=-1)
        hist3 = np.concatenate([obs3, r3, act3], axis=-1)
        hist4 = np.concatenate([obs4, r4, act4], axis=-1)
        hist5 = np.concatenate([obs5, r5, act5], axis=-1)
        hist6 = np.concatenate([obs6, r6, act6], axis=-1)

        hist1 = np.expand_dims(hist1, axis=0)
        hist2 = np.expand_dims(hist2, axis=0)
        hist3 = np.expand_dims(hist3, axis=0)
        hist4 = np.expand_dims(hist4, axis=0)
        hist5 = np.expand_dims(hist5, axis=0)
        hist6 = np.expand_dims(hist6, axis=0)

        # all_hist = np.concatenate([hist1, hist2, hist3], axis=0)
        all_hist = np.concatenate([hist1, hist2, hist3, hist4, hist5, hist6], axis=0)
        max_min = np.load("/workspace/yinshaofeng/JAX-CORL/anylearn_two_pole/max_min_values.npz")
        max_values = max_min["max"]
        min_values = max_min["min"]



        for i, val_batch in enumerate(one_val_loader):

            if i > 0:
                break

            # eval_input_batch, eval_target_batch, eval_target_masks = self.preprocess_batch(val_batch)
            act_dim = 1
            obs_dim = 8

            max_obs_variates = 90  # max: 78
            max_act_variates = 30  # max: 21
            obs_padding_dim = max_obs_variates - obs_dim
            act_padding_dim = max_act_variates - act_dim
            history = (all_hist - min_values) / (max_values - min_values)
            input_obs = history[:, :, :obs_dim]
            input_act = history[:, :, obs_dim + 1:obs_dim + 1 + act_dim]
            input_obs_padded = jnp.concatenate([input_obs, jnp.zeros((history.shape[0], 20, obs_padding_dim))], axis=-1)
            input_act_padded = jnp.concatenate([input_act, jnp.zeros((history.shape[0], 20, act_padding_dim))], axis=-1)
            input_batch = jnp.concatenate([input_obs_padded, input_act_padded], axis=-1)
            input_batch = input_batch * 2 - 1  # normalize to [-1, 1]
            input_batch = jnp.tile(input_batch[None, :], (self.config.n_ensemble, 1, 1, 1)).transpose(0, 2, 1, 3)

            rollout_len = 10
            history_len = 10
            self.rng = jax.random.PRNGKey(0)
            for j in range(rollout_len):
                mean, logvar, max_logvar, min_logvar = self.dynamics.apply_fn(self.dynamics.params, input_batch[:, history_len + j - 1])
                mean = mean.at[..., :-1].add(input_batch[:, history_len + j - 1, :, :90])
                std = jnp.sqrt(jnp.exp(logvar))

                self.rng, key = jax.random.split(self.rng)
                # ensemble_samples = (mean + jax.random.normal(key, mean.shape) * std)
                ensemble_samples = mean

                num_models, batch_size, _ = ensemble_samples.shape
                self.rng, model_idxs = self.random_elite_idxs(self.rng, batch_size)
                samples = ensemble_samples[model_idxs, jnp.arange(batch_size)]

                next_obs = samples[..., :-1]
                # reward = samples[..., -1:]
                input_batch = input_batch.at[:, history_len + j].set(jnp.concatenate([jnp.tile(next_obs, (self.config.n_ensemble, 1, 1)), input_batch[:, history_len + j,:,90:]], axis=-1))
                # set action
                # pred = pred.at[:,0,9].set(real_history[:, history_len + j, 9])
                # history = jnp.concatenate([history, pred], axis=1)
                # history_mask = jnp.concatenate([history_mask, jnp.ones((history_mask.shape[0], 1))], axis=1)

            # history = history[:, :, :len(max_values)]
            # real_history = real_history[:, :, :len(max_values)]
            # obss = history[2:4, :, :8][:, :, ::-1]
            # others = history[2:4, :, 8:]
            # hist = jnp.concatenate([obss, others], axis=-1)
            # history = history.at[2:4].set(hist)
            # obss = history[4:6, :, :8][:, :, inverse_idx]
            # others = history[4:6, :, 8:]
            # hist = jnp.concatenate([obss, others], axis=-1)
            # history = history.at[4:6].set(hist)
            #
            # obss = real_history[2:4, :, :8][:, :, ::-1]
            # others = real_history[2:4, :, 8:]
            # hist = jnp.concatenate([obss, others], axis=-1)
            # real_history = real_history.at[2:4].set(hist)
            #
            # obss = real_history[4:6, :, :8][:, :, inverse_idx]
            # others = real_history[4:6, :, 8:]
            # hist = jnp.concatenate([obss, others], axis=-1)
            # real_history = real_history.at[4:6].set(hist)

            input_batch = input_batch.mean(axis=0).transpose(1, 0, 2)
            input_batch = input_batch + 1
            input_batch = input_batch / 2.0

            input_batch = input_batch[:, :, :10]

            input_batch = input_batch * (max_values - min_values + 1e-8) + min_values
            real_history = history * (max_values - min_values + 1e-8) + min_values

        return mse, loss, naive_mse, abs, naive_abs, abs_obs, naive_abs_obs, input_batch, real_history

    def train_with_dataloader(
            self,
            train_loader,
            val_loader,
            logger: Logger,
            start_updated_iters: int = 0,
            logvar_loss_coef: float = 0.01,
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
            assert history.shape[1] == 2, "history shape should be (batch_size, 2, obs_dim + act_dim + 1) for mlp dynamics"
            act_dim = obs_act_indicator[0, 0].sum().item()
            obs_dim = obs_act_indicator.shape[-1] - act_dim - 1

            max_obs_variates = 90 # max: 78
            max_act_variates = 30 # max: 21
            obs_padding_dim = max_obs_variates - obs_dim
            act_padding_dim = max_act_variates - act_dim

            input_obs = history[:, 0, :obs_dim]
            input_act = history[:, 0, obs_dim + 1 :obs_dim + 1 + act_dim]
            input_obs_padded = jnp.concatenate([input_obs, jnp.zeros((history.shape[0], obs_padding_dim))], axis=-1)
            input_act_padded = jnp.concatenate([input_act, jnp.zeros((history.shape[0], act_padding_dim))], axis=-1)
            input_batch = jnp.concatenate([input_obs_padded, input_act_padded], axis=-1)
            input_batch = input_batch * 2 - 1 # normalize to [-1, 1]
            input_batch = input_batch.reshape(self.config.n_ensemble, self.config.batch_size, -1)

            # target is not scaled in scratch
            delta_target_obs = (history[:, 1, :obs_dim] - input_obs) * 10.0
            target_rew = history[:, 1, obs_dim:obs_dim + 1]
            target_obs_padded = jnp.concatenate([delta_target_obs, jnp.zeros((history.shape[0], obs_padding_dim))], axis=-1)
            target_batch = jnp.concatenate([target_obs_padded, target_rew], axis=-1)
            target_batch = target_batch.reshape(self.config.n_ensemble, self.config.batch_size, -1)

            training_target_masks = np.concatenate([np.ones((obs_dim)),
                                                     np.zeros((obs_padding_dim)), np.ones(1),], axis=-1)
            return input_batch, target_batch, training_target_masks

        updated_iters = start_updated_iters
        bar = trange(max_iters - updated_iters, desc="training dynamics")
        for batch in train_loader:
            input_batch, target_batch, training_target_masks = preprocess_batch(batch)
            self.dynamics, info = update_model_for_pretrain(
                self.dynamics,
                input_batch, target_batch, training_target_masks,
                logvar_loss_coef, self.dynamics_weight_decay
            )

            if updated_iters % log_interval == 0:
                for key, value in info.items():
                    logger.logkv("dynamics_train/" + key, value)

            if updated_iters % val_interval == 0:
                def validate(val_name, one_val_loader):
                    info_sum = None
                    for i, val_batch in enumerate(one_val_loader):
                        eval_input_batch, eval_target_batch, eval_target_masks = preprocess_batch(val_batch)
                        info = eval_model_for_pretrain(
                            self.dynamics,
                            eval_input_batch, eval_target_batch, eval_target_masks,
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
            bar.set_postfix(loss=info["train_mse"])
            if updated_iters >= max_iters:
                break

    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.num_elites)]
        return elites

    def load_save(self):
        self.dynamics = self.dynamics.replace(params=self.saved_dynamics.params)

    def update_save(self, indexes):
        new_saved_params = jax.tree.map(
            lambda p, tp: p.at[indexes].set(tp[indexes]) if p.shape[0] == self.num_elites else tp,
            self.saved_dynamics.params, self.dynamics.params
        )
        self.saved_dynamics = self.saved_dynamics.replace(params=new_saved_params)

    def save(self, save_path: str, suffix='') -> None:
        pickle.dump({"params": self.dynamics.params, "elites": self.elites},
                    open(os.path.join(save_path, f"mlp_dynamics{suffix}.pkl"), 'wb'))
        self.scaler.save_scaler(save_path)

    def load(self, load_path: str) -> None:
        if load_path.endswith('.pkl'):
            ckpt = pickle.load(open(load_path, 'rb'))
        else:
            ckpt = pickle.load(open(os.path.join(load_path, "mlp_dynamics.pkl"), 'rb'))
        self.dynamics = self.dynamics.replace(params=ckpt["params"])
        self.elites = ckpt["elites"]
        # self.scaler.load_scaler(load_path)

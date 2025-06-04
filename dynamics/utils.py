from typing import Callable, Dict, Tuple, Any
import os
import numpy as np

import jax
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState

InfoDict = Dict[str, float]
Params = flax.core.FrozenDict[str, Any]


def update_by_loss_grad(
    train_state: TrainState, loss_fn: Callable, has_aux: bool = True, return_grad_norm: bool = False) -> Tuple[TrainState, jnp.ndarray]:
    grad_fn = jax.grad(loss_fn, has_aux=has_aux)
    if has_aux:
        grad, aux = grad_fn(train_state.params)
    else:
        grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    if has_aux:
        if return_grad_norm:
            grad_norm = jnp.linalg.norm(jnp.array(jax.tree_util.tree_leaves(jax.tree.map(jnp.linalg.norm, grad))))
            aux['grad_norm'] = grad_norm
        return new_train_state, aux
    else:
        return new_train_state

def update_by_loss_grad_mlp(
    train_state: TrainState, loss_fn: Callable, has_aux: bool = True, return_grad_norm: bool = False
) -> Tuple[TrainState, jnp.ndarray]:
    grad_fn = jax.grad(loss_fn, has_aux=has_aux)
    if has_aux:
        grad, aux = grad_fn(train_state.params)
    else:
        grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    if has_aux:
        if return_grad_norm:
            grad_norm = jnp.linalg.norm(jnp.array(jax.tree_util.tree_leaves(jax.tree.map(jnp.linalg.norm, grad))))
            aux['grad_norm'] = grad_norm
        return new_train_state, aux
    else:
        return new_train_state

def update_by_loss_grad_rl(
    train_state: TrainState, loss_fn: Callable, has_aux: bool = True, return_grad_norm: bool = False
) -> Tuple[TrainState, jnp.ndarray]:
    grad_fn = jax.grad(loss_fn, has_aux=has_aux)
    if has_aux:
        grad, aux = grad_fn(train_state.params)
    else:
        grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    if has_aux:
        if return_grad_norm:
            grad_norm = jnp.linalg.norm(jnp.array(jax.tree_util.tree_leaves(jax.tree.map(jnp.linalg.norm, grad))))
            aux['grad_norm'] = grad_norm
        return new_train_state, aux
    else:
        return new_train_state

class StandardScaler(object):
    def __init__(self, mu=None, std=None):
        self.mu = mu
        self.std = std

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu

    def save_scaler(self, save_path):
        mu_path = os.path.join(save_path, "mu.npy")
        std_path = os.path.join(save_path, "std.npy")
        np.save(mu_path, self.mu)
        np.save(std_path, self.std)

    def load_scaler(self, load_path):
        if load_path.endswith('.pkl'):
            suffix_len = len(load_path.split("/")[-1]) + 1
            load_path = load_path[:-suffix_len]
        mu_path = os.path.join(load_path, "mu.npy")
        std_path = os.path.join(load_path, "std.npy")
        self.mu = np.load(mu_path)
        self.std = np.load(std_path)

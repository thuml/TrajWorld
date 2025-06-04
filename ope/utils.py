# import numpy as np
# import torch

import os
import random
import imageio
import gym
from tqdm import trange
import pickle

import jax
import jax.numpy as jnp
import flax

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def set_seed_everywhere(seed):
    random.seed(seed)
    print(f"Seed value: {seed}, Type: {type(seed)}")
    jax.random.PRNGKey(seed)

def snapshot_src(src, target, exclude_from):
    make_dir(target)
    os.system(f"rsync -rv --exclude-from={exclude_from} {src} {target}")

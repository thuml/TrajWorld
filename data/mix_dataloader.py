import os
import random

import numpy as np
import glob
import time
from tqdm import tqdm
import time
from typing import Iterator
from tqdm import tqdm

from jax.tree_util import tree_map
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, Sampler


class SingleHistoryDataset(Dataset):
    def __init__(self,
                 data_files,
                 min_max_values_path,
                 history_len, validate=False, val_ratio=0.01, verbose=False, in_memory=False,
                 symlog=False, data_ratio=1.0):
        self.history_len = history_len
        file_list = glob.glob(data_files)
        random.seed(0)
        random.shuffle(file_list)
        size = int(len(file_list) * data_ratio)
        size = max(8, size)
        if data_ratio < 1.0:
            file_list = file_list[: size]
        self.file_list = sorted(file_list)
        assert len(self.file_list) > 0, f"no files found in {data_files}"
        min_max_values = np.load(min_max_values_path)
        self.max_values = min_max_values['max']
        self.min_values = min_max_values['min']
        self.symlog = symlog

        if val_ratio != 0:
            if 'macaw/preprocessed/walker_dir' in data_files:
                # 43,15,12,39,2
                if validate:
                    self.file_list = [f for f in self.file_list if any([f'task{val_taskid}/' in f for val_taskid in [43]])]
                else:
                    self.file_list = [f for f in self.file_list if not any([f'task{val_taskid}/' in f for val_taskid in [43,15,12,39,2]])]
            elif 'pendulum' in data_files:
                # validation will be handled outside
                self.file_list = [f for i, f in enumerate(self.file_list)]
            else:
                val_iter = int(1 / val_ratio)
                if validate:
                    self.file_list = [f for i, f in enumerate(self.file_list) if i % val_iter == 0]
                else:
                    self.file_list = [f for i, f in enumerate(self.file_list) if i % val_iter != 0]

        if in_memory:
            self.all_files = [dict(np.load(file_path)) for file_path in tqdm(self.file_list, desc=data_files)]

        self.in_memory = in_memory
        self.size = len(self.file_list)

        if verbose:
            print(f"Loaded {len(self.file_list)} {'validation' if validate else 'training'} files from {data_files}")

            # Get example obs and act dim
            data = np.load(self.file_list[0])
            obs_dim, act_dim = data['observation'].shape[-1], data['action'].shape[-1]
            print(data_files, "example data shape: ", obs_dim, act_dim)

            if (self.max_values - self.min_values).min() < 1e-3:
                print(f"Warning: {data_files} min and max values are too close to each other")
                print(self.max_values - self.min_values)

                # Get example obs and act dim
                data = np.load(self.file_list[0])
                obs_dim, act_dim = data['observation'].shape[-1], data['action'].shape[-1]
                if (self.max_values - self.min_values)[obs_dim] == 0:
                    print("warning: rewards are all the same")
                    print(data_files, "example data shape: ", obs_dim, act_dim)
                    print((self.max_values - self.min_values)[obs_dim], data['reward'].flatten()[:10])

    def __len__(self):
        # uniform sampling, infinite length
        return int(1e10)

    def __getitem__(self, idx):
        if self.in_memory:
            data = self.all_files[np.random.choice(len(self.all_files))]
        else:
            data = dict(np.load(np.random.choice(self.file_list)))
        step_idx = np.random.randint(len(data['observation']))
        data['action'] = np.concatenate([data['action'], np.zeros((1, data['action'].shape[-1]), np.float32)], axis=0)

        # Check if the last element of the observation is 0
        while np.all(data['observation'][step_idx] == 0):
            step_idx = np.random.randint(len(data['observation']))

        # make the above faster
        obs_hist = data['observation'][max(0, step_idx - self.history_len + 1): step_idx + 1]
        hist_mask = np.concatenate([np.zeros(self.history_len - obs_hist.shape[0], np.int32),
                                   np.ones(obs_hist.shape[0], np.int32)], axis=0)
        obs_hist = np.concatenate(
            [np.zeros((self.history_len - obs_hist.shape[0], obs_hist.shape[1]), np.float32), obs_hist], axis=0)
        act_hist = data['action'][max(0, step_idx - self.history_len + 2): step_idx + 2]
        act_hist = np.concatenate(
            [np.zeros((self.history_len - act_hist.shape[0], act_hist.shape[1]), np.float32), act_hist], axis=0)
        rew_hist = data['reward'][max(0, step_idx - self.history_len + 1): step_idx + 1]
        rew_hist = np.concatenate(
            [np.zeros((self.history_len - rew_hist.shape[0], rew_hist.shape[1]), np.float32), rew_hist], axis=0)
        # assert (ref_obs_hist==obs_hist).all() and (ref_act_hist==act_hist).all() and (ref_rew_hist==rew_hist).all() and (ref_hist_mask==hist_mask).all()
        history = np.concatenate([obs_hist, rew_hist, act_hist], axis=-1)

        # Normalize
        if self.symlog:
            history = np.sign(history) * np.log(1 + np.abs(history)) / 2.5
        else:
            history = (history - self.min_values) / (self.max_values - self.min_values + 1e-8)
            history = np.clip(history, 0, 1)

        obs_dim, act_dim = obs_hist.shape[-1], act_hist.shape[-1]

        return {
            'history': history,
            'history_mask': hist_mask,
            'obs_act_indicator': np.concatenate([np.zeros((1, obs_dim + 1), np.int32), np.ones((1, act_dim), np.int32),], axis=-1),
        }


class MixedHistoryDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.size = sum([dataset.size for dataset in datasets])

    def __len__(self):
        # uniform sampling, infinite length
        return int(1e10)

    def __getitem__(self, idx):
        return self.datasets[idx].__getitem__(0)


class BatchConsistentSampler(Sampler[int]):
    def __init__(self, mixed_size: int, batch_size: int, weights=None) -> None:
        self.mixed_size = mixed_size
        self.batch_size = batch_size
        if weights is None:
            self.weights = np.ones(mixed_size) / mixed_size
        else:
            self.weights = np.array(weights) / np.sum(weights)  # Normalize weights

    def __len__(self) -> int:
        return 1e10

    def __iter__(self) -> Iterator[int]:
        while True:
            yield [np.random.choice(self.mixed_size, p=self.weights)] * self.batch_size


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


class NumpyLoader(DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)


def mix_dataloader(dataloaders, weights=None):
    # If weights are not provided, assign equal weights to all datasets
    if weights is None:
        weights = np.ones(len(dataloaders)) / len(dataloaders)
    else:
        weights = np.array(weights) / np.sum(weights)  # Normalize weights
    iters = [iter(dataloader) for dataloader in dataloaders]
    while True:
        try:
            batch = next(np.random.choice(iters, p=weights))
            yield batch
        except StopIteration:
            iters = [iter(dataloader) for dataloader in dataloaders]

def get_jat_mujoco_dataloader(batch_size, history_len, num_workers, root_path, verbose=False, in_memory=False):
    assert num_workers == 0 or batch_size % num_workers == 0, "batch_size should be divisible by num_workers"
    mix_weights = {
        'jat_mujoco': 1.0,
    }
    train_datasets, val_datasets = [], []
    train_weights = []

    """JAT MuJoCo"""
    # 50k episodes
    data_dirs_jat_mujuco = glob.glob(os.path.join(root_path, f'jat/mujoco-*'))
    filtered_data_dirs_jat_mujuco = [
        data_dir for data_dir in data_dirs_jat_mujuco
        if 'walker' not in data_dir and 'cheetah' not in data_dir and 'hopper' not in data_dir and 'ant' not in data_dir and 'standup' not in data_dir and 'humanoid' not in data_dir
    ]
    print("Filtered JAT MuJoCo data dirs: ", [
        data_dir for data_dir in data_dirs_jat_mujuco if data_dir not in filtered_data_dirs_jat_mujuco])

    def make_jat_mujoco_dataset(validate):
        mixed_dataset = [
            SingleHistoryDataset(
                os.path.join(dir, '*_eps_*.npz'),
                os.path.join(dir, 'max_min_values.npz'),
                history_len, validate=validate, val_ratio=0.01, in_memory=in_memory
            )
            for dir in filtered_data_dirs_jat_mujuco
        ]
        dims = [dataset.max_values.shape[0] for dataset in mixed_dataset]
        print(f"total size of jat mujoco {'val' if validate else 'train'} dataset: ",
              sum([dataset.size for dataset in mixed_dataset]))
        print("jat mujoco dataset dims: ", dims)

        return mixed_dataset, [mix_weights['jat_mujoco'] / len(filtered_data_dirs_jat_mujuco)] * len(filtered_data_dirs_jat_mujuco)

    datasets, weights = make_jat_mujoco_dataset(False)
    train_datasets.extend(datasets)
    train_weights.extend(weights)
    val_datasets.append(('jat_mujoco', make_jat_mujoco_dataset(True)[0]))

    """DONE"""
    if sum(train_weights) != 1.0:
        print("warning: sum of train_weights is not 1.0, sum(weights) = ", sum(train_weights))
        train_weights = [w / sum(train_weights) for w in train_weights]

    train_dataloader = NumpyLoader(MixedHistoryDataset(train_datasets), num_workers=num_workers,
                                   batch_sampler=BatchConsistentSampler(len(train_datasets), batch_size, weights=train_weights))
    val_dataloaders = {name: NumpyLoader(MixedHistoryDataset(dataset), num_workers=num_workers,
                                         batch_sampler=BatchConsistentSampler(len(dataset), batch_size)) for name, dataset in val_datasets}
    return train_dataloader, val_dataloaders

def get_final_dataloader_v2(batch_size, history_len, num_workers, root_path, verbose=False, in_memory=False, data_ratio=1.0):
    assert num_workers == 0 or batch_size % num_workers == 0, "batch_size should be divisible by num_workers"
    mix_weights = {
        'exorl': 0.75,
        'rlu': 0.05,
        'modular_rl': 0.30,
        'db1': 0.01,
        'jat_mujoco': 0.9,
        'tdmpc2': 0.9,
    }
    train_datasets, val_datasets = [], []
    train_weights = []

    """Modular RL"""
    # 37k episodes
    # 1000 steps per episode
    # 20 tasks, 20 robots
    def make_modular_rl_dataset(validate):
        envs_path = glob.glob(os.path.join(root_path, f'modular-rl/*-v2'))
        mixed_dataset = [
            SingleHistoryDataset(
                os.path.join(env_path, 'episode_*.npz'),
                os.path.join(env_path, 'max_min_values.npz'),
                history_len, validate=validate, val_ratio=0.01, in_memory=in_memory, data_ratio=data_ratio
            )
            for env_path in envs_path
        ]
        print(f"total size of modular rl {'val' if validate else 'train'} dataset: ", sum(
            [dataset.size for dataset in mixed_dataset]))
        dims = [dataset.max_values.shape[0] for dataset in mixed_dataset]
        print("modular rl dataset dims: ", dims)
        return mixed_dataset, [mix_weights['modular_rl'] / len(mixed_dataset)] * len(mixed_dataset)

    datasets, weights = make_modular_rl_dataset(False)
    train_datasets.extend(datasets)
    train_weights.extend(weights)
    val_datasets.append(('modular_rl', make_modular_rl_dataset(True)[0]))

    """TDMPC2"""
    # 690k episodes
    # 5000 steps per episode
    # 30 tasks, 24 robots
    def make_tdmpc2_dataset(validate):
        # buggy_test_id = [14, 15, 25, 18, 17， 13]
        # all_tasks = [f"task_{i}" for i in range(30)]
        # buggy_tasks = [f"task_{i}" for i in buggy_test_id]
        # dataset_tasks = [task for task in all_tasks if task not in buggy_tasks]
        mixed_dataset = [
            SingleHistoryDataset(
                os.path.join(root_path, f'tdmpc2/preprocessed_data_no_padding_new/{task}/episode_*.npz'),
                os.path.join(root_path, f'tdmpc2/preprocessed_data_no_padding_new/{task}/max_min_values.npz'),
                history_len, validate=validate, val_ratio=0.01, in_memory=in_memory, data_ratio=data_ratio
            )
            for task in [f"task_{i}" for i in range(30)]
        ]
        dims = [dataset.max_values.shape[0] for dataset in mixed_dataset]
        print(f"total size of tdmpc2 {'val' if validate else 'train'} dataset: ", sum(
            [dataset.size for dataset in mixed_dataset]))
        print("tdmpc2 dataset dims: ", dims)
        return mixed_dataset, [mix_weights['tdmpc2'] / len(mixed_dataset)] * len(mixed_dataset)

    datasets, weights = make_tdmpc2_dataset(False)
    train_datasets.extend(datasets)
    train_weights.extend(weights)
    val_datasets.append(('tdmpc2', make_tdmpc2_dataset(True)[0]))

    """ExoRL"""
    # 78 + 1 + 12
    def make_exorl_dataset(validate):
        # exorl_envs = ['cartpole', 'cheetah', 'jaco', 'quadruped', 'walker']
        # cartpole: 97k
        # jaco: 280k
        # quadruped: 80k
        # walker: 83k
        # 540k in total
        exorl_envs = ['cartpole', 'jaco', 'quadruped', 'walker']
        mixed_dataset = [
            SingleHistoryDataset(
                os.path.join(root_path, f'exorl/datasets/{env}/*/buffer/episode_*.npz'),
                os.path.join(root_path, f'exorl/datasets/{env}/max_min_values.npz'),
                history_len, validate=validate, val_ratio=0.01, in_memory=in_memory, data_ratio=data_ratio
            )
            for env in exorl_envs
        ]
        dims = [dataset.max_values.shape[0] for dataset in mixed_dataset]
        print(f"total size of exorl {'val' if validate else 'train'} dataset: ",
              sum([dataset.size for dataset in mixed_dataset]))
        print("exorl dataset dims: ", dims)
        return mixed_dataset, [mix_weights['exorl'] / len(exorl_envs)] * len(exorl_envs)

    datasets, weights = make_exorl_dataset(False)
    train_datasets.extend(datasets)
    train_weights.extend(weights)
    val_datasets.append(('exorl', make_exorl_dataset(True)[0]))

    """RL Unplugged"""
    def make_rlu_dataset(validate):
        # rl_unplugged_envs = ['cartpole', 'cheetah', 'fish', 'humanoid', 'manipulator', 'walker']
        # 6k
        # 1000 steps per episode
        # cartpole: 40
        # fish: 200
        # humanoid: 2998
        # manipulator: 2763
        # walker: 400
        rl_unplugged_envs = ['cartpole', 'fish', 'humanoid', 'manipulator', 'walker']
        mixed_dataset = [
            SingleHistoryDataset(
                os.path.join(root_path, f'rl_unplugged/preprocessed_d4rl_format/{env}/*/episode_*.npz'),
                os.path.join(root_path, f'rl_unplugged/preprocessed_d4rl_format/{env}/max_min_values.npz'),
                history_len, validate=validate, val_ratio=0.01, in_memory=in_memory, data_ratio=data_ratio
            )
            for env in rl_unplugged_envs
        ]
        dims = [dataset.max_values.shape[0] for dataset in mixed_dataset]
        print(f"total size of rl unplugged {'val' if validate else 'train'} dataset: ", sum(
            [dataset.size for dataset in mixed_dataset]))
        print("rl unplugged dataset dims: ", dims)

        return mixed_dataset, [mix_weights['rlu'] / len(rl_unplugged_envs)] * len(rl_unplugged_envs)

    datasets, weights = make_rlu_dataset(False)
    train_datasets.extend(datasets)
    train_weights.extend(weights)
    val_datasets.append(('rlu', make_rlu_dataset(True)[0]))

    """DB1"""
    # 67 + 1 + 21
    data_dirs_db1 = glob.glob(os.path.join(root_path, f'db1/*/'))
    filtered_data_dirs_db1 = [
        data_dir for data_dir in data_dirs_db1
        if 'rl_minimal_exp_data' not in data_dir and 'point_mass' not in data_dir and 'walker_7' not in data_dir and 'cheetah_7' not in data_dir and 'hopper_4' not in data_dir and 'dmc-cheetah' not in data_dir
    ]
    print("Filtered DB1 data dirs: ", [
        data_dir for data_dir in data_dirs_db1 if data_dir not in filtered_data_dirs_db1])

    def make_db1_dataset(validate):
        mixed_dataset = [
            SingleHistoryDataset(
                os.path.join(dir, 'episode_*.npz'),
                os.path.join(dir, 'max_min_values.npz'),
                history_len, validate=validate, val_ratio=0.01, in_memory=in_memory, data_ratio=data_ratio
            )
            for dir in filtered_data_dirs_db1
        ]
        dims = [dataset.max_values.shape[0] for dataset in mixed_dataset]
        print(f"total size of db1 {'val' if validate else 'train'} dataset: ",
              sum([dataset.size for dataset in mixed_dataset]))
        print("db1 dataset dims: ", dims)

        return mixed_dataset, [mix_weights['db1'] / len(filtered_data_dirs_db1)] * len(filtered_data_dirs_db1)

    datasets, weights = make_db1_dataset(False)
    train_datasets.extend(datasets)
    train_weights.extend(weights)
    val_datasets.append(('db1', make_db1_dataset(True)[0]))

    """JAT MuJoCo"""
    # 50k episodes
    data_dirs_jat_mujuco = glob.glob(os.path.join(root_path, f'jat/mujoco-*'))
    filtered_data_dirs_jat_mujuco = [
        data_dir for data_dir in data_dirs_jat_mujuco
        if 'walker' not in data_dir and 'cheetah' not in data_dir and 'hopper' not in data_dir and 'ant' not in data_dir and 'standup' not in data_dir and 'humanoid' not in data_dir
    ]
    print("Filtered JAT MuJoCo data dirs: ", [
        data_dir for data_dir in data_dirs_jat_mujuco if data_dir not in filtered_data_dirs_jat_mujuco])

    def make_jat_mujoco_dataset(validate):
        mixed_dataset = [
            SingleHistoryDataset(
                os.path.join(dir, '*_eps_*.npz'),
                os.path.join(dir, 'max_min_values.npz'),
                history_len, validate=validate, val_ratio=0.01, in_memory=in_memory, data_ratio=data_ratio
            )
            for dir in filtered_data_dirs_jat_mujuco
        ]
        dims = [dataset.max_values.shape[0] for dataset in mixed_dataset]
        print(f"total size of jat mujoco {'val' if validate else 'train'} dataset: ",
              sum([dataset.size for dataset in mixed_dataset]))
        print("jat mujoco dataset dims: ", dims)

        return mixed_dataset, [mix_weights['jat_mujoco'] / len(filtered_data_dirs_jat_mujuco)] * len(filtered_data_dirs_jat_mujuco)

    datasets, weights = make_jat_mujoco_dataset(False)
    train_datasets.extend(datasets)
    train_weights.extend(weights)
    val_datasets.append(('jat_mujoco', make_jat_mujoco_dataset(True)[0]))

    """DONE"""
    if sum(train_weights) != 1.0:
        print("warning: sum of train_weights is not 1.0, sum(weights) = ", sum(train_weights))
        train_weights = [w / sum(train_weights) for w in train_weights]

    train_dataloader = NumpyLoader(MixedHistoryDataset(train_datasets), num_workers=num_workers,
                                   batch_sampler=BatchConsistentSampler(len(train_datasets), batch_size, weights=train_weights))
    val_dataloaders = {name: NumpyLoader(MixedHistoryDataset(dataset), num_workers=num_workers,
                                         batch_sampler=BatchConsistentSampler(len(dataset), batch_size)) for name, dataset in val_datasets}
    return train_dataloader, val_dataloaders
def get_debug_dataloader(batch_size, history_len, num_workers, root_path, verbose=False, in_memory=False, symlog=False):
    assert num_workers == 0 or batch_size % num_workers == 0, "batch_size should be divisible by num_workers"
    mix_weights = {
        'modular_rl': 1.0,
    }
    train_datasets, val_datasets = [], []
    train_weights = []

    """Modular RL"""
    # 37k episodes
    # 1000 steps per episode
    # 20 tasks, 20 robots
    def make_modular_rl_dataset(validate):
        envs_path = glob.glob(os.path.join(root_path, f'modular-rl/*-v2'))
        mixed_dataset = [
            SingleHistoryDataset(
                os.path.join(env_path, 'episode_*.npz'),
                os.path.join(env_path, 'max_min_values.npz'),
                history_len, validate=validate, val_ratio=0.01, in_memory=in_memory
            )
            for env_path in envs_path
        ]
        print(f"total size of modular rl {'val' if validate else 'train'} dataset: ", sum(
            [dataset.size for dataset in mixed_dataset]))
        dims = [dataset.max_values.shape[0] for dataset in mixed_dataset]
        print("modular rl dataset dims: ", dims)
        return mixed_dataset, [mix_weights['modular_rl'] / len(mixed_dataset)] * len(mixed_dataset)

    datasets, weights = make_modular_rl_dataset(False)
    train_datasets.extend(datasets)
    train_weights.extend(weights)
    val_datasets.append(('modular_rl', make_modular_rl_dataset(True)[0]))

    """DONE"""
    if sum(train_weights) != 1.0:
        print("warning: sum of train_weights is not 1.0, sum(weights) = ", sum(train_weights))
        train_weights = [w / sum(train_weights) for w in train_weights]

    train_dataloader = NumpyLoader(MixedHistoryDataset(train_datasets), num_workers=num_workers,
                                   batch_sampler=BatchConsistentSampler(len(train_datasets), batch_size, weights=train_weights))
    val_dataloaders = {name: NumpyLoader(MixedHistoryDataset(dataset), num_workers=num_workers,
                                         batch_sampler=BatchConsistentSampler(len(dataset), batch_size)) for name, dataset in val_datasets}
    return train_dataloader, val_dataloaders

def get_pendulum_dataloader(batch_size, history_len, num_workers, root_path, verbose=False, in_memory=False, symlog=False):
    assert num_workers == 0 or batch_size % num_workers == 0, "batch_size should be divisible by num_workers"
    mix_weights = {
        'pendulum': 1.0,
    }
    train_datasets, val_datasets = [], []
    train_weights = []

    """Pendulum"""
    # 115222 episodes
    # 200 steps per episode
    def make_pendulum_dataset(validate):
        if validate:
            envs_path = glob.glob(os.path.join(root_path, f'pendulum/g*_val_final_'))
            print("Validation envs: ")
            for env_path in envs_path:
                print(env_path)
        else:
            envs_all = glob.glob(os.path.join(root_path, f'pendulum/g*'))
            envs_path = [env for env in envs_all if 'val' not in env]
            print("Training envs: ")
            for env_path in envs_path:
                print(env_path)
        pendulum_path = os.path.join(root_path, 'pendulum')
        mixed_dataset = [
            SingleHistoryDataset(
                os.path.join(env_path, 'episode_*.npz'),
                os.path.join(pendulum_path, 'max_min_values.npz'),
                history_len, validate=validate, val_ratio=0.01, in_memory=in_memory
            )
            for env_path in envs_path
        ]
        print(f"total size of pendulum {'val' if validate else 'train'} dataset: ", sum(
            [dataset.size for dataset in mixed_dataset]))
        dims = [dataset.max_values.shape[0] for dataset in mixed_dataset]
        print("pendulum dataset dims: ", dims)
        return mixed_dataset, [mix_weights['pendulum'] / len(mixed_dataset)] * len(mixed_dataset)

    datasets, weights = make_pendulum_dataset(False)
    train_datasets.extend(datasets)
    train_weights.extend(weights)
    val_datasets.append(('pendulum', make_pendulum_dataset(True)[0]))

    """DONE"""
    if sum(train_weights) != 1.0:
        print("warning: sum of train_weights is not 1.0, sum(weights) = ", sum(train_weights))
        train_weights = [w / sum(train_weights) for w in train_weights]

    train_dataloader = NumpyLoader(MixedHistoryDataset(train_datasets), num_workers=num_workers,
                                   batch_sampler=BatchConsistentSampler(len(train_datasets), batch_size, weights=train_weights))
    val_dataloaders = {name: NumpyLoader(MixedHistoryDataset(dataset), num_workers=num_workers,
                                         batch_sampler=BatchConsistentSampler(len(dataset), batch_size)) for name, dataset in val_datasets}
    return train_dataloader, val_dataloaders

def get_two_pole_dataloader(batch_size, history_len, num_workers, root_path, verbose=False, in_memory=False, symlog=False):
    assert num_workers == 0 or batch_size % num_workers == 0, "batch_size should be divisible by num_workers"
    mix_weights = {
        'two_pole': 1.0,
    }
    train_datasets, val_datasets = [], []
    train_weights = []

    """Pendulum"""
    # 115222 episodes
    # 200 steps per episode
    def make_two_pole_dataset(validate):
        pendulum_path = os.path.join(root_path, 'two_poles_demo')
        mixed_dataset = [
            SingleHistoryDataset(
                os.path.join(pendulum_path, 'episode_*.npz'),
                os.path.join(pendulum_path, 'max_min_values.npz'),
                history_len, validate=validate, val_ratio=0.01, in_memory=in_memory
            )
        ]
        print(f"total size of pendulum {'val' if validate else 'train'} dataset: ", sum(
            [dataset.size for dataset in mixed_dataset]))
        dims = [dataset.max_values.shape[0] for dataset in mixed_dataset]
        print("two_pole dataset dims: ", dims)
        return mixed_dataset, [mix_weights['two_pole'] / len(mixed_dataset)] * len(mixed_dataset)

    datasets, weights = make_two_pole_dataset(False)
    train_datasets.extend(datasets)
    train_weights.extend(weights)
    val_datasets.append(('two_pole', make_two_pole_dataset(False)[0]))

    """DONE"""
    if sum(train_weights) != 1.0:
        print("warning: sum of train_weights is not 1.0, sum(weights) = ", sum(train_weights))
        train_weights = [w / sum(train_weights) for w in train_weights]

    train_dataloader = NumpyLoader(MixedHistoryDataset(train_datasets), num_workers=num_workers,
                                   batch_sampler=BatchConsistentSampler(len(train_datasets), batch_size, weights=train_weights))
    val_dataloaders = {name: NumpyLoader(MixedHistoryDataset(dataset), num_workers=num_workers,
                                         batch_sampler=BatchConsistentSampler(len(dataset), batch_size)) for name, dataset in val_datasets}
    return train_dataloader, val_dataloaders

def get_two_pole_demo_dataloader(batch_size, history_len, num_workers, root_path, verbose=False, in_memory=False, symlog=False):
    assert num_workers == 0 or batch_size % num_workers == 0, "batch_size should be divisible by num_workers"
    mix_weights = {
        'two_pole': 1.0,
    }
    train_datasets, val_datasets = [], []
    train_weights = []

    """Pendulum"""
    # 115222 episodes
    # 200 steps per episode
    def make_two_pole_dataset(validate):
        pendulum_path = os.path.join(root_path, 'two_poles_demo')
        mixed_dataset = [
            SingleHistoryDataset(
                os.path.join(pendulum_path, 'episode_*.npz'),
                os.path.join(pendulum_path, 'max_min_values.npz'),
                history_len, validate=validate, val_ratio=0.0, in_memory=in_memory
            )
        ]
        print(f"total size of pendulum {'val' if validate else 'train'} dataset: ", sum(
            [dataset.size for dataset in mixed_dataset]))
        dims = [dataset.max_values.shape[0] for dataset in mixed_dataset]
        print("two_pole dataset dims: ", dims)
        return mixed_dataset, [mix_weights['two_pole'] / len(mixed_dataset)] * len(mixed_dataset)

    datasets, weights = make_two_pole_dataset(False)
    train_datasets.extend(datasets)
    train_weights.extend(weights)
    val_datasets.append(('two_pole', make_two_pole_dataset(False)[0]))

    """DONE"""
    if sum(train_weights) != 1.0:
        print("warning: sum of train_weights is not 1.0, sum(weights) = ", sum(train_weights))
        train_weights = [w / sum(train_weights) for w in train_weights]

    train_dataloader = NumpyLoader(MixedHistoryDataset(train_datasets), num_workers=num_workers,
                                   batch_sampler=BatchConsistentSampler(len(train_datasets), batch_size, weights=train_weights))
    val_dataloaders = {name: NumpyLoader(MixedHistoryDataset(dataset), num_workers=num_workers,
                                         batch_sampler=BatchConsistentSampler(len(dataset), batch_size)) for name, dataset in val_datasets}
    return train_dataloader, val_dataloaders

if __name__ == "__main__":
    train_dataloader, val_dataloader = get_final_dataloader_v2(
        batch_size=64, history_len=20, num_workers=16, root_path='/data/heterogeneous_rl_datasets', in_memory=True)
    start_time = time.time()
    max_iter = 1000
    iter_time = time.time()
    for i, batch in enumerate(train_dataloader):
        print(i, time.time() - iter_time, batch['history'].shape)
        iter_time = time.time()
        if i > max_iter:
            break
    print(f"Time: {(time.time() - start_time) / max_iter}")

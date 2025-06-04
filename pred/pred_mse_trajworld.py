import argparse
import json
from coolname import generate_slug
from ope.logger import Logger
from ope import utils
import time
import random
import gym
from tqdm import tqdm
from omegaconf import OmegaConf
import pickle
import d4rl.gym_mujoco
from dynamics.config import MOPOConfig
from data.history_data import HistoryBatch, get_history_dataset, HistoryReplayBuffer
from dynamics.trajworld_dynamics import TrajWorldDynamics, transform_to_onehot, transform_to_probs, transform_from_probs, transform_from_probs_sample
from dynamics.utils import *

Params = flax.core.FrozenDict[str, Any]

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

'''
!!!Note: Use the checkpoint with the lowest validation MSE.
For the same domain, use the same model to make predictions on datasets of different levels, and then calculate the MSE.
Run Example:
python pred/pred_mse_trajworld.py --env walker2d-random-v2 --model_path trm_dynamics.pkl 
'''

def eval_tsm(
        dynamics: TrajWorldDynamics,
        data: HistoryBatch,
        batch_size: int = 1024,
    ) -> None:
        inputs = data.histories
        masks = data.history_masks

        obs_mse = 0
        all_mse = 0
        obs_abs_err = 0
        all_abs_err = 0
        for i in tqdm(range(0, len(inputs), batch_size), desc="Validating batches"):
            batch_inputs = inputs[i:i+batch_size]
            batch_masks = masks[i:i+batch_size]

            info = dynamics.validate(batch_inputs, batch_masks, batch_size)

            obs_mse += info['obs_mse']*len(batch_inputs)/len(inputs)
            all_mse += info['mse']*len(batch_inputs)/len(inputs)
            obs_abs_err += info['obs_abs_err']*len(batch_inputs)/len(inputs)
            all_abs_err += info['abs_err']*len(batch_inputs)/len(inputs)

        obs_mse_avg = obs_mse
        all_mse_avg = all_mse
        obs_abs_err_avg = obs_abs_err
        all_abs_err_avg = all_abs_err

        return obs_mse_avg, all_mse_avg, obs_abs_err_avg, all_abs_err_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='ens', type=str, choices=['trm', 'ens'])
    parser.add_argument("--env", default="halfcheetah-medium-v2")
    parser.add_argument('--task', default='d4rl')
    parser.add_argument('--seed', default=0, type=int)

    # Work dir
    parser.add_argument('--notes', default=None, type=str)
    parser.add_argument('--work_dir', default='pred_mse_exp', type=str)
    parser.add_argument('--model_path', default='path/to/pretrained/architecture', type=str)
    parser.add_argument('--n_blocks', default=6, type=int)

    args = parser.parse_args()
    args.cooldir = generate_slug(2)
    args.algo = 'tsm'
    print("algo:", args.algo)

    conf_dict = OmegaConf.from_cli()
    config = MOPOConfig(**conf_dict)
    config.penalty_coef = 0.0  # as we are evaluating, the reward should be raw_rewards instead of penalized rewards
    config.env_name = args.env
    config.n_blocks = args.n_blocks

    def create_work_dir():
        # Build work dir
        base_dir = 'runs_new'
        utils.make_dir(base_dir)
        base_dir = os.path.join(base_dir, args.work_dir)
        utils.make_dir(base_dir)
        base_dir = os.path.join(base_dir, args.env)
        utils.make_dir(base_dir)
        work_dir = base_dir
        utils.make_dir(work_dir)

        # make directory
        ts = time.gmtime()
        ts = time.strftime("%m-%d-%H:%M", ts)
        exp_name = str(args.algo) + '-' + str(args.env) + '-' + ts + '-s' + str(args.seed)
        if args.algo == 'reme':
            exp_name += '-K' + str(args.reme_K)
        exp_name += '-' + args.cooldir
        if args.notes is not None:
            exp_name = args.notes + '_' + exp_name
        work_dir = work_dir + '/' + exp_name
        utils.make_dir(work_dir)

        utils.make_dir(os.path.join(work_dir, 'architecture'))
        utils.make_dir(os.path.join(work_dir, 'video'))

        with open(os.path.join(work_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        return work_dir

    work_dir = create_work_dir()
    logger = Logger(work_dir, use_tb=True)
    env_name = config.env_name.split('-')[0]

    for leval in ['random', 'medium', 'medium-replay', 'medium-expert', 'expert']:
        env = gym.make(f"{env_name}-{leval}-v2")
        example_env = env
        hist_dataset, dataset, max_values, min_values, obs_mean, obs_std = get_history_dataset(
            env, config,
            dataset=pickle.load(open(config.dataset_path, 'rb')) if config.dataset_path is not None else None)

        config.act_dim = dataset.actions.shape[-1]
        config.obs_dim = dataset.observations.shape[-1]
        config.target_entropy = config.target_entropy if config.target_entropy else -config.act_dim

        if config.force_max_reward is not None:
            max_values[config.obs_dim] = config.force_max_reward

        # seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        env.seed(config.seed)

        history_buffer = HistoryReplayBuffer(
            config.obs_dim,
            config.act_dim,
            len(dataset.observations) * 2,
            config.history_length,
        )
        history_buffer.initialize_with_dataset(hist_dataset)

        """Get architecture"""
        model = TrajWorldDynamics(config, max_values, min_values)
        model.load(args.model_path)

        data = history_buffer.sample_all()
        # hist_data = history_buffer.sample_all()
        obs_mse_avg, all_mse_avg, obs_abs_err_avg, all_abs_err_avg = eval_tsm(model, data)
        logger.log(f'eval/obs_mse_avg_{env_name}-{leval}-v2', obs_mse_avg, step=0)
        logger.log(f'eval/mse_avg_{env_name}-{leval}-v2', all_mse_avg, step=0)
        logger.log(f'eval/obs_abs_err_avg_{env_name}-{leval}-v2', obs_abs_err_avg, step=0)
        logger.log(f'eval/all_abs_err_avg_{env_name}-{leval}-v2', all_abs_err_avg, step=0)

        print(f'eval/obs_mse_avg_{env_name}-{leval}-v2', obs_mse_avg)
        print(f'eval/mse_avg_{env_name}-{leval}-v2', all_mse_avg)
        print(f'eval/obs_abs_err_avg_{env_name}-{leval}-v2', obs_abs_err_avg)
        print(f'eval/all_abs_err_avg_{env_name}-{leval}-v2', all_abs_err_avg)

    logger._sw.close()

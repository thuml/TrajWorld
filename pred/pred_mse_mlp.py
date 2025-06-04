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
from data.data import Batch, ReplayBuffer, get_dataset
from dynamics.mlp_ensemble_dynamics import EnsembleDynamics
from dynamics.utils import *

Params = flax.core.FrozenDict[str, Any]

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

'''
!!!Note: Use the checkpoint with the lowest validation MSE.
For the same domain, use the same model to make predictions on datasets of different levels, and then calculate the MSE.
Run Example:
python pred/pred_mse_mlp.py --env walker2d-random-v2 --model_path mlp_dynamics.pkl --mlp_huge=true --log_root_dir "log_pred_mse_huge_mlp"
'''

def eval_mlp(
        dynamics: EnsembleDynamics,
        scaler: StandardScaler,
        data: Batch,
        mlp_huge: bool = False,
        batch_size: int = 2048,
    ):
        obss = data.observations
        actions = data.actions
        next_obss = data.next_observations
        rewards = np.expand_dims(data.rewards, axis=-1)

        if mlp_huge:
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

        scaler.fit(inputs)
        inputs = scaler.transform(inputs)

        obs_mse_avg = 0
        all_mse_avg = 0
        obs_abs_err_avg = 0
        all_abs_err_avg = 0

        for i in tqdm(range(0, len(inputs), batch_size), desc="Validating batches"):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]

            mse, info = dynamics.validate(batch_inputs, batch_targets, training_target_masks)
            dyna_mse = info["obs_mse_all"]
            dyna_abs = info["obs_abs_err_all"]
            all_abs = info["abs_err_all"]
            for elite in dynamics.elites:
                obs_mse_avg += dyna_mse[elite] * len(batch_inputs)/ len(inputs) / len(dynamics.elites)
                all_mse_avg += mse[elite] * len(batch_inputs)/ len(inputs) / len(dynamics.elites)
                obs_abs_err_avg += dyna_abs[elite] * len(batch_inputs) / len(inputs) / len(dynamics.elites)
                all_abs_err_avg += all_abs[elite] * len(batch_inputs) / len(inputs) / len(dynamics.elites)

        obs_mse_avg = obs_mse_avg
        mse_avg = all_mse_avg
        obs_abs_err_avg = obs_abs_err_avg
        all_abs_err_avg = all_abs_err_avg

        return obs_mse_avg, mse_avg, obs_abs_err_avg, all_abs_err_avg


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
    parser.add_argument('--mlp_huge', default=False, type=bool)
    parser.add_argument('--log_root_dir', default='runs_new', type=str)

    args = parser.parse_args()
    args.cooldir = generate_slug(2)
    args.algo = 'ens'
    print("algo:", args.algo)

    conf_dict = OmegaConf.from_cli()
    config = MOPOConfig(**conf_dict)
    config.penalty_coef = 0.0  # as we are evaluating, the reward should be raw_rewards instead of penalized rewards
    config.env_name = args.env

    def create_work_dir():
        # Build work dir
        base_dir = args.log_root_dir
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

    all_abs_err_avg_list = []
    obs_abs_err_avg_list = []
    mse_avg_list = []
    obs_mse_avg_list = []
    for leval in ['random', 'medium', 'medium-replay', 'medium-expert', 'expert']:
        env = gym.make(f"{env_name}-{leval}-v2")

        # seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        env.seed(config.seed)

        dataset, obs_mean, obs_std = get_dataset(
            env, config,
            dataset=pickle.load(open(config.dataset_path, 'rb')) if config.dataset_path is not None else None
        )
        config.act_dim = dataset.actions.shape[-1]
        config.obs_dim = dataset.observations.shape[-1]

        real_buffer = ReplayBuffer(
            config.obs_dim,
            config.act_dim,
            len(dataset.observations),
        )
        real_buffer.initialize_with_dataset(dataset)

        if args.mlp_huge:
            config.dynamics_hidden_dims = (640, 640, 640, 640)
            config.obs_dim, config.act_dim = 90, 30
            print('Using Huge MLP Dynamics Model')

        """Get architecture"""
        model = EnsembleDynamics(config)
        model.load(args.model_path)

        data = real_buffer.sample_all()
        # hist_data = history_buffer.sample_all()
        scaler = StandardScaler()
        obs_mse_avg, mse_avg, obs_abs_err_avg, all_abs_err_avg = eval_mlp(model, scaler, data, args.mlp_huge)

        logger.log(f'eval/obs_mse_avg_{env_name}-{leval}-v2', obs_mse_avg, step=0)
        logger.log(f'eval/mse_avg_{env_name}-{leval}-v2', mse_avg, step=0)
        logger.log(f'eval/obs_abs_err_avg_{env_name}-{leval}-v2', obs_abs_err_avg, step=0)
        logger.log(f'eval/all_abs_err_avg_{env_name}-{leval}-v2', all_abs_err_avg, step=0)

        print(f'eval/obs_mse_avg_{env_name}-{leval}-v2', obs_mse_avg)
        print(f'eval/mse_avg_{env_name}-{leval}-v2', mse_avg)
        print(f'eval/obs_abs_err_avg_{env_name}-{leval}-v2', obs_abs_err_avg)
        print(f'eval/all_abs_err_avg_{env_name}-{leval}-v2', all_abs_err_avg)

        # Append metrics to the lists
        all_abs_err_avg_list.append(all_abs_err_avg)
        obs_abs_err_avg_list.append(obs_abs_err_avg)
        mse_avg_list.append(mse_avg)
        obs_mse_avg_list.append(obs_mse_avg)

    # After the loop, print the collected data
    print(' '.join(map(str, all_abs_err_avg_list)))
    print(' '.join(map(str, obs_abs_err_avg_list)))
    print(' '.join(map(str, mse_avg_list)))
    print(' '.join(map(str, obs_mse_avg_list)))

    logger._sw.close()

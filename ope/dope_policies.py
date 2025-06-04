import numpy as np
import pickle
import os
import distrax
from tqdm import tqdm
import gym

import jax
import jax.numpy as jnp
import flax

import torch
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
import torch.distributions.transforms as transforms
from tensorflow import io as tfio


def estimate_monte_carlo_returns(
    env_name,
    discount,
    actor,
    num_episodes,
    seed,
    max_length=1000,
    seed_offset=100,
    verbose=False
):
    """Estimate policy returns using with Monte Carlo.

    Args:
      env_name: Learning environment.
      discount: MDP discount.
      actor: Policy to estimate returns for.
      num_episodes: Number of episodes.
      max_length: Maximum length of episodes.

    Returns:
      A dictionary that contains trajectories.
    """

    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    eval_env._duration = max_length  # pylint: disable=protected-access

    episode_returns = []
    episode_returns_nondiscounted = []
    for _ in tqdm(range(num_episodes), desc='Estimation Returns'):
        state, done = eval_env.reset(), False
        episode_return = 0.0
        episode_return_nondiscounted = 0.0

        t = 0
        while not done:
            action = actor(state)
            state, reward, done, _ = eval_env.step(action)
            episode_return += reward * (discount**t)
            episode_return_nondiscounted += reward
            t += 1

        episode_returns.append(episode_return)
        episode_returns_nondiscounted.append(episode_return_nondiscounted)
        if verbose:
            print(episode_return, episode_return_nondiscounted)

    if verbose:
        print(np.mean(episode_returns_nondiscounted), np.std(episode_returns_nondiscounted))
    return np.mean(episode_returns), \
        np.mean(episode_returns_nondiscounted)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DopeD4rlPolicyTorch:
    """D4RL policy."""

    def __init__(self, weights):
        self.fc0_w = torch.from_numpy(weights['fc0/weight']).to(device)
        self.fc0_b = torch.from_numpy(weights['fc0/bias']).to(device)
        self.fc1_w = torch.from_numpy(weights['fc1/weight']).to(device)
        self.fc1_b = torch.from_numpy(weights['fc1/bias']).to(device)
        self.fclast_w = torch.from_numpy(weights['last_fc/weight']).to(device)
        self.fclast_b = torch.from_numpy(weights['last_fc/bias']).to(device)
        self.fclast_w_logstd = torch.from_numpy(weights['last_fc_log_std/weight']).to(device)
        self.fclast_b_logstd = torch.from_numpy(weights['last_fc_log_std/bias']).to(device)
        self.nonlinearity = torch.tanh if weights['nonlinearity'] == 'tanh' else torch.relu

        def identity(x): return x
        self.output_transformation = torch.tanh if weights[
            'output_distribution'] == 'tanh_gaussian' else identity

    def forward(self, states):
        states = torch.transpose(states, 0, 1)
        x = torch.matmul(self.fc0_w, states) + self.fc0_b.reshape(-1, 1)
        x = self.nonlinearity(x)
        x = torch.matmul(self.fc1_w, x) + self.fc1_b.reshape(-1, 1)
        x = self.nonlinearity(x)
        mean = torch.matmul(self.fclast_w, x) + self.fclast_b.reshape(-1, 1)
        logstd = torch.matmul(self.fclast_w_logstd, x) + self.fclast_b_logstd.reshape(-1, 1)
        return torch.transpose(mean, 0, 1), torch.transpose(logstd, 0, 1)

    def get_dist_and_mode(self, states):
        mean, logstd = self.forward(states)

        dist = Normal(mean, torch.exp(logstd))
        dist = TransformedDistribution(dist, transforms.TanhTransform())
        dist = torch.distributions.Independent(dist, 1)

        mode = torch.tanh(mean)
        return dist, mode

    def get_target_actions(self, states):
        with torch.no_grad():
            dist, _ = self.get_dist_and_mode(states)
            action = dist.sample().detach()
        return action

    def get_target_logprobs(self, states, actions):
        with torch.no_grad():
            dist, _ = self.get_dist_and_mode(states)
            logprobs = dist.log_prob(actions).detach()
        return logprobs

    def __call__(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            x = torch.matmul(self.fc0_w, state) + self.fc0_b
            x = self.nonlinearity(x)
            x = torch.matmul(self.fc1_w, x) + self.fc1_b
            x = self.nonlinearity(x)
            mean = torch.matmul(self.fclast_w, x) + self.fclast_b
            logstd = torch.matmul(self.fclast_w_logstd, x) + self.fclast_b_logstd

            noise = torch.randn_like(mean)
            action = self.output_transformation(mean + torch.exp(logstd) * noise)

        return action.detach().cpu().numpy()


def get_jax_policy_actions(rng, weights, nonlinearity, states: jnp.ndarray):
    nonlinearity = jnp.tanh if nonlinearity == 'tanh' else flax.linen.relu
    x = jnp.dot(weights['fc0/weight'], states.T) + weights['fc0/bias'][:, None]
    x = nonlinearity(x)
    x = jnp.dot(weights['fc1/weight'], x) + weights['fc1/bias'][:, None]
    x = nonlinearity(x)
    mean = jnp.dot(weights['last_fc/weight'], x) + weights['last_fc/bias'][:, None]
    logstd = jnp.dot(weights['last_fc_log_std/weight'], x) + weights['last_fc_log_std/bias'][:, None]
    means, log_stds = mean.T, logstd.T

    base_dist = distrax.MultivariateNormalDiag(
        loc=means, scale_diag=jnp.exp(log_stds)
    )
    dist = distrax.Transformed(distribution=base_dist,
                               bijector=distrax.Block(distrax.Tanh(), ndims=1))

    rng, key = jax.random.split(rng)
    action = dist.sample(seed=key)
    return rng, action


class DopeD4rlPolicyJax:
    """D4RL policy."""

    def __init__(self, weights, seed):
        self.weights = weights
        self.rng = jax.random.PRNGKey(seed)
        self.fc0_w = jnp.array(weights['fc0/weight'])
        self.fc0_b = jnp.array(weights['fc0/bias'])
        self.fc1_w = jnp.array(weights['fc1/weight'])
        self.fc1_b = jnp.array(weights['fc1/bias'])
        self.fclast_w = jnp.array(weights['last_fc/weight'])
        self.fclast_b = jnp.array(weights['last_fc/bias'])
        self.fclast_w_logstd = jnp.array(weights['last_fc_log_std/weight'])
        self.fclast_b_logstd = jnp.array(weights['last_fc_log_std/bias'])
        self.nonlinearity = jnp.tanh if weights['nonlinearity'] == 'tanh' else flax.linen.relu

        def identity(x): return x
        self.output_transformation = jnp.tanh if weights[
            'output_distribution'] == 'tanh_gaussian' else identity
        assert weights['output_distribution'] == 'tanh_gaussian'

    def forward(self, states):
        # print(f"self.fc0_w.shape:{self.fc0_w.shape},states:{states.shape},self.fc0_b.shape:{self.fc0_b.shape}")
        x = jnp.dot(self.fc0_w, states.T) + self.fc0_b[:, None]
        x = self.nonlinearity(x)
        x = jnp.dot(self.fc1_w, x) + self.fc1_b[:, None]
        x = self.nonlinearity(x)
        mean = jnp.dot(self.fclast_w, x) + self.fclast_b[:, None]
        logstd = jnp.dot(self.fclast_w_logstd, x) + self.fclast_b_logstd[:, None]
        return mean.T, logstd.T

    def get_dist_and_mode(self, states):
        means, log_stds = self.forward(states)

        base_dist = distrax.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds)
        )
        dist = distrax.Transformed(distribution=base_dist,
                                   bijector=distrax.Block(distrax.Tanh(), ndims=1))
        mode = jnp.tanh(means)
        return dist, mode

    def get_target_actions(self, states, rng=None):
        dist, _ = self.get_dist_and_mode(states)
        if rng is None:
            self.rng, key = jax.random.split(self.rng)
            action = dist.sample(seed=key)
            return action
        else:
            rng, key = jax.random.split(rng)
            action = dist.sample(seed=key)
            return rng, action

def get_target_policy(target_policy, args, jax_policy=True):
    env_name = target_policy.split('_')[0]
    policy_weight_path = os.path.join('d4rl_policies', env_name, f'{target_policy}.pkl')
    policy_weight_path = policy_weight_path.replace('walker2d', 'walker')

    with tfio.gfile.GFile(policy_weight_path, 'rb') as f:
        policy_weights = pickle.load(f)
    if jax_policy:
        actor = DopeD4rlPolicyJax(policy_weights, args.seed)
    else:
        raise NotImplementedError

    if args.mc_estimate_target_policy:
        policy_returns, _ = estimate_monte_carlo_returns(
            args.env, args.discount, actor, args.num_mc_episodes, args.seed,
            verbose=True
        )
    else:
        gt = pickle.load(open('dope_benchmark/d4rl_gt.pkl', 'rb'))
        policy_id = target_policy.split('_')[-1]
        policy_alias = '-'.join(args.env.split('-')[:-1]) + '_' + '%02d' % int(policy_id)
        if policy_alias not in gt:
            print(f'warning: {policy_alias} not in gt')
            policy_alias = args.env.split('-')[0] + '-medium_' + '%02d' % int(policy_id)
        policy_returns = gt[policy_alias][0]

    return actor, policy_returns

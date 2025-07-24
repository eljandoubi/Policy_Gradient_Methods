import os
from logging import Logger
from typing import Optional, List, Deque, Dict, Literal
import torch
import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
from general import get_logger, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, np2torch, DEVICE
from policy import CategoricalPolicy, GaussianPolicy
from config import base_config


class PolicyGradient:
    """
    Class for implementing a policy gradient algorithm
    """

    def __init__(
        self,
        env: gym.Env,
        config: base_config,
        seed: int,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize Policy Gradient Class

        Args:
                env: an OpenAI Gym environment
                config: class with hyperparameters
                logger: logger instance from the logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyperparameters
        self.config = config
        self.seed = seed

        if logger is None:
            self.logger = get_logger(config.log_path)
        else:
            self.logger = logger

        env.reset(seed=seed)
        self.env = env

        # discrete vs continuous action space
        self.discrete = isinstance(env.action_space, Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        if isinstance(env.action_space, Discrete):
            self.action_dim = env.action_space.n
        else:
            assert len(env.action_space.shape) == 1, (
                "Only 1D continuous action spaces are supported"
            )
            self.action_dim = env.action_space.shape[0]

        self.lr = self.config.learning_rate

        self.init_policy()

        if config.use_baseline:
            self.baseline_network = BaselineNetwork(self.observation_dim, config).to(
                DEVICE
            )

    def init_policy(self):
        """
        Initializes the policy network and optimizer.
        This method creates a multi-layer perceptron (MLP) for the policy
        and assigns it to self.policy. It also creates an optimizer for the policy.
        """
        mlp = build_mlp(
            input_size=self.observation_dim,
            output_size=self.action_dim,
            n_layers=self.config.n_layers,
            size=self.config.layer_size,
        )
        if self.discrete:
            self.policy = CategoricalPolicy(mlp)
        else:
            self.policy = GaussianPolicy(mlp, action_dim=self.action_dim)
        # Create an Adam optimizer for the policy
        self.policy.to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def init_averages(self):
        """
        Initializes the averages for rewards.
        """
        self.avg_reward = 0.0
        self.max_reward = 0.0
        self.std_reward = 0.0
        self.eval_reward = 0.0

    def update_averages(self, rewards: List, scores_eval: Deque) -> None:
        """
        Update the averages.
        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        pass

    def sample_path(self, env: gym.Env, num_episodes: Optional[int] = None):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym envinronment

        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"
        """
        episode = 0
        episode_rewards = []
        paths = []
        t = 0

        while num_episodes or t < self.config.batch_size:
            state = env.reset()[0]
            states, actions, rewards = [], [], []
            episode_reward = 0

            for step in range(self.config.max_ep_len):
                states.append(state)
                action = self.policy.act(states[-1][None])[0]
                state, reward, done, *_ = env.step(action)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if done or step == self.config.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break

            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    def get_returns(self, paths: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Calculate the returns G_t for each timestep

        Args:
            paths: recorded sample paths. See sample_path() for details.

        Return:
            returns: return G_t for each timestep

        After acting in the environment, we record the observations, actions, and
        rewards. To get the advantages that we need for the policy update, we have
        to convert the rewards into returns, G_t, which are themselves an estimate
        of Q^π (s_t, a_t):

           G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

        where T is the last timestep of the episode.

        Note that here we are creating a list of returns for each path
        """

        all_returns = []
        for path in paths:
            rewards = path["reward"]
            returns = np.zeros_like(rewards, dtype=np.float32)
            for t in range(len(rewards) - 1, -1, -1):
                if t == len(rewards) - 1:
                    returns[t] = rewards[t]
                else:
                    returns[t] = rewards[t] + self.config.gamma * returns[t + 1]
            all_returns.append(returns)
        returns = np.concatenate(all_returns)

        return returns

    def normalize_advantage(self, advantages: np.ndarray) -> np.ndarray:
        """
        Args:
            advantages: np.array of shape [batch size]
        Returns:
            normalized_advantages: np.array of shape [batch size]
        """
        normalized_advantages = (advantages - np.mean(advantages)) / (
            np.std(advantages) + 1e-9
        )
        return normalized_advantages

    def calculate_advantage(
        self, returns: np.ndarray, observations: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the advantage for each of the observations
        Args:
            returns: np.array of shape [batch size]
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        if self.config.use_baseline:
            # override the behavior of advantage by subtracting baseline
            advantages = self.baseline_network.calculate_advantage(
                returns, observations
            )
        else:
            advantages = returns

        if self.config.normalize_advantage:
            advantages = self.normalize_advantage(advantages)

        return advantages

    def update_policy(
        self, observations: np.ndarray, actions: np.ndarray, advantages: np.ndarray
    ) -> None:
        """
        Update the policy network using the provided observations, actions, and advantages.
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size]

        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        self.optimizer.zero_grad()
        log_probs = self.policy.action_distribution(observations).log_prob(actions)
        loss = -torch.mean(log_probs * advantages)
        loss.backward()
        self.optimizer.step()

    def train(self):
        """
        Performs training
        """
        last_record = 0

        self.init_averages()
        all_total_rewards = []  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration

        for t in range(self.config.num_batches):
            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)

            # advantage will depend on the baseline implementation
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            if self.config.use_baseline:
                self.baseline_network.update_baseline(returns, observations)
            self.update_policy(observations, actions, advantages)

            # logging
            if t % self.config.summary_freq == 0:
                self.update_averages(total_rewards, all_total_rewards)
                self.record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                t, avg_reward, sigma_reward
            )
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg)

            if self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        self.logger.info("- Training done.")
        np.save(self.config.scores_output, averaged_total_rewards)
        export_plot(
            averaged_total_rewards,
            "Score",
            self.config.env_name,
            self.config.plot_output,
        )

    def evaluate(self, env: Optional[gym.Env] = None, num_episodes: int = 1):
        """
        Evaluates the return for num_episodes episodes.
        Not used right now, all evaluation statistics are computed during training
        episodes.
        """
        if env is None:
            env = self.env
        _, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def record(self, time: Literal["before", "after"]):
        """
        Recreate an env and record a video for one episode
        """
        env = gym.make(self.config.env_name, render_mode="rgb_array")
        env.reset(seed=self.seed)
        env = gym.wrappers.RecordVideo(
            env,
            self.config.record_path,
            episode_trigger=lambda episode_id: True,
            name_prefix=time,
        )
        self.evaluate(env, 1)
        env.close()

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # record one game at the beginning
        if self.config.record:
            self.record("before")
        # model
        self.train()
        # record one game at the end
        if self.config.record:
            self.record("after")


if __name__ == "__main__":
    # This is just to ensure that the code runs without errors.
    from config import get_config

    for name in ["cartpole", "pendulum", "cheetah"]:
        config = get_config(name, baseline=True, ppo=False)
        env = gym.make(config.env_name)
        config.num_batches = 3  # Reduce for quick testing
        config.batch_size = 16  # Reduce for quick testing
        config.max_ep_len = 10  # Reduce for quick testing
        config.normalize_advantage = True  # Enable advantage normalization
        pg = PolicyGradient(env, config, seed=config.seed)
        pg.logger.info("Starting training PolicyGradient... for %s", config.env_name)
        pg.run()
        env.close()

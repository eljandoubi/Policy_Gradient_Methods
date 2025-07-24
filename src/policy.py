import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
from network_utils import np2torch


class BasePolicy:
    def action_distribution(self, observations: torch.Tensor) -> ptd.Distribution:
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution

        This is an abstract method and must be overridden by subclasses.
        It will return an object representing the policy's conditional
        distribution(s) given the observations. The distribution will have a
        batch shape matching that of observations, to allow for a different
        distribution for each observation in the batch.
        """
        raise NotImplementedError

    def act(self, observations: np.ndarray, return_log_prob=False):
        """
        * Sample actions from the policy given observations.
        * Optionally return the log probabilities of the sampled actions.
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]
            log_probs: np.array of shape [batch size] (optionally, if return_log_prob)
        """
        observations = np2torch(observations)
        distribution = self.action_distribution(observations)
        sampled_actions = distribution.sample()
        np_sampled_actions = sampled_actions.detach().cpu().numpy()
        if return_log_prob:
            log_probs = distribution.log_prob(sampled_actions)

            return np_sampled_actions, log_probs.detach().cpu().numpy()

        return np_sampled_actions


class CategoricalPolicy(BasePolicy, nn.Module):
    def __init__(self, network: nn.Module):
        nn.Module.__init__(self)
        self.network = network

    def action_distribution(self, observations: torch.Tensor) -> ptd.Categorical:
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network
        """
        logits = self.network(observations)
        distribution = ptd.Categorical(logits=logits)
        return distribution


class GaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, network: nn.Module, action_dim: int):
        """
        Args:
            network: nn.Module, a neural network that outputs the mean of the action distribution
            action_dim: int, the dimension of the action space
        """
        nn.Module.__init__(self)
        self.network = network
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: an instance of a subclass of
                torch.distributions.Distribution representing a diagonal
                Gaussian distribution whose mean (loc) is computed by
                self.network and standard deviation (scale) is self.std()

        """
        distribution = ptd.MultivariateNormal(
            loc=self.network(observations), scale_tril=self.log_std.exp().diag()
        )
        return distribution


if __name__ == "__main__":
    # This is just to ensure that the code runs without errors.
    print("Testing CategoricalPolicy")
    policy = CategoricalPolicy(nn.Linear(4, 2))
    obs = np.random.rand(5, 4)
    actions, log_probs = policy.act(obs, return_log_prob=True)
    print("Sampled Actions:", actions)
    print("Log Probabilities:", log_probs)
    print("Testing GaussianPolicy")
    policy = GaussianPolicy(nn.Linear(4, 2), action_dim=2)
    obs = np.random.rand(5, 4)
    actions, log_probs = policy.act(obs, return_log_prob=True)
    print("Sampled Actions:", actions)
    print("Log Probabilities:", log_probs)

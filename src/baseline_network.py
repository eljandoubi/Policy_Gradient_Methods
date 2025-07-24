import torch
import torch.nn as nn
import numpy as np
from network_utils import build_mlp, np2torch
from config import base_config


class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network
    """

    def __init__(self, observation_dim: int, config: base_config):
        """
        Initializes the baseline network with a multi-layer perceptron (MLP) and an optimizer.
        Args:
            observation_dim: int, dimension of the observation space
            config: base_config, configuration object containing hyperparameters

        """
        super().__init__()
        self.config = config
        self.lr = self.config.learning_rate
        self.network = build_mlp(
            input_size=observation_dim,
            output_size=1,
            n_layers=self.config.n_layers,
            size=self.config.layer_size,
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]
        """
        return self.network(observations).squeeze(-1)

    def calculate_advantage(
        self, returns: np.ndarray, observations: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the advantages using the baseline network.
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]

        """
        observations = np2torch(observations)
        base = self(observations).detach().cpu().numpy()  # Evaluate the baseline
        advantages = returns.astype(np.float32) - base  # Ensure the type is float32
        return advantages

    def update_baseline(self, returns: np.ndarray, observations: np.ndarray) -> None:
        """
        Update the baseline network using the provided returns and observations.
        Args:
            returns: np.array of shape [batch size], containing all discounted
                future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        """
        returns = np2torch(returns)
        observations = np2torch(observations)
        self.optimizer.zero_grad()
        outputs = self(observations)
        loss = nn.functional.mse_loss(outputs, returns)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # Example usage
    import gym
    from config import get_config

    for name in ["cartpole", "pendulum", "cheetah"]:
        config = get_config(name, baseline=True, ppo=False)
        env = gym.make(config.env_name)
        observation_dim = env.observation_space.shape[0]
        baseline_network = BaselineNetwork(observation_dim, config)
        # Dummy data for testing
        observations = np.random.rand(10, observation_dim)
        returns = np.random.rand(10)
        # Update baseline
        baseline_network.update_baseline(returns, observations)
        # Calculate advantages
        advantages = baseline_network.calculate_advantage(returns, observations)
        print(f"Advantages for {name}:", advantages)

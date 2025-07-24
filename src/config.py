from dataclasses import dataclass, field
from typing import Literal


@dataclass
class base_config:
    ppo: bool
    seed: int
    use_baseline: bool = field(default=False)
    env_name: str = field(init=False)
    record: bool = field(default=True)
    output_path: str = field(init=False)
    model_output: str = field(init=False)
    log_path: str = field(init=False)
    scores_output: str = field(init=False)
    plot_output: str = field(init=False)
    record_path: str = field(init=False)
    record_freq: int = field(init=False, default=5)
    summary_freq: int = field(init=False, default=1)
    num_batches: int = field(init=False)
    batch_size: int = field(init=False)
    max_ep_len: int = field(init=False)
    learning_rate: float = field(init=False)
    gamma: float = field(init=False)
    normalize_advantage: bool = field(init=False)
    n_layers: int = field(init=False)
    layer_size: int = field(init=False)
    eps_clip: float = field(init=False)
    update_freq: int = field(init=False)

    def __post_init__(self):
        baseline_str = (
            ("baseline" if self.use_baseline else "no_baseline")
            if not self.ppo
            else "ppo"
        )
        print("baseline_str", baseline_str)
        if self.ppo:
            self.use_baseline = True
        seed_str = f"seed={self.seed}"
        self.output_path = f"results/{self.env_name}-{baseline_str}-{seed_str}/"
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size


@dataclass
class config_cartpole(base_config):
    def __post_init__(self):
        self.env_name = "CartPole-v1"
        self.num_batches = 100
        self.batch_size = 2000
        self.max_ep_len = 200
        self.learning_rate = 3e-2
        self.gamma = 1.0
        self.normalize_advantage = True
        self.n_layers = 1
        self.layer_size = 64
        self.eps_clip = 0.2
        self.update_freq = 5
        super().__post_init__()


@dataclass
class config_pendulum(base_config):
    def __post_init__(self):
        self.env_name = "InvertedPendulum-v5"
        self.num_batches = 100
        self.batch_size = 10000
        self.max_ep_len = 1000
        self.learning_rate = 3e-2
        self.gamma = 1.0
        self.normalize_advantage = True
        self.n_layers = 1
        self.layer_size = 64
        self.eps_clip = 0.2
        self.update_freq = 20
        super().__post_init__()


@dataclass
class config_cheetah(base_config):
    def __post_init__(self):
        self.env_name = "HalfCheetah-v5"
        self.num_batches = 200
        self.batch_size = 10000
        self.max_ep_len = 1000
        self.learning_rate = 3e-2
        self.gamma = 0.9
        self.normalize_advantage = True
        self.n_layers = 2
        self.layer_size = 64
        self.eps_clip = 0.1
        self.update_freq = 10
        super().__post_init__()


def get_config(
    env_name: Literal["cartpole", "pendulum", "cheetah"],
    ppo: bool,
    baseline: bool = False,
    seed: int = 15,
) -> base_config:
    if env_name == "cartpole":
        return config_cartpole(ppo, seed, baseline)
    elif env_name == "pendulum":
        return config_pendulum(ppo, seed, baseline)
    elif env_name == "cheetah":
        return config_cheetah(ppo, seed, baseline)
    else:
        raise ValueError(
            f"Unknown environment name: {env_name}. "
            f"Expected one of ['cartpole', 'pendulum', 'cheetah']."
        )

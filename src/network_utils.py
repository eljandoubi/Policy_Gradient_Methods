import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_mlp(input_size: int, output_size: int, n_layers: int, size: int) -> nn.Module:
    """
    Build a feed-forward network (multi-layer perceptron, or mlp) that maps
    input_size-dimensional vectors to output_size-dimensional vectors.
    Args:
        input_size: int, the dimension of inputs to be given to the network
        output_size: int, the dimension of the output
        n_layers: int, the number of hidden layers of the network
        size: int, the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.

    """
    model = nn.Sequential(
        nn.Sequential(nn.Linear(input_size, size), nn.ReLU()),  # First layer
        *[
            nn.Sequential(nn.Linear(size, size), nn.ReLU()) for _ in range(n_layers - 1)
        ],  # Hidden layers
        nn.Linear(size, output_size),  # Final linear layer
    )
    return model


def np2torch(arr: np.ndarray, cast_double_to_float: bool = True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(arr).to(DEVICE)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x


if __name__ == "__main__":
    # Example usage
    mlp = build_mlp(4, 4, 1, 16)
    print(mlp)
    x = np.random.rand(8, 4)
    x_torch = np2torch(x)
    print("input", x_torch)
    y = mlp(x_torch)
    print("output", y)

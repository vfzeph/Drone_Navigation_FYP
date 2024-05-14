import torch
import numpy as np
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_gae(next_values, rewards, masks, values, gamma, tau):
    """
    Compute generalized advantage estimation (GAE) for stable policy updates.
    
    Args:
        next_values (torch.Tensor): Values estimated for the next states.
        rewards (torch.Tensor): Rewards received after taking actions.
        masks (torch.Tensor): Masks indicating if an episode continues (0 for terminal state).
        values (torch.Tensor): Values estimated for the current states.
        gamma (float): Discount factor for future rewards.
        tau (float): GAE smoothing coefficient.

    Returns:
        torch.Tensor: Returns calculated as sum of current value estimates and GAEs.
    """
    logging.debug("Computing Generalized Advantage Estimation.")
    gae = 0
    returns = []
    values = torch.cat([values, next_values[-1].unsqueeze(0)], dim=0)
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return torch.tensor(returns, device=values.device)

def normalize(x, eps=1e-8):
    """
    Normalize input tensor by subtracting mean and dividing by standard deviation.
    
    Args:
        x (torch.Tensor): Input tensor for normalization.
        eps (float): A small epsilon value to prevent division by zero.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    if x.std().item() < eps:  # Prevent division by zero
        logging.debug("Standard deviation is very small, adjusting normalization process.")
        return x - x.mean()
    return (x - x.mean()) / (x.std() + eps)

def to_tensor(np_array, device='cpu', dtype=torch.float32):
    """
    Convert numpy array to PyTorch tensor.
    
    Args:
        np_array (np.ndarray): Numpy array to convert.
        device (str): The device to load the tensor onto.
        dtype (torch.dtype): Data type of the tensor.

    Returns:
        torch.Tensor: The converted tensor.
    """
    logging.debug("Converting numpy array to tensor.")
    return torch.tensor(np_array, dtype=dtype, device=device)

# Example usage within a training loop or PPO update method
if __name__ == "__main__":
    logging.info("Starting utility functions demonstration.")

    # Assuming some example data for demonstration
    values = torch.rand(10)
    next_values = torch.rand(1)
    rewards = torch.rand(10)
    masks = torch.ones(10)
    gamma = 0.99
    tau = 0.95

    returns = compute_gae(next_values, rewards, masks, values, gamma, tau)
    normalized_returns = normalize(returns)
    print("Returns:", returns)
    print("Normalized Returns:", normalized_returns)

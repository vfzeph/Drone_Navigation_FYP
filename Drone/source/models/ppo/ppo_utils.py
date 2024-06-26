import torch
import numpy as np
import logging

# Configure enhanced structured logging
def configure_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Adjusted level to DEBUG to capture all logs during development
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = configure_logger()

def compute_gae(next_values, rewards, masks, values, gamma, tau):
    """
    Compute Generalized Advantage Estimation (GAE) for better policy optimization.
    GAE helps in reducing the variance of advantage estimates without increasing bias too much.

    Parameters:
    - next_values (torch.Tensor): Values of the next states.
    - rewards (torch.Tensor): Rewards received after taking the actions.
    - masks (torch.Tensor): Masks that indicate whether an episode has ended.
    - values (torch.Tensor): Values of the current states.
    - gamma (float): Discount factor for future rewards.
    - tau (float): Smoothing coefficient for GAE calculation.

    Returns:
    - torch.Tensor: Computed returns using GAE.
    """
    logger.info("Computing Generalized Advantage Estimation (GAE).")
    with torch.no_grad():
        logger.debug(f"next_values shape before: {next_values.shape}")
        if next_values.dim() == 1:
            next_values = next_values.view(1, -1)
        next_values = next_values.expand_as(values[-1].unsqueeze(0))  # Ensure next_values has the same shape as values[-1]
        logger.debug(f"next_values shape after: {next_values.shape}")
        logger.debug(f"values shape: {values.shape}")
        logger.debug(f"rewards shape: {rewards.shape}")
        logger.debug(f"masks shape: {masks.shape}")

        values = torch.cat([values, next_values], dim=0)
        
        # Ensure masks has the same shape as rewards
        if masks.dim() == 1:
            masks = masks.unsqueeze(1)
        masks = masks.expand(rewards.size(0), values.size(1))
        logger.debug(f"expanded masks shape: {masks.shape}")

        deltas = rewards.unsqueeze(1) + gamma * values[1:] * masks - values[:-1]
        gae = torch.zeros_like(values[:-1])
        returns = torch.zeros_like(values[:-1])

        gae_accumulated = torch.zeros_like(values[0])
        for t in reversed(range(len(rewards))):
            gae_accumulated = deltas[t] + gamma * tau * masks[t] * gae_accumulated
            gae[t] = gae_accumulated
            returns[t] = gae[t] + values[t]

        logger.debug("GAE and returns computed successfully.")

    return returns.squeeze(-1), gae.squeeze(-1)

def normalize(x, eps=1e-8):
    """
    Normalize the input tensor to have zero mean and unit standard deviation, ensuring numerical stability.

    Parameters:
    - x (torch.Tensor): Input tensor to normalize.
    - eps (float): Small epsilon value to prevent division by zero.

    Returns:
    - torch.Tensor: Normalized tensor.
    """
    logger.debug("Normalizing tensor.")
    x_mean = x.mean()
    x_std = x.std().clamp(min=eps)
    normalized_x = (x - x_mean) / x_std
    logger.debug(f"Data normalized: mean={x_mean.item()}, std={x_std.item()}")
    return normalized_x

def to_tensor(np_array, device='cpu', dtype=torch.float32):
    """
    Convert a numpy array to a PyTorch tensor, efficiently transferring it to the specified device.

    Parameters:
    - np_array (numpy.ndarray): Array to convert.
    - device (str): The device to which the tensor should be transferred.
    - dtype (torch.dtype): The desired data type of the tensor.

    Returns:
    - torch.Tensor: The resulting tensor on the specified device.
    """
    logger.debug("Converting numpy array to tensor.")
    tensor = torch.tensor(np_array, device=device, dtype=dtype)
    logger.debug(f"Tensor created on {device} with data: {tensor}")
    return tensor

def reward_normalize(rewards):
    """
    Normalize rewards to improve training stability.

    Parameters:
    - rewards (torch.Tensor): Input rewards to normalize.

    Returns:
    - torch.Tensor: Normalized rewards.
    """
    logger.debug("Normalizing rewards.")
    mean = rewards.mean()
    std = rewards.std().clamp(min=1e-8)
    normalized_rewards = (rewards - mean) / std
    logger.debug(f"Rewards normalized: mean={mean.item()}, std={std.item()}")
    return normalized_rewards

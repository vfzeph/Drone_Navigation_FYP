import numpy as np
import random
from collections import namedtuple, deque
import torch

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples with prioritization."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, device="cpu"):
        """Initialize a PrioritizedReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): exponent parameter for priorities
            beta (float): importance-sampling parameter, annealing towards 1
            beta_increment_per_sampling (float): increment value for beta per sample
            device (str): device to store tensors ('cpu' or 'cuda')
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.device = device
        self.seed = random.seed(seed)
        self.epsilon = 1e-5

    def add(self, state, action, reward, next_state, done, error):
        """Add a new experience to memory."""
        max_priority = max(self.priorities) if self.memory else 1.0
        self.memory.append(self.experience(state, action, reward, next_state, done))
        self.priorities.append(max_priority ** self.alpha)

    def sample(self):
        """Randomly sample a batch of experiences from memory with prioritization."""
        priorities = np.array(self.priorities)
        prob_bs = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), self.batch_size, p=prob_bs)
        experiences = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * prob_bs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)

        return (states, actions, rewards, next_states, dones, weights, indices)

    def update_priorities(self, indices, errors):
        """Update priorities of sampled experiences based on their TD error."""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + self.epsilon) ** self.alpha

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

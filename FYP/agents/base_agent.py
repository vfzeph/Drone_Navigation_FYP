from abc import ABC, abstractmethod
import torch

class BaseAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.
    """

    def __init__(self, state_size, action_size, seed, device='cpu'):
        """
        Initializes the agent.

        Parameters:
        - state_size (int): Dimensionality of the state space.
        - action_size (int): Dimensionality of the action space.
        - seed (int): Random seed for reproducibility.
        - device (str): The device (cpu or cuda) to perform computations on.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.device = device
        torch.manual_seed(self.seed)

    @abstractmethod
    def select_action(self, state, eps=0.0):
        """
        Returns actions for given state as per current policy.
        
        Parameters:
        - state (array_like): Current state of the environment.
        - eps (float): Epsilon, for epsilon-greedy action selection.
        
        Returns:
        - action (int): Chosen action.
        """
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        """
        Update the agent's knowledge, using the most recently sampled tuple.

        Parameters:
        - state (array_like): The previous state of the environment.
        - action (int): The action taken by the agent.
        - reward (float): The reward received from the environment.
        - next_state (array_like): The next state of the environment.
        - done (bool): Whether the episode is complete (True or False).
        """
        pass

    @abstractmethod
    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Parameters:
        - experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done) tuples.
        - gamma (float): Discount factor.
        """
        pass

    @abstractmethod
    def save(self, filename):
        """
        Save the model weights.

        Parameters:
        - filename (str): Path to the file where the model weights will be saved.
        """
        pass

    @abstractmethod
    def load(self, filename):
        """
        Load the model weights.

        Parameters:
        - filename (str): Path to the file from which the model weights will be loaded.
        """
        pass

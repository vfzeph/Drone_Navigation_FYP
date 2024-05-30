import os
import sys
import json
import logging
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from stable_baselines3.common.vec_env import DummyVecEnv

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)

from Drone.source.envs.airsim_env import AirSimEnv
from Drone.source.models.nn.policy_network import AdvancedPolicyNetwork
from Drone.source.models.nn.critic_network import AdvancedCriticNetwork
from Drone.source.models.ppo.ppo_utils import compute_gae, normalize

def configure_logger(name, log_dir='./logs'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'ppo_agent.log'))
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append(self.experience(state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
        else:
            priorities = np.array(self.priorities)[:len(self.buffer)]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = self.experience(*zip(*experiences))
        
        states = np.vstack(batch.state)
        actions = np.vstack(batch.action)
        rewards = np.vstack(batch.reward)
        next_states = np.vstack(batch.next_state)
        dones = np.vstack(batch.done)
        
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

class PPOAgent:
    class Memory:
        def __init__(self):
            self.actions = []
            self.states = []
            self.log_probs = []
            self.rewards = []
            self.dones = []

        def reset(self):
            self.actions.clear()
            self.states.clear()
            self.log_probs.clear()
            self.rewards.clear()
            self.dones.clear()

        def add(self, action, state, log_prob, reward, done):
            action = np.atleast_1d(action)
            state = np.atleast_2d(state)

            self.actions.append(action)
            self.states.append(state)
            self.log_probs.append(log_prob)
            self.rewards.append(reward if reward is not None else 0.0)
            self.dones.append(done)

        def get_tensors(self, device):
            actions = np.vstack(self.actions)
            states = np.vstack(self.states)

            return (
                torch.tensor(actions, device=device).float(),
                torch.tensor(states, device=device).float(),
                torch.tensor(np.array(self.log_probs), device=device).float(),
                torch.tensor(np.array(self.rewards), device=device).float(),
                torch.tensor(np.array(self.dones), device=device).bool()
            )

    def __init__(self, config, action_space):
        self.config = config
        self.state_dim = config['policy_network']['input_size']
        self.action_dim = config['policy_network']['output_size']
        self.action_space = action_space
        self.logger = configure_logger(__name__, config['logging']['log_dir'])

        device_config = config['ppo']['device']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_config == 'auto' else torch.device(device_config)
        self.logger.info(f"Using device: {self.device}")

        self.policy_network = AdvancedPolicyNetwork(
            config['policy_network']['input_size'], 
            config['policy_network']['output_size'], 
            continuous=True, 
            hidden_sizes=config['policy_network']['hidden_layers']
        ).to(self.device)

        self.critic_network = AdvancedCriticNetwork(
            config['critic_network']['input_size'], 
            hidden_sizes=config['critic_network']['hidden_layers']
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self.policy_network.parameters()) + list(self.critic_network.parameters()), 
            lr=config['ppo']['learning_rate']
        )

        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.gamma = config['ppo']['gamma']
        self.tau = config['ppo']['gae_lambda']
        self.epsilon = config['ppo']['clip_range']
        self.k_epochs = config['ppo']['n_epochs']
        self.clip_grad = config['ppo']['max_grad_norm']
        self.continuous = True  # Assuming continuous action space for simplicity
        self.writer = SummaryWriter(config['ppo']['tensorboard_log'])
        self.memory = self.Memory()
        self.replay_buffer = PrioritizedReplayBuffer(capacity=10000)  # Initialize Prioritized Replay Buffer

        self.initial_epsilon = config['exploration']['initial_epsilon']
        self.epsilon_decay_rate = config['exploration']['epsilon_decay_rate']
        self.min_epsilon = config['exploration']['min_epsilon']
        self.epsilon = self.initial_epsilon

        self.action_value_estimates = np.zeros(self.action_space)

        self.policy_loss = None
        self.value_loss = None
        self.total_loss = None
        self.entropy = None

    def select_action(self, state):
        state = np.atleast_1d(state)
        self.logger.debug(f"Original state shape: {state.shape}")

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.logger.debug(f"Corrected state tensor shape: {state_tensor.shape}")

        with autocast(enabled=self.device.type == 'cuda'):
            if self.continuous:
                action_mean, action_std_log = self.policy_network(state_tensor)
                action_std = torch.exp(action_std_log)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            else:
                action_probs = self.policy_network(state_tensor)
                dist = torch.distributions.Categorical(probs=action_probs)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)

        action = action.cpu().numpy().flatten()
        action_log_prob = action_log_prob.detach().cpu().numpy().flatten()

        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(self.action_space)
            self.logger.info(f"Exploration: Random action {action_index} selected.")
        else:
            action_value = action[0]
            action_index = int(np.clip(np.round(action_value).astype(int), 0, self.action_space - 1))
            self.logger.info(f"Exploitation: Action {action_index} selected.")

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)

        self.memory.add(action_index, state, action_log_prob, 0, False)
        return action_index, action_log_prob

    def _update_action_values(self, reward, action_index):
        self.action_value_estimates[action_index] = (self.action_value_estimates[action_index] + reward) / 2
        self.logger.debug(f"Updated action value for action {action_index}: {self.action_value_estimates[action_index]}")

    def update(self):
        if len(self.replay_buffer) < self.config['ppo']['batch_size']:
            return  # Skip update if not enough samples in the buffer

        batch_size = self.config['ppo']['batch_size']
        beta = self.config['ppo'].get('beta', 0.4)
        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(batch_size, beta)

        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).float().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)
        weights = torch.tensor(weights).float().to(self.device)

        next_value = self.critic_network(next_states).detach()
        returns, advantages = compute_gae(next_value, rewards, dones, self.critic_network(states), self.gamma, self.tau)
        
        if self.config['advanced_training_techniques']['normalize_advantages']:
            returns = normalize(returns).view(-1, 1)
            advantages = normalize(advantages).view(-1, 1)

        old_log_probs = torch.tensor(np.array(self.memory.log_probs), device=self.device).float()

        for epoch in range(self.k_epochs):
            with autocast(enabled=self.device.type == 'cuda'):
                log_probs, state_values, entropy = self.evaluate(states, actions)
                ratios = torch.exp(log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
                self.policy_loss = -(torch.min(surr1, surr2) * weights).mean()

                state_values = state_values.view(-1, 1)
                self.value_loss = (weights * nn.functional.mse_loss(state_values, returns, reduction='none')).mean()
                
                self.total_loss = self.policy_loss + self.config['ppo']['vf_coef'] * self.value_loss - self.config['ppo']['ent_coef'] * entropy

            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(self.total_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.clip_grad)
                torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.total_loss.backward()
                self.optimizer.step()

            if epoch % (self.k_epochs // 2) == 0:
                self.logger.info(f'Epoch {epoch+1}/{self.k_epochs}, Policy Loss: {self.policy_loss.item()}, Value Loss: {self.value_loss.item()}, Total Loss: {self.total_loss.item()}, Entropy: {entropy.item()}')
                self.writer.add_scalar('Training/Policy Loss', self.policy_loss.item(), epoch)
                self.writer.add_scalar('Training/Value Loss', self.value_loss.item(), epoch)
                self.writer.add_scalar('Training/Total Loss', self.total_loss.item(), epoch)
                self.writer.add_scalar('Training/Entropy', entropy.item(), epoch)

        self.memory.reset()

    def evaluate(self, states, actions):
        with autocast(enabled=self.device.type == 'cuda'):
            if self.continuous:
                action_means, action_std_logs = self.policy_network(states)
                action_stds = torch.exp(action_std_logs)
                dist = torch.distributions.Normal(action_means, action_stds)
                log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().mean()
            else:
                action_probs = self.policy_network(states)
                dist = torch.distributions.Categorical(probs=action_probs)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

            state_values = self.critic_network(states).squeeze(1)
        return log_probs, state_values, entropy

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'critic_state_dict': self.critic_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        self.logger.info(f'Model saved at {path}')

    def load_model(self, path):
        checkpoint = torch.load(path)
        if 'policy_state_dict' in checkpoint and 'critic_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
            self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
            self.critic_network.load_state_dict(checkpoint['critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info('Model loaded')
        else:
            self.logger.warning('Checkpoint does not contain required keys: policy_state_dict, critic_state_dict, optimizer_state_dict')

if __name__ == '__main__':
    try:
        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'drone/configs/learning/ppo_config.json'))
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            ppo_config = json.load(f)

        action_space = np.array([
            (5, 0, 0, 0, 0, 0), (-5, 0, 0, 0, 0, 0),
            (0, 5, 0, 0, 0, 0), (0, -5, 0, 0, 0, 0),
            (0, 0, 5, 0, 0, 0), (0, 0, -5, 0, 0, 0),
            (0, 0, 0, 5, 0, 0), (0, 0, 0, -5, 0, 0),
            (0, 0, 0, 0, 5, 0), (0, 0, 0, 0, -5, 0),
            (0, 0, 0, 0, 0, 5), (0, 0, 0, 0, 0, -5),
            (0, 0, 0, 0, 0, 0)
        ])

        logger = configure_logger("AirSimEnvLogger", log_dir="./logs")
        env = AirSimEnv(state_dim=10, action_dim=4, logger=logger, tensorboard_log_dir="./tensorboard_logs", log_enabled=True)
        env = DummyVecEnv([lambda: env])

        agent = PPOAgent(ppo_config, env.action_space.n)

        total_timesteps = 100000
        for timestep in range(total_timesteps):
            state = env.reset()
            done = False
            while not done:
                action, log_prob = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.memory.add(action, state, log_prob, reward, done)
                state = next_state
                if done:
                    agent.update()

        agent.save_model('./models/ppo_trained_model.pth')

    except Exception as e:
        logging.error(f"An error occurred: {e}")

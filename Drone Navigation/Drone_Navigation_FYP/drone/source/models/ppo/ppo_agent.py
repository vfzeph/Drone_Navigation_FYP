import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',))
from drone.source.envs.airsim_env import AirSimEnv
from drone.source.models.nn.policy_network import AdvancedPolicyNetwork
from drone.source.models.ppo.ppo_utils import compute_gae, normalize

class CriticNetwork(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)  # Predicts state value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def add(self, action, state, log_prob, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, tau=0.95, epsilon=0.2, k_epochs=10, continuous=False, device=None):
        self.policy_network = AdvancedPolicyNetwork(state_dim, action_dim, continuous=continuous).to(device)
        self.critic_network = CriticNetwork(state_dim).to(device)
        self.optimizer = optim.Adam(list(self.policy_network.parameters()) + list(self.critic_network.parameters()), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.99)
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.k_epochs = k_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_history = []
        logging.basicConfig(level=logging.INFO)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.policy_network.continuous:
                action_mean, action_std = self.policy_network(state)
                dist = torch.distributions.Normal(action_mean, action_std)
            else:
                action_probs = self.policy_network(state)
                dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.cpu().numpy().flatten(), action_log_prob.item()

    def update(self, memory):
        states = torch.tensor(np.array(memory.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(memory.actions), dtype=torch.float32 if self.policy_network.continuous else torch.long).to(self.device)
        log_probs = torch.tensor(np.array(memory.log_probs), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(memory.rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(memory.dones), dtype=torch.float32).to(self.device)

        values = self.critic_network(states)
        next_values = torch.cat([values[1:], torch.zeros(1, 1, device=self.device)], dim=0)

        returns, advantages = compute_gae(next_values, rewards, dones, values, self.gamma, self.tau)
        advantages = normalize(advantages)

        self.update_policy(states, actions, log_probs, returns.squeeze(-1), advantages)

    def update_policy(self, states, actions, log_probs_old, returns, advantages):
        for _ in range(self.k_epochs):
            if self.policy_network.continuous:
                action_means, action_stds = self.policy_network(states)
                dist = torch.distributions.Normal(action_means, action_stds)
            else:
                action_probs = self.policy_network(states)
                dist = torch.distributions.Categorical(action_probs)

            log_probs = dist.log_prob(actions)
            state_values = self.critic_network(states).squeeze(-1)
            dist_entropy = dist.entropy().mean()

            ratios = torch.exp(log_probs - log_probs_old.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(state_values, returns)
            loss = policy_loss + 0.5 * value_loss - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())

            if len(self.loss_history) > 5 and np.mean(self.loss_history[-5:]) < 0.01:
                logging.info("Convergence reached, stopping training...")
                break

        self.scheduler.step()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.envs.airsim_env import AirSimDroneEnv
from src.models.nn.policy_network import EnhancedPolicyNetwork
from src.models.ppo.ppo_utils import compute_gae, normalize, to_tensor
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, input_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 1)  # Predicts state value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value

class PPOAgent:
    def __init__(self, policy_network, critic_network, lr=1e-4, gamma=0.99, tau=0.95, epsilon=0.2, k_epochs=10):
        self.policy_network = policy_network
        self.critic_network = critic_network
        self.optimizer = optim.Adam(list(policy_network.parameters()) + list(critic_network.parameters()), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.k_epochs = k_epochs

    def update_policy(self, states, actions, log_probs_old, returns, advantages):
        for _ in range(self.k_epochs):
            log_probs, state_values, dist_entropy = self.policy_network.evaluate(states, actions)
            ratios = torch.exp(log_probs - log_probs_old.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(state_values.squeeze(-1), returns)
            loss = policy_loss + 0.5 * value_loss - 0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Setup the environment, policy, critic, and PPO agent
env = AirSimDroneEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

policy_network = EnhancedPolicyNetwork(state_dim, action_dim, continuous=True)
critic_network = CriticNetwork(state_dim)
ppo_agent = PPOAgent(policy_network, critic_network)

def collect_data(env, policy_network, episodes=10):
    states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            state = to_tensor(state).unsqueeze(0)
            action, log_prob = policy_network.act(state)
            next_state, reward, done, _ = env.step(action.numpy())

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            log_probs.append(log_prob)

            state = next_state
    return states, actions, rewards, next_states, dones, log_probs

def train_agent(env, ppo_agent, episodes=10):
    states, actions, rewards, next_states, dones, log_probs = collect_data(env, ppo_agent.policy_network, episodes)
    states = torch.cat(states)
    actions = torch.cat(actions)
    log_probs = torch.cat(log_probs)
    rewards = to_tensor(rewards)
    next_states = to_tensor(next_states)
    dones = to_tensor(dones, dtype=torch.bool)
    
    returns, advantages = compute_gae(next_states, rewards, dones, ppo_agent.critic_network, ppo_agent.gamma, ppo_agent.tau)
    ppo_agent.update_policy(states, actions, log_probs, returns, advantages)

if __name__ == "__main__":
    for _ in range(100):  # Train for 100 iterations
        train_agent(env, ppo_agent, episodes=10)

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.envs.airsim_env import AirSimEnv
from src.models.nn.policy_network import AdvancedPolicyNetwork
from src.utils.my_logging import setup_logger


def load_config(config_path='configs/ppo_config.json'):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except IOError as e:
        print(f"Could not read file: {config_path}. {e}")
        exit()
        
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.logprobs = []

    def store_transition(self, state, action, reward, next_state, done, logprob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.logprobs.append(logprob)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.logprobs = []

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip, collision_penalty, continuous=False, layer_sizes=[256, 128, 64], activation_fn=nn.ReLU, dropout_prob=0.0):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.collision_penalty = collision_penalty

        self.policy = AdvancedPolicyNetwork(state_dim, action_dim, continuous, layer_sizes, activation_fn, dropout_prob)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = AdvancedPolicyNetwork(state_dim, action_dim, continuous, layer_sizes, activation_fn, dropout_prob)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.continuous = continuous
        self.memory = Memory()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            if self.continuous:
                action_mean, action_std = self.policy_old(state)
                action_distribution = torch.distributions.Normal(action_mean, action_std)
                action = action_distribution.sample()
                action_log_prob = action_distribution.log_prob(action).sum(dim=-1)
            else:
                action_probs = self.policy_old(state)
                action_distribution = torch.distributions.Categorical(action_probs)
                action = action_distribution.sample()
                action_log_prob = action_distribution.log_prob(action)
        return action.item(), action_log_prob.item()

    def update(self, memory):  # Corrected to accept memory as an argument
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.tensor(memory.states, dtype=torch.float32)
        old_actions = torch.tensor(memory.actions, dtype=torch.float32)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)

        for _ in range(self.K_epochs):
            logprobs, state_values = self.policy(old_states)
            dist = torch.distributions.Categorical(logprobs)
            new_logprobs = dist.log_prob(old_actions)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(new_logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist.entropy()

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        memory.clear_memory()  # Clear memory after updating

    def store_transition(self, state, action, reward, next_state, done, logprob):
        self.memory.store_transition(state, action, reward, next_state, done, logprob)

def train(config, env, ppo_agent, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    best_reward = -np.inf
    for episode in range(config['num_episodes']):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        episode_reward = 0
        memory = Memory()

        for t in range(config['max_timesteps_per_episode']):
            action, action_log_prob = ppo_agent.select_action(state.cpu().numpy())
            next_state, reward, done, collision = env.step(action)  # Update to include collision flag
            episode_reward += reward

            if collision:
                reward -= config['collision_penalty']  # Deduct collision penalty from reward
            
            memory.store_transition(state.cpu().numpy(), action, reward, next_state, done, action_log_prob)
            state = torch.tensor(next_state, dtype=torch.float32).to(device)

            if done:
                break
        
        ppo_agent.update()
        
        if episode % config['logging']['log_interval'] == 0:
            logger.info(f"Episode: {episode + 1}, Reward: {episode_reward}")
            if episode_reward > best_reward:
                best_reward = episode_reward
                policy_path = os.path.join(config['model_checkpointing']['checkpoint_dir'], 'policy_net_best.pth')
                torch.save(ppo_agent.policy.state_dict(), policy_path)
                logger.info("Saved new best model.")

def main(config_path='configs/ppo_config.json'):
    config = load_config(config_path)
    
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    env = AirSimEnv()  # Adjust as necessary for your environment setup
    
    ppo_config = config['ppo']
    ppo_agent = PPOAgent(
        state_dim=ppo_config['state_dim'], 
        action_dim=ppo_config['action_dim'], 
        lr=ppo_config['lr'], 
        gamma=ppo_config['gamma'], 
        K_epochs=ppo_config['K_epochs'], 
        eps_clip=ppo_config['eps_clip'],
        continuous=ppo_config.get('continuous', False),
        layer_sizes=ppo_config.get('layer_sizes', [256, 128, 64]),
        activation_fn=torch.nn.ReLU,
        dropout_prob=ppo_config.get('dropout_prob', 0.0)
    )

    logger = setup_logger('ppo_training', os.path.join(config['logging']['log_dir'], 'training.log'))

    train(config, env, ppo_agent, logger)

if __name__ == "__main__":
    main()

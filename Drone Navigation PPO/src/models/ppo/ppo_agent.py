import torch
import torch.nn.functional as F
import torch.optim as optim

class PPOAgent:
    def __init__(self, policy_model, critic_model, lr=3e-4, gamma=0.99, tau=0.95, epsilon=0.2, k_epochs=4, critic_coeff=0.5, entropy_coeff=0.01):
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.optimizer = optim.Adam(list(policy_model.parameters()) + list(critic_model.parameters()), lr=lr)
        
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.k_epochs = k_epochs
        self.critic_coeff = critic_coeff
        self.entropy_coeff = entropy_coeff

    def compute_gae(self, next_value, rewards, masks, values):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update_policy(self, states, actions, log_probs, returns, advantages):
        advantages = torch.tensor(advantages, dtype=torch.float32).detach()
        returns = torch.tensor(returns, dtype=torch.float32).detach()

        for _ in range(self.k_epochs):
            new_log_probs, state_values, dist_entropy = self.evaluate(states, actions)
            
            ratios = torch.exp(new_log_probs - log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(state_values.squeeze(-1), returns)
            
            loss = policy_loss + self.critic_coeff * value_loss - self.entropy_coeff * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def evaluate(self, states, actions):
        action_probs = self.policy_model(states)
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        state_values = self.critic_model(states)
        
        return action_log_probs, state_values, dist_entropy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        action_prob = self.actor(state)
        value = self.critic(state)
        return action_prob, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-3, lr_critic=2e-3, gamma=0.99, eps_clip=0.2, K_epochs=4, device="cpu"):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action_prob, _ = self.policy_old(state)
        action = torch.multinomial(action_prob, 1).cpu().data.numpy().flatten()
        memory.states.append(state)
        memory.actions.append(action)
        return action

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.tensor(memory.actions, dtype=torch.long)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values = self.policy(old_states)
            dist_entropy = -(logprobs * torch.exp(logprobs)).sum(dim=1).mean()

            state_values = torch.squeeze(state_values)

            new_logprobs = logprobs.gather(1, old_actions.unsqueeze(-1)).squeeze(-1)
            policy_ratio = torch.exp(new_logprobs - old_logprobs)

            advantages = rewards - state_values.detach()
            surr1 = policy_ratio * advantages
            surr2 = torch.clamp(policy_ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

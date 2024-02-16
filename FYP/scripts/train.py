from collections import deque
import gym
import numpy as np
import torch
from dqn_agent import DQNAgent  # Adjust this import based on your project structure
import matplotlib.pyplot as plt

def train_dqn(env_name='LunarLander-v2', n_episodes=2000, max_t=1000,
              eps_start=1.0, eps_end=0.01, eps_decay=0.995, target_score=200):
    """Function to train DQN agent.
    
    Params
    ======
        env_name (str): Environment name
        n_episodes (int): Maximum number of training episodes
        max_t (int): Maximum number of timesteps per episode
        eps_start (float): Starting value of epsilon
        eps_end (float): Minimum value of epsilon
        eps_decay (float): Multiplicative factor for epsilon decay
        target_score (float): Average score goal, after which training will stop
    """
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, seed=0)
    
    scores = []                        # List of scores from each episode
    scores_window = deque(maxlen=100)  # Last 100 scores
    eps = eps_start                    # Initialize epsilon for epsilon-greedy policy
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)  # Decrease epsilon
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        if np.mean(scores_window) >= target_score:
            print(f'\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    env.close()
    return scores

# Plotting function to visualize the scores
def plot_scores(scores):
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

if __name__ == "__main__":
    scores = train_dqn()
    plot_scores(scores)

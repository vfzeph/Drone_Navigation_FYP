import gym
import torch
from dqn_agent import DQNAgent  # Adjust this import based on your project structure
import numpy as np

def test_dqn(env_name='LunarLander-v2', n_episodes=100, model_path='checkpoint.pth'):
    """Function to test the trained DQN agent.
    
    Params
    ======
        env_name (str): Environment name
        n_episodes (int): Number of episodes to run the agent
        model_path (str): Path to the trained model weights
    """
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, seed=42)  # Initialize agent

    # Load the trained weights
    agent.qnetwork_local.load_state_dict(torch.load(model_path))

    total_scores = []  # List to store scores from each episode

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        while True:
            action = agent.act(state, eps=0.0)  # Select action based on policy (no exploration)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if done:
                break
        total_scores.append(score)
        print(f'Episode {i_episode} Score: {score:.2f}')
    
    print(f'Average Score over {n_episodes} episodes: {np.mean(total_scores):.2f}')
    env.close()

if __name__ == "__main__":
    test_dqn()

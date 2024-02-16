import argparse
import yaml
from pathlib import Path
import torch

# Import your project modules
from envs.unity_env_wrapper import DroneEnv
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from util.replay_buffer import ReplayBuffer

def load_config(config_path='config.yaml'):
    """Load the project configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_environment(config):
    """Initialize the drone environment."""
    env = DroneEnv()  # Add any necessary arguments based on your DroneEnv class
    return env

def select_agent(config, state_size, action_size, device):
    """Initialize the appropriate RL agent."""
    if config['agent']['type'] == 'PPO':
        agent = PPOAgent(state_size=state_size, action_size=action_size, config=config, device=device)
    elif config['agent']['type'] == 'DQN':
        agent = DQNAgent(state_size=state_size, action_size=action_size, config=config, device=device)
    else:
        raise ValueError("Unsupported agent type specified in config.")
    return agent

def train(agent, env, config):
    """The main training loop."""
    for episode in range(config['training']['n_episodes']):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode: {episode}, Total Reward: {total_reward}")
                break
    # Save the trained model
    agent.save('path_to_save_your_model')

def main(config_path='config.yaml'):
    config = load_config(config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = setup_environment(config)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = select_agent(config, state_size, action_size, device)
    train(agent, env, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the drone navigation training.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    main(config_path=args.config)

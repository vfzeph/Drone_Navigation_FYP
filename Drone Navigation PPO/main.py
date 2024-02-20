import os
import torch
import numpy as np
import json
from src.envs.airsim_env import AirSimEnv  # Ensure this import path is correct
from src.utils.my_logging import setup_logger
from src.models.ppo.ppo_agent import PPOAgent, Memory  # Adjust import paths as necessary

def load_config(config_path='configs/ppo_config.json'):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except IOError as e:
        print(f"Could not read file: {config_path}. {e}")
        exit()

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
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # Store experience in memory
            memory.store_transition(state.cpu().numpy(), action, reward, next_state, done, action_log_prob)
            state = torch.tensor(next_state, dtype=torch.float32).to(device)

            if done:
                break
        
        ppo_agent.update(memory)
        
        if episode % config['logging']['log_interval'] == 0:
            logger.info(f"Episode: {episode + 1}, Reward: {episode_reward}")
            if episode_reward > best_reward:
                best_reward = episode_reward
                policy_path = os.path.join(config['model_checkpointing']['checkpoint_dir'], 'policy_net_best.pth')
                torch.save(ppo_agent.policy.state_dict(), policy_path)
                logger.info("Saved new best model.")


def main(config_path='configs/ppo_config.json'):
    config = load_config(config_path)
    
    # Ensure the log directory exists
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    env = AirSimEnv()  # Adjust as necessary for your environment setup
    
    # Initialize PPOAgent with config parameters
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
        activation_fn=torch.nn.ReLU,  # Assuming ReLU; adjust if necessary
        dropout_prob=ppo_config.get('dropout_prob', 0.0),
        collision_penalty=config['collision_penalty']  # Include collision_penalty here
    )


    logger = setup_logger('ppo_training', os.path.join(config['logging']['log_dir'], 'training.log'))

    train(config, env, ppo_agent, logger)

if __name__ == "__main__":
    # Optionally, add argument parsing here to accept config_path from the command line
    main()

import torch
import numpy as np
import json
from envs.airsim_env import AirSimDroneEnv  # Ensure this path is correct
from src.models.nn.policy_network import EnhancedPolicyNetwork
from src.models.nn.critic_network import CriticNetwork
from src.models.ppo.ppo_agent import PPOAgent
from src.utils.my_logging import setup_logger
from torch.optim.lr_scheduler import StepLR
import os

def load_config(config_path='configs/ppo_config.json'):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except IOError as e:
        print(f"Could not read file: {config_path}. {e}")
        exit()

def main():
    config = load_config()
    
    # Setting device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = AirSimDroneEnv()
    policy_net = EnhancedPolicyNetwork(input_size=config['policy_network']['input_size'],
                                        output_size=config['policy_network']['output_size'],
                                        continuous=True).to(device)
    critic_net = CriticNetwork(input_size=config['critic_network']['input_size'],
                               output_size=config['critic_network']['output_size']).to(device)
    ppo_agent = PPOAgent(policy_network=policy_net,
                         critic_network=critic_net,
                         lr=config['learning_rate'],
                         gamma=config['gamma'],
                         tau=config['tau'],
                         epsilon=config['epsilon'],
                         k_epochs=config['k_epochs'],
                         batch_size=config['batch_size'])

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['lr_scheduler_step'], gamma=config['lr_scheduler_gamma'])
    
    logger = setup_logger('ppo_training', 'training.log')
    best_reward = -np.inf

    for episode in range(config['num_episodes']):
        state = env.reset()
        state = torch.tensor(state, device=device).float()
        episode_reward = 0
        ppo_agent.buffer.clear()

        for t in range(config['max_timesteps_per_episode']):
            action, action_log_prob, _ = ppo_agent.select_action(state.unsqueeze(0))
            
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            next_state = torch.tensor(next_state, device=device).float()
            episode_reward += reward
            ppo_agent.store_transition(state.cpu().numpy(), action.cpu().numpy(), reward, next_state.cpu().numpy(), done, action_log_prob.cpu().numpy())

            state = next_state
            if done:
                break

        # Perform PPO update
        ppo_agent.update_policy(optimizer, device)

        # Logging and saving
        logger.info(f"Episode: {episode+1}, Reward: {episode_reward}")
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(policy_net.state_dict(), f"models/policy_net_best.pth")
            torch.save(critic_net.state_dict(), f"models/critic_net_best.pth")
            logger.info("Saved new best model.")

        scheduler.step()

if __name__ == "__main__":
    main()

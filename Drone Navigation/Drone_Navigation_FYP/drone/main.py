import argparse
import json
import os
import sys
import datetime
import torch
import numpy as np
import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from source.envs.airsim_env import AirSimEnv
from source.utils.my_logging import setup_logger
from sympy import evaluate

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging episode rewards to TensorBoard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self):
        if self.locals['dones'][0]:
            episode_reward = np.sum(self.episode_rewards)
            self.logger.record('episode_reward', episode_reward)
            self.episode_rewards = []
        else:
            self.episode_rewards.append(self.locals['rewards'][0])
        return True

def load_config(config_path='configs/ppo_config.json'):
    """Load the configuration JSON file from the given path."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_config_path = os.path.join(base_dir, config_path)
    try:
        with open(full_config_path, 'r') as file:
            config = json.load(file)
        return config
    except IOError as e:
        print(f"Could not read file: {full_config_path}. {e}")
        sys.exit(1)

def train(config, env, logger):
    """
    Set up and train the PPO agent.
    """
    # Define valid PPO configuration keys based on Stable Baselines3 documentation
    valid_ppo_keys = {'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
                      'gae_lambda', 'clip_range', 'clip_range_vf', 'ent_coef', 'vf_coef',
                      'max_grad_norm', 'use_sde', 'sde_sample_freq', 'target_kl', 'tensorboard_log',
                      'create_eval_env', 'policy_kwargs', 'verbose', 'seed', 'device', 'policy'}
    
    # Filter the config to include only valid keys
    filtered_ppo_config = {k: v for k, v in config['ppo'].items() if k in valid_ppo_keys}

    # Initialize the model with the filtered configuration
    model = PPO("MlpPolicy", env, **filtered_ppo_config)
    model.learn(total_timesteps=config['num_timesteps'], callback=TensorboardCallback())
    model.save(os.path.join(config['model_checkpointing']['checkpoint_dir'], 'ppo_airsim_model'))
    logger.info("Training completed and model saved.")

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate a PPO agent using Stable Baselines3.')
    parser.add_argument('--config_path', type=str, default='configs/ppo_config.json', help='Path to configuration file')
    args = parser.parse_args()

    config = load_config(args.config_path)

    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    env = DummyVecEnv([lambda: AirSimEnv()])

    logger = setup_logger('ppo_training', os.path.join(config['logging']['log_dir'], 'training.log'))

    train(config, env, logger)

    model_path = os.path.join(config['model_checkpointing']['checkpoint_dir'], 'ppo_airsim_model.zip')
    evaluate(model_path, env)

if __name__ == "__main__":
    main()

import airsim
import numpy as np
import time
import os
import sys
import gym
from gym import spaces
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)

from Drone.source.models.nn.common_layers import ICM  # Import the necessary class

# Custom Logger
class CustomLogger:
    def __init__(self, name, log_dir=None):
        self.name = name
        self.log_dir = log_dir
        self._setup_logger()

    def _setup_logger(self):
        import logging
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(self.log_dir, f'{self.name}.log'))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

# Domain Randomization
def randomize_environment(client):
    weather_params = [
        airsim.WeatherParameter.Rain,
        airsim.WeatherParameter.Enabled
    ]
    weather = random.choice(weather_params)
    client.simSetWeatherParameter(weather, 0)

# AirSim Environment
class AirSimEnv(gym.Env):
    def __init__(self, state_dim, action_dim, target_position=np.array([0, 0, -10]), action_frequency=1.0, log_enabled=False,
                 exploration_strategy="epsilon_greedy", epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1,
                 temperature=1.0, ucb_c=2.0, logger=None, tensorboard_log_dir=None):
        super(AirSimEnv, self).__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.target_position = target_position
        self.action_frequency = action_frequency
        self.start_time = None
        self.max_episode_duration = 120
        self.exploration_area = {
            "x_min": -1000, "x_max": 1000,
            "y_min": -1000, "y_max": 1000,
            "z_min": -100, "z_max": 100
        }
        self.action_space = spaces.Discrete(len([
            (20, 0, 0), (-20, 0, 0),  # Forward, Backward
            (0, 20, 0), (0, -20, 0),  # Right, Left
            (0, 0, 20), (0, 0, -20),  # Up, Down
            (0, 0, 0)  # Hover
        ]))  # Increased the velocity for each action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.log_enabled = log_enabled
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.temperature = temperature
        self.ucb_c = ucb_c
        self.action_counts = np.zeros(self.action_space.n)
        self.action_values = np.zeros(self.action_space.n)
        self.total_steps = 0
        self.prev_action = np.array([0, 0, 0])
        self.logger = logger
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir) if tensorboard_log_dir else None
        self.prev_state = None  # Used for curiosity-based reward

        # Initialize ICM module
        self.icm = ICM(state_dim, action_dim)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)

        # Randomize environment
        self.randomize_environment()

    def log(self, message):
        if self.log_enabled:
            if self.logger:
                self.logger.info(message)
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def randomize_environment(self):
        randomize_environment(self.client)

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.start_time = time.time()
        self.prev_action = np.array([0, 0, 0])
        self.action_counts = np.zeros(self.action_space.n)
        self.action_values = np.zeros(self.action_space.n)
        self.total_steps = 0
        self.prev_state = None
        self.log("Environment reset and takeoff completed.")
        return self._get_state()

    def step(self, action_index):
        action = self._map_action(action_index)
        action = action + np.random.normal(0, 0.1, size=action.shape)
        duration = 0.5 / self.action_frequency  # Decreased duration to increase frequency of actions

        action = self._smooth_action(action)
        vx, vy, vz = float(action[0]), float(action[1]), float(action[2])
        self.client.moveByVelocityAsync(vx, vy, vz, duration).join()
        time.sleep(duration)
        new_state = self._get_state()
        reward = self._compute_reward(new_state, action)
        done = self._check_done(new_state)
        self.log(f"Action: {action_index}, Reward: {reward}, Done: {done}")
        self._update_action_values(action_index, reward)
        if self.writer:
            self.writer.add_scalar('Reward', reward, self.total_steps)
            self.writer.add_scalar('Epsilon', self.epsilon, self.total_steps)
            self.writer.flush()
        return new_state, reward, done, {}

    def _map_action(self, action_index):
        return np.array([
            (20, 0, 0), (-20, 0, 0),  # Forward, Backward
            (0, 20, 0), (0, -20, 0),  # Right, Left
            (0, 0, 20), (0, 0, -20),  # Up, Down
            (0, 0, 0)  # Hover
        ])[action_index]  # Increased the velocity for each action

    def _smooth_action(self, action):
        action = np.tanh(action)  # Apply non-linear transformation (hyperbolic tangent)
        action = self.prev_action + 0.5 * (action - self.prev_action)
        self.prev_action = action
        return action

    def _get_state(self):
        pose = self.client.simGetVehiclePose().position
        velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        orientation = self.client.simGetVehiclePose().orientation
        position = np.array([pose.x_val, pose.y_val, pose.z_val])
        linear_velocity = np.array([velocity.x_val, velocity.y_val, velocity.z_val])
        orientation_quat = np.array([orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val])
        state = np.concatenate((position, linear_velocity, orientation_quat), axis=0)
        return state

    def _compute_reward(self, state, action):
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        reward = -distance_to_target  # Negative reward for distance

        # Add more granular rewards and penalties
        potential_reward = self._compute_potential_reward(distance_to_target)
        reward += potential_reward

        collision_penalty = self._compute_collision_penalty()
        reward += collision_penalty

        height_penalty = self._compute_height_penalty(position[2])
        reward += height_penalty

        time_penalty = self._compute_time_penalty()
        reward += time_penalty

        movement_penalty = self._compute_movement_penalty(action)
        reward += movement_penalty

        smoothness_penalty = self._compute_smoothness_penalty(action)
        reward += smoothness_penalty

        curiosity_reward = self._compute_curiosity_reward(state, action)
        reward += curiosity_reward

        exploration_bonus = self._compute_exploration_bonus(action)
        reward += exploration_bonus

        return reward

    def _compute_potential_reward(self, distance_to_target):
        return 1.0 / (1.0 + distance_to_target)

    def _compute_collision_penalty(self):
        collision_info = self.client.simGetCollisionInfo()
        return -50 if collision_info.has_collided else 0

    def _compute_height_penalty(self, current_height):
        height_target = self.target_position[2]
        height_tolerance = 1.0
        height_penalty = 1.0
        return -height_penalty if abs(current_height - height_target) > height_tolerance else 0

    def _compute_time_penalty(self):
        time_penalty = 0.01
        return -time_penalty * (time.time() - self.start_time)

    def _compute_movement_penalty(self, action):
        movement_penalty = 0.1
        return -movement_penalty * np.linalg.norm(action)

    def _compute_smoothness_penalty(self, action):
        smoothness_penalty = 0.1
        return -smoothness_penalty * np.linalg.norm(action - self.prev_action)

    def _compute_curiosity_reward(self, state, action):
        curiosity_reward = self.icm.intrinsic_reward(
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(self.prev_state, dtype=torch.float32) if self.prev_state is not None else torch.zeros_like(torch.tensor(state, dtype=torch.float32)),
            torch.tensor(action, dtype=torch.float32)
        )
        self.prev_state = state
        return curiosity_reward

    def _compute_exploration_bonus(self, action):
        exploration_bonus = 0.1 * np.linalg.norm(action - self.prev_action)
        return exploration_bonus

    def _update_action_values(self, action_index, reward):
        self.action_counts[action_index] += 1
        n = self.action_counts[action_index]
        value = self.action_values[action_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.action_values[action_index] = new_value

    def _check_done(self, state):
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        if distance_to_target < 1.0:
            self.log("Target reached.")
            return True
        if time.time() - self.start_time > self.max_episode_duration:
            self.log("Episode timed out.")
            return True
        return False

if __name__ == '__main__':
    logger = CustomLogger("AirSimEnvLogger", log_dir="./logs")
    env = AirSimEnv(state_dim=10, action_dim=3, logger=logger, tensorboard_log_dir="./tensorboard_logs", log_enabled=True)
    state = env.reset()
    done = False
    while not done:
        action = np.random.randint(env.action_space.n)
        state, reward, done, _ = env.step(action)

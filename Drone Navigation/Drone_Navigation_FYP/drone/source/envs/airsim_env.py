import gym
import airsim
import numpy as np
import time
from gym import spaces

class AirSimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, target_position=np.array([100, 100, -10]), action_frequency=1.0, log_enabled=False):
        super(AirSimEnv, self).__init__()
        self.client = None
        self.target_position = target_position
        self.action_frequency = action_frequency
        self.start_time = None
        self.max_episode_duration = 180  # Allowing more time for complex tasks
        self.action_space = spaces.Discrete(8)  # Including diagonal movement
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.log_enabled = log_enabled
        self.connect_to_client()

    def connect_to_client(self):
        """Establishes connection to the AirSim simulation environment."""
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def log(self, message):
        """Logs messages to the console if logging is enabled."""
        if self.log_enabled:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def reset(self):
        """Resets the environment to start a new episode."""
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.start_time = time.time()
        self.log("Environment reset and takeoff completed.")
        return self._get_state()

    def step(self, action_index):
        """Executes the given action and returns the new state, reward, and whether the episode is done."""
        actions = [
            (1, 0, 0),  # Forward
            (-1, 0, 0), # Backward
            (0, 1, 0),  # Right
            (0, -1, 0), # Left
            (0, 0, 1),  # Up
            (0, 0, -1), # Down
            (1, 1, 0),  # Diagonal forward-right
            (0, 0, 0)   # Hover
        ]
        if not (0 <= action_index < len(actions)):
            raise ValueError(f"Invalid action index: {action_index}. Must be an integer within range [0, {len(actions) - 1}].")

        vx, vy, vz = actions[action_index]
        vx, vy, vz = vx * 2, vy * 2, vz * 2  # Apply scaling factor for more dynamic movement.
        duration = 1 / self.action_frequency
        self.client.moveByVelocityAsync(vx, vy, vz, duration).join()
        time.sleep(duration)
        
        new_state = self._get_state()
        reward = self._compute_reward(new_state)
        done = self._check_done(new_state)
        self.log(f"Action: {action_index}, State: {new_state}, Reward: {reward}, Done: {done}")
        return new_state, float(reward), done, {}

    def _get_state(self):
        """Fetches the current state of the drone including position, orientation, and velocity."""
        pose = self.client.simGetVehiclePose()
        position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
        orientation = np.array([pose.orientation.w_val, pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val])
        velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        linear_velocity = np.array([velocity.x_val, velocity.y_val, velocity.z_val])
        state = np.concatenate((position, orientation, linear_velocity))  # Includes orientation in the state vector.
        return state

    def _compute_reward(self, state):
        """Calculates the reward based on the distance from the target position."""
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        reward = -distance_to_target  # Negative reward based on the distance to encourage minimization.
        return float(reward)

    def _check_done(self, state):
        """Checks whether the episode should end based on the target proximity or elapsed time."""
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        if distance_to_target < 1.0 or (time.time() - self.start_time > self.max_episode_duration):
            self.log("Mission concluded.")
            return True
        return False

    def close(self):
        """Cleans up by disabling API control and resetting the simulation."""
        if self.client:
            self.client.armDisarm(False)
            self.client.reset()
            self.client.enableApiControl(False)
            self.client = None
            print("AirSim client connection closed.")


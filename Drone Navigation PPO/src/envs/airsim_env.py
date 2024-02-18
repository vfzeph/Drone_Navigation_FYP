import airsim
import numpy as np
import time  # Use standard Python time module for clarity and reliability

class AirSimEnv:
    def __init__(self, target_position=np.array([0, 0, -10]), log_enabled=False):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.target_position = target_position
        self.start_time = None
        self.max_episode_duration = 120  # seconds
        self.action_space = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0, 0, 0)]  # Explicit action space definition
        self.log_enabled = log_enabled

    def log(self, message):
        if self.log_enabled:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()  # Ensuring takeoff completes
        self.start_time = time.time()
        self.log("Environment reset and takeoff completed.")
        return self._get_state()

    def step(self, action):
        vx, vy, vz = self.action_space[action]
        self.client.moveByVelocityAsync(vx, vy, vz, 1).join()  # Using join for simplicity, consider asynchronous patterns for more complex scenarios
        new_state = self._get_state()
        reward = self._compute_reward(new_state)
        done = self._check_done(new_state)
        self.log(f"Action: {action}, Reward: {reward}, Done: {done}")
        return new_state, reward, done, {}

    def _get_state(self):
        pose = self.client.simGetVehiclePose().position
        velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        position = np.array([pose.x_val, pose.y_val, pose.z_val])
        linear_velocity = np.array([velocity.x_val, velocity.y_val, velocity.z_val])
        # Consider adding more state information here
        state = np.concatenate((position, linear_velocity))
        return state

    def _compute_reward(self, state):
        position = state[:3]  # Extracting position
        distance_to_target = np.linalg.norm(self.target_position - position)
        reward = -distance_to_target  # Simple distance-based reward; consider more complex functions
        # Potential for additional reward components here
        return reward

    def _check_done(self, state):
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        if distance_to_target < 1.0:  # Threshold for success
            self.log("Target reached.")
            return True
        if time.time() - self.start_time > self.max_episode_duration:
            self.log("Episode timed out.")
            return True
        return False

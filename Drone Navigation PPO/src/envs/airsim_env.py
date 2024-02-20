import airsim
import numpy as np
import time

class AirSimEnv:
    def __init__(self, target_position=np.array([0, 0, -10]), action_frequency=1.0, log_enabled=False):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.target_position = target_position
        self.action_frequency = action_frequency  # Frequency of actions in seconds
        self.start_time = None
        self.max_episode_duration = 120  # Maximum duration of an episode in seconds
        self.action_space = np.array([
            (1, 0, 0), (-1, 0, 0),  # Forward, Backward
            (0, 1, 0), (0, -1, 0),  # Right, Left
            (0, 0, 1), (0, 0, -1),  # Up, Down
            (0, 0, 0)  # Hover
        ])
        self.log_enabled = log_enabled

    def log(self, message):
        if self.log_enabled:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.start_time = time.time()
        self.log("Environment reset and takeoff completed.")
        return self._get_state()

    def step(self, action_index):
        vx, vy, vz = self.action_space[action_index] * 5  # Results in NumPy types
        duration = 1 / self.action_frequency
        # Convert to native Python types
        vx, vy, vz, duration = float(vx), float(vy), float(vz), float(duration)
        self.client.moveByVelocityAsync(vx, vy, vz, duration).join()
        time.sleep(duration)  # Wait for the action to complete
        new_state = self._get_state()
        reward = self._compute_reward(new_state)
        done = self._check_done(new_state)
        self.log(f"Action: {action_index}, Reward: {reward}, Done: {done}")
        return new_state, reward, done, {}


    def _get_state(self):
        pose = self.client.simGetVehiclePose().position
        velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        position = np.array([pose.x_val, pose.y_val, pose.z_val])
        linear_velocity = np.array([velocity.x_val, velocity.y_val, velocity.z_val])
        state = np.concatenate((position, linear_velocity), axis=0)
        return state

    def _compute_reward(self, state):
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        reward = -distance_to_target  # Negative reward based on distance to target
        return reward

    def _check_done(self, state):
        position = state[:3]
        distance_to_target = np.linalg.norm(self.target_position - position)
        if distance_to_target < 1.0:  # Consider done if within 1 meter of the target
            self.log("Target reached.")
            return True
        if time.time() - self.start_time > self.max_episode_duration:  # Episode timeout
            self.log("Episode timed out.")
            return True
        return False

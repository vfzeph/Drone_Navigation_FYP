import os
import json
import numpy as np
import airsim
import gym
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address, image_shape, env_config, input_mode):
        self.image_shape = image_shape
        self.sections = env_config.get("sections", [])
        self.input_mode = input_mode

        self.drone = airsim.MultirotorClient(ip=ip_address)

        if self.input_mode == "multi_rgb":
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(image_shape[0], image_shape[1] * 3, 1),
                dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=self.image_shape, dtype=np.uint8)

        self.action_space = gym.spaces.Box(
            low=-0.6, high=0.6, shape=(2,), dtype=np.float32)

        self.info = {"collision": False}
        self.collision_time = 0
        self.random_start = True
        self.setup_flight()

    def step(self, action):
        self.do_action(action)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        return obs, reward, done, info

    def reset(self):
        self.setup_flight()
        obs, _ = self.get_obs()
        return obs

    def render(self):
        return self.get_obs()

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Prevent drone from falling after reset
        self.drone.moveToZAsync(-1, 1)

        # Get collision time stamp
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp

        # Get a random section
        if self.random_start:
            self.target_pos_idx = np.random.randint(len(self.sections))
        else:
            self.target_pos_idx = 0

        section = self.sections[self.target_pos_idx]
        self.agent_start_pos = section.get("offset", (0, 0, 0))[0]
        self.target_pos = section.get("target", (0, 0))

        # Start the agent at random section at a random yz position
        y_pos, z_pos = ((np.random.rand(1, 2) - 0.5) * 2).squeeze()
        pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos, y_pos, z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)

    def do_action(self, action):
        # Execute action
        self.drone.moveByVelocityBodyFrameAsync(
            0.4, float(action[0]), float(action[1]), duration=1).join()

        # Prevent swaying
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)

    def get_obs(self):
        self.info["collision"] = self.is_collision()

        if self.input_mode == "multi_rgb":
            obs_t = self.get_rgb_image()
            obs_t_gray = cv2.cvtColor(obs_t, cv2.COLOR_BGR2GRAY)
            obs = np.dstack((obs_t_gray,) * 3)
        elif self.input_mode == "single_rgb":
            obs = self.get_rgb_image()
        elif self.input_mode == "depth":
            obs = self.get_depth_image(thresh=3.4).reshape(self.image_shape)
            obs = ((obs / 3.4) * 255).astype(int)
        return obs, self.info

    def compute_reward(self):
        reward = 0
        done = 0

        # Add reward computation logic here

        return reward, done

    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return current_collision_time != self.collision_time

    def get_rgb_image(self):
        rgb_image_request = airsim.ImageRequest(
            0, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3))
        return img2d

    def get_depth_image(self, thresh=2.0):
        depth_image_request = airsim.ImageRequest(
            1, airsim.ImageType.DepthPerspective, True, False)
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = depth_image.reshape(responses[0].height, responses[0].width)
        depth_image[depth_image > thresh] = thresh
        return depth_image


class TestEnv(AirSimDroneEnv):
    def __init__(self, ip_address, image_shape, env_config, input_mode, test_mode):
        self.start_pos = -1
        super(TestEnv, self).__init__(ip_address, image_shape, env_config, input_mode)
        self.test_mode = test_mode
        self.total_traveled = 0
        self.eps_n = 0
        self.eps_success = 0

        if self.test_mode == "sequential":
            print("Enter start position \n0: easy, 20: medium, 40: hard")
            self.start_pos = int(input())

    def setup_flight(self):
        super(TestEnv, self).setup_flight()

        if self.start_pos != -1:
            self.agent_start_pos = self.start_pos

        # Start the agent at a random yz position
        y_pos, z_pos = ((np.random.rand(1, 2) - 0.5) * 2).squeeze()
        pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos, y_pos, z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)

    def compute_reward(self):
        reward = 0
        done = 0

        # Add reward computation logic here

        return reward, done

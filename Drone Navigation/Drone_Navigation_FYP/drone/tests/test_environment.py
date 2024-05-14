import unittest
import numpy as np
import os
import sys

# Ensure the directory containing the custom environment is on the Python path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from drone.source.envs.airsim_env import AirSimEnv

class TestAirSimDroneEnv(unittest.TestCase):
    def setUp(self):
        """Initialize the environment before each test."""
        self.env = AirSimEnv()

    def test_initialization(self):
        """Test environment initialization and reset."""
        initial_state = self.env.reset()
        # Check if the initial state is a numpy array and has the correct shape.
        self.assertIsInstance(initial_state, np.ndarray, "Initial state should be a numpy array")
        self.assertEqual(initial_state.shape[0], 9, "Initial state should have the correct length")

    def test_step(self):
        """Test the environment's step function with a dummy action."""
        self.env.reset()
        dummy_action = 0  # Assuming valid index for discrete action space
        next_state, reward, done, _ = self.env.step(dummy_action)
        
        # Check types and structures of return values from step method
        self.assertIsInstance(next_state, np.ndarray, "Next state should be a numpy array")
        self.assertIsInstance(reward, float, "Reward should be a float")
        self.assertIsInstance(done, bool, "Done should be a boolean")

    def test_reset(self):
        """Test environment reset functionality."""
        initial_state = self.env.reset()
        # Perform some actions
        for _ in range(5):
            action = np.random.choice(self.env.action_space.n)  # Assuming a discrete action space
            self.env.step(action)
        
        # Reset and get a new initial state
        new_initial_boat = self.env.reset()

        # Check if the new initial state differs from the initial state before reset
        self.assertNotEqual(np.array_equal(initial_state, new_initial_boat), True, "New state should differ from state before reset")

if __name__ == '__main__':
    unittest.main()

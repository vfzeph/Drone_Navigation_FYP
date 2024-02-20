import unittest
import numpy as np
from src.envs.airsim_env import AirSimDroneEnv

class TestAirSimDroneEnv(unittest.TestCase):
    def setUp(self):
        """Initialize the environment before each test."""
        self.env = AirSimDroneEnv()
    
    def test_initialization(self):
        """Test environment initialization and reset."""
        initial_state = self.env.reset()
        # Assuming the state is a numpy array; adjust according to your implementation
        self.assertIsInstance(initial_state, np.ndarray, "Initial state should be a numpy array")
    
    def test_step(self):
        """Test stepping through the environment."""
        self.env.reset()
        # Example dummy action; adjust based on your environment's action space
        dummy_action = np.array([0, 0, 0])  
        next_state, reward, done, _ = self.env.step(dummy_action)
        
        self.assertIsInstance(next_state, np.ndarray, "Next state should be a numpy array")
        self.assertIsInstance(reward, float, "Reward should be a float")
        self.assertIsInstance(done, bool, "Done should be a boolean")
    
    def test_reset(self):
        """Test environment reset functionality."""
        initial_state = self.env.reset()
        # Perform some steps
        for _ in range(5):
            self.env.step(np.array([0, 0, 0]))  # Adjust this dummy action as needed
        
        # Reset and get new initial state
        new_initial_state = self.env.reset()
        
        # Check if the new initial state is correctly reset; specifics depend on your env
        self.assertEqual(initial_state.shape, new_initial_state.shape, "State shape should match after reset")
        # Further checks can be added here based on expected reset behavior

if __name__ == '__main__':
    unittest.main()

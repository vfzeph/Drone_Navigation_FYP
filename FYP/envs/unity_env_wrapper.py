from mlagents_envs.environment import UnityEnvironment
import numpy as np

class UnityEnvWrapper:
    def __init__(self, env_path, worker_id=0, seed=None):
        self.env_path = env_path
        self.worker_id = worker_id
        self.seed = seed
        self.env = None

    def reset(self):
        self.env.reset()
        decision_steps, _ = self.env.get_steps()
        return self._process_decision_steps(decision_steps)

    def step(self, action):
        self.env.set_actions(self.env.external_brain_names[0], action)
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps()
        obs = self._process_decision_steps(decision_steps)
        rewards = self._process_terminal_steps(terminal_steps)
        done = terminal_steps[0] is not None
        info = {}
        return obs, rewards, done, info

    def _process_decision_steps(self, decision_steps):
        obs = decision_steps.observation
        # Process observation if needed (e.g., convert to numpy array)
        return np.array(obs)

    def _process_terminal_steps(self, terminal_steps):
        if terminal_steps[0] is not None:
            return terminal_steps[0].reward
        return 0  # No reward if episode is not done

    def close(self):
        self.env.close()

    def __enter__(self):
        self.env = UnityEnvironment(file_name=self.env_path, worker_id=self.worker_id, seed=self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    with UnityEnvWrapper(env_path="path_to_your_environment_file") as env:
        obs = env.reset()
        done = False
        while not done:
            action = np.random.randint(env.action_space.n)
            obs, reward, done, info = env.step(action)
            print(f"Observation: {obs}, Reward: {reward}, Done: {done}")

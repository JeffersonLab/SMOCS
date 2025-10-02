from gymnasium import spaces
import gymnasium as gym
import numpy as np

class IndustryParticleAcceleratorEnv(gym.Env):
    """Continuous control Particle Accelerator environment."""
    
    def __init__(self, gym_space_type=None):
#        super(IndustryParticleAcceleratorEnv, self).__init__(gym_space_type=None)
        
        # Define action space: continuous values for beam energy adjustment and conveyor speed
        # Beam Energy Level and Conveyor Speed can vary between 0.5 (low) and 2.0 (high)
        self.gym_space_type = gym_space_type

        if self.gym_space_type == 'Dict':
            # Define the action space using a dictionary
            self.action_space = gym.spaces.Dict({
                "beam_energy_change": gym.spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32),
                "conveyor_speed_change": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            })
        else:
            self.action_space = spaces.Box(low=np.array([-0.5, -1.0]), high=np.array([0.5, 1.0]), dtype=np.float32)

        # Define observation space: continuous values for beam energy level, dose accumulated,
        # leakage level, and conveyor speed
        if self.gym_space_type == 'Dict':
            # Define the action space using a dictionary
            self.observation_space = gym.spaces.Dict({
                "beam_energy": gym.spaces.Box(low=-0.5, high=2.0, shape=(1,), dtype=np.float32),
                "dose_accumulation": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "radiation_leakage_level": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "conveyor_speed": gym.spaces.Box(low=0.5, high=2.0, shape=(1,), dtype=np.float32),
            })
        else:
            self.observation_space = spaces.Box(
                low=np.array([0.5, 0.0, 0.0, 0.5]),
                high=np.array([2.0, np.inf, 1.0, 2.0]),
                dtype=np.float32
            )

        # Initial state variables
        self.beam_energy_level = 1.0  # start with medium energy level
        self.dose_accumulated = 0.0
        self.leakage_level = 0.0
        self.conveyor_speed = 1.0  # normal speed initially
        self.step_count = 0

    def reset(self):
        """Reset the environment to an initial state."""
        
        self.beam_energy_level = 1.0
        self.dose_accumulated = 0.0
        self.leakage_level = 0.0
        self.conveyor_speed = 1.0
        self.step_count = 0

        if self.gym_space_type == 'Dict':
            state = {
                "beam_energy": np.array([self.beam_energy_level]),
                "dose_accumulation": np.array([self.dose_accumulated]),
                "radiation_leakage_level": np.array([self.leakage_level]),
                "conveyor_speed": np.array([self.conveyor_speed])
            }
        else:
            state = np.array([self.beam_energy_level, self.dose_accumulated, self.leakage_level, self.conveyor_speed])

        return state, {}

    def step(self, action):
        """Take an action and observe the result."""
        
        # Apply actions to adjust beam energy level and conveyor speed
        if self.gym_space_type == 'Dict':
            beam_energy_adjustment = action["beam_energy_change"]
            speed_adjustment = action["conveyor_speed_change"]
        else:
            beam_energy_adjustment, speed_adjustment = action

        self.beam_energy_level += beam_energy_adjustment
        self.conveyor_speed += speed_adjustment

        # Ensure state variables remain within bounds
        if self.gym_space_type == 'Dict':
            self.beam_energy_level = np.clip(self.beam_energy_level,
                                             self.observation_space['beam_energy'].low[0],
                                             self.observation_space['beam_energy'].high[0])
            self.conveyor_speed = np.clip(self.conveyor_speed,
                                          self.observation_space['conveyor_speed'].low[0],
                                          self.observation_space['conveyor_speed'].high[0])
        else:
            self.beam_energy_level = np.clip(self.beam_energy_level,
                                             self.observation_space.low[0],
                                             self.observation_space.high[0])
            self.conveyor_speed = np.clip(self.conveyor_speed,
                                          self.observation_space.low[3],
                                          self.observation_space.high[3])

        # Simulate radiation dose accumulation based on the beam energy and conveyor speed
        dose_increment = self.beam_energy_level/self.conveyor_speed # Using simple proportional estimation
        self.dose_accumulated += dose_increment

        # Increase leakage level with higher energy, decrease it otherwise
        leakage_change = (self.beam_energy_level / 2) - 0.1
        self.leakage_level += leakage_change

        if self.gym_space_type == 'Dict':
            self.beam_energy_level = np.clip(self.leakage_level,
                                             self.observation_space['radiation_leakage_level'].low[0],
                                             self.observation_space['radiation_leakage_level'].high[0])
        else:
            self.leakage_level = np.clip(self.leakage_level,
                                          self.observation_space.low[2],
                                          self.observation_space.high[2])

        self.step_count += 1
        
        truncate = self.step_count >= 100
        terminate = False
        # Reward: positive for reaching target dose, negative penalties for high leakage and excessive steps
        reward = dose_increment - leakage_change
        if truncate:
            if self.dose_accumulated >= 10 and self.leakage_level <= 0.5:
                reward += 50  # bonus for completing the task within safe parameters
            else:
                reward -= 50  # penalty for failing to meet requirements or safety

        if self.gym_space_type == 'Dict':
            observation = {
                "beam_energy": np.array([self.beam_energy_level]),
                "dose_accumulation": np.array([self.dose_accumulated]),
                "radiation_leakage_level": np.array([self.leakage_level]),
                "conveyor_speed": np.array([self.conveyor_speed])
            }
        else:
            observation = np.array([self.beam_energy_level, self.dose_accumulated, self.leakage_level, self.conveyor_speed])

        return observation, reward, terminate, truncate, {}

    def render(self, mode='console'):
        """Render the environment to the screen."""
        if mode == 'console':
            print(f"Step: {self.step_count}")
            print(f"Beam Energy Level: {self.beam_energy_level:.2f}, Dose Accumulated: {self.dose_accumulated:.2f}")
            print(f"Leakage Level: {self.leakage_level:.2f}, Conveyor Speed: {self.conveyor_speed:.2f}")

# # Example of how to use this environment
# if __name__ == "__main__":
#     env = IndustryParticleAcceleratorEnv()
#
#     obs = env.reset()
#     for _ in range(10):
#         action = env.action_space.sample()  # Random action, replace with RL agent decision
#         obs, reward, done, info = env.step(action)
#         env.render()
#         if done:
#             break
#
#     env.close()
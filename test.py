import gymnasium as gym
from gym_sokoban.envs.sokoban_env import SokobanEnv
import stable_baselines3 as sb3

def main():
    env = SokobanEnv()
    model = sb3.PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

if __name__ == "__main__":
    main()  
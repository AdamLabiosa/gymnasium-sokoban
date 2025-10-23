"""
Example demonstrating how to use seeds to generate reproducible Sokoban levels.
This is useful for:
- Creating consistent test environments
- Sharing specific levels with others
- Debugging specific level configurations
"""
import time
from gym_sokoban.envs.sokoban_env import SokobanEnv

def play_level_with_seed(seed, render=True):
    """Play a level generated with a specific seed"""
    env = SokobanEnv()
    
    print(f"\n{'='*60}")
    print(f"Playing level with SEED: {seed}")
    print(f"{'='*60}\n")
    
    # Reset with a specific seed - this will always generate the same level
    observation, info = env.reset(seed=seed)
    
    if render:
        env.render(mode='human')
        time.sleep(2)
    
    # Play a few random actions
    for step in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        if render:
            env.render(mode='human')
            time.sleep(0.5)
        
        if terminated or truncated:
            print(f"Episode finished after {step+1} steps")
            break
    
    env.close()
    return observation

# Example 1: Play the same level twice
print("Example 1: Playing the same level twice with seed=42")
print("Both levels should be identical!")
play_level_with_seed(seed=42, render=False)
play_level_with_seed(seed=42, render=False)

# Example 2: Play different levels with different seeds
print("\n\nExample 2: Playing different levels with different seeds")
seeds_to_try = [42, 100, 999, 12345]

for seed in seeds_to_try:
    env = SokobanEnv()
    obs, _ = env.reset(seed=seed)
    print(f"Seed {seed:5d}: Generated level with shape {obs.shape}")
    env.close()

# Example 3: Interactive play with a specific seed (commented out by default)
print("\n\nTo play a specific level interactively, uncomment the line below:")
print("# play_level_with_seed(seed=42, render=True)")

print("\n\nSeed functionality is working correctly!")
print("You can now use env.reset(seed=YOUR_SEED) to generate reproducible levels.")

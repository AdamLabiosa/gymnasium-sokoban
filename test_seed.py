"""
Test script to verify that seeding works correctly in Sokoban environment.
Two resets with the same seed should produce identical levels.
"""
import numpy as np
from gym_sokoban.envs.sokoban_env import SokobanEnv

# Create environment
env = SokobanEnv()

# Test with seed=42
print("Testing with seed=42...")
obs1, _ = env.reset(seed=42)
room_state1 = env.room_state.copy()
room_fixed1 = env.room_fixed.copy()

obs2, _ = env.reset(seed=42)
room_state2 = env.room_state.copy()
room_fixed2 = env.room_fixed.copy()

# Check if they are identical
if np.array_equal(room_state1, room_state2) and np.array_equal(room_fixed1, room_fixed2):
    print("✓ SUCCESS: Same seed produces identical levels!")
    print(f"  Room state match: {np.array_equal(room_state1, room_state2)}")
    print(f"  Room fixed match: {np.array_equal(room_fixed1, room_fixed2)}")
    print(f"  Observation match: {np.array_equal(obs1, obs2)}")
else:
    print("✗ FAILED: Same seed produces different levels!")
    print(f"  Room state match: {np.array_equal(room_state1, room_state2)}")
    print(f"  Room fixed match: {np.array_equal(room_fixed1, room_fixed2)}")

# Test with different seeds
print("\nTesting with different seeds (42 vs 123)...")
obs3, _ = env.reset(seed=42)
room_state3 = env.room_state.copy()

obs4, _ = env.reset(seed=123)
room_state4 = env.room_state.copy()

if not np.array_equal(room_state3, room_state4):
    print("✓ SUCCESS: Different seeds produce different levels!")
else:
    print("✗ WARNING: Different seeds produced identical levels (unlikely but possible)")

# Test multiple resets with same seed
print("\nTesting 3 consecutive resets with seed=42...")
observations = []
for i in range(3):
    obs, _ = env.reset(seed=42)
    observations.append(obs)

all_equal = all(np.array_equal(observations[0], obs) for obs in observations[1:])
if all_equal:
    print("✓ SUCCESS: All 3 resets with seed=42 produced identical levels!")
else:
    print("✗ FAILED: Resets with same seed produced different levels!")

env.close()
print("\nTest completed!")

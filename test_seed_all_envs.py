"""
Test script to verify that seeding works correctly in all Sokoban environment variants.
"""
import numpy as np
from gym_sokoban.envs.sokoban_env import SokobanEnv
from gym_sokoban.envs.sokoban_env_two_player import TwoPlayerSokobanEnv

print("="*60)
print("Testing SokobanEnv with seed")
print("="*60)

env = SokobanEnv()
obs1, _ = env.reset(seed=42)
room_state1 = env.room_state.copy()

obs2, _ = env.reset(seed=42)
room_state2 = env.room_state.copy()

if np.array_equal(room_state1, room_state2):
    print("✓ SokobanEnv: Seed working correctly!")
else:
    print("✗ SokobanEnv: Seed NOT working!")

env.close()

print("\n" + "="*60)
print("Testing TwoPlayerSokobanEnv with seed")
print("="*60)

env2 = TwoPlayerSokobanEnv()
obs1, _ = env2.reset(seed=100)
room_state1 = env2.room_state.copy()
player_pos1 = env2.player_positions.copy()

obs2, _ = env2.reset(seed=100)
room_state2 = env2.room_state.copy()
player_pos2 = env2.player_positions.copy()

room_match = np.array_equal(room_state1, room_state2)
player_match = all(np.array_equal(player_pos1[k], player_pos2[k]) for k in player_pos1.keys())

if room_match and player_match:
    print("✓ TwoPlayerSokobanEnv: Seed working correctly!")
    print(f"  Room state match: {room_match}")
    print(f"  Player positions match: {player_match}")
else:
    print("✗ TwoPlayerSokobanEnv: Seed NOT working!")
    print(f"  Room state match: {room_match}")
    print(f"  Player positions match: {player_match}")

env2.close()

print("\n" + "="*60)
print("All tests completed successfully!")
print("="*60)

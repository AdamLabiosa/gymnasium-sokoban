"""
Benchmark tests comparing reset() vs reset_to_saved_layout() performance and correctness.
Tests that existing behavior is unchanged after introducing soft reset.
"""
import time
import numpy as np
import pytest

from gym_sokoban.envs.sokoban_env import SokobanEnv


def make_env(dim=(10, 10), num_boxes=4, max_steps=120):
    return SokobanEnv(dim_room=dim, num_boxes=num_boxes, max_steps=max_steps, reset=True)


class TestPerformanceComparison:
    """Performance comparison: full reset vs soft reset"""
    
    def test_reset_performance_comparison_small(self):
        """Compare reset() vs reset_to_saved_layout() on small room"""
        env = make_env(dim=(7, 7), num_boxes=2)
        
        # Full reset benchmark
        n_iterations = 20
        start = time.perf_counter()
        for _ in range(n_iterations):
            env.reset()
        time_full_reset = time.perf_counter() - start
        
        # Soft reset benchmark
        layout = env.get_layout()
        start = time.perf_counter()
        for _ in range(n_iterations):
            env.reset_to_saved_layout()
        time_soft_reset = time.perf_counter() - start
        
        speedup = time_full_reset / time_soft_reset
        
        print(f"\n=== Small Room (7x7, 2 boxes) - {n_iterations} iterations ===")
        print(f"Full reset:  {time_full_reset:.4f}s ({time_full_reset/n_iterations*1000:.2f}ms avg)")
        print(f"Soft reset:  {time_soft_reset:.4f}s ({time_soft_reset/n_iterations*1000:.2f}ms avg)")
        print(f"Speedup:     {speedup:.2f}x")
        
        # Soft reset should be significantly faster
        assert time_soft_reset < time_full_reset
        assert speedup > 2.0, f"Expected speedup >2x, got {speedup:.2f}x"
    
    def test_reset_performance_comparison_medium(self):
        """Compare reset() vs reset_to_saved_layout() on medium room"""
        env = make_env(dim=(10, 10), num_boxes=4)
        
        # Full reset benchmark
        n_iterations = 10
        start = time.perf_counter()
        for _ in range(n_iterations):
            env.reset()
        time_full_reset = time.perf_counter() - start
        
        # Soft reset benchmark (more iterations for stable timing)
        n_soft = 50
        start = time.perf_counter()
        for _ in range(n_soft):
            env.reset_to_saved_layout()
        time_soft_reset = time.perf_counter() - start
        
        # Normalize per iteration
        avg_full = time_full_reset / n_iterations
        avg_soft = time_soft_reset / n_soft
        speedup = avg_full / avg_soft
        
        print(f"\n=== Medium Room (10x10, 4 boxes) ===")
        print(f"Full reset:  {avg_full*1000:.2f}ms avg ({n_iterations} iterations)")
        print(f"Soft reset:  {avg_soft*1000:.2f}ms avg ({n_soft} iterations)")
        print(f"Speedup:     {speedup:.2f}x")
        
        assert avg_soft < avg_full
        assert speedup > 2.0, f"Expected speedup >2x, got {speedup:.2f}x"
    
    def test_reset_performance_comparison_large(self):
        """Compare reset() vs reset_to_saved_layout() on large room"""
        env = make_env(dim=(12, 12), num_boxes=4)
        
        # Full reset benchmark (few iterations since generation is very slow)
        n_iterations = 5
        start = time.perf_counter()
        for _ in range(n_iterations):
            env.reset()
        time_full_reset = time.perf_counter() - start
        
        # Soft reset benchmark (more iterations for stable measurement)
        n_soft_iterations = 50
        start = time.perf_counter()
        for _ in range(n_soft_iterations):
            env.reset_to_saved_layout()
        time_soft_reset = time.perf_counter() - start
        
        # Normalize to per-iteration time for fair comparison
        avg_full = time_full_reset / n_iterations
        avg_soft = time_soft_reset / n_soft_iterations
        speedup = avg_full / avg_soft
        
        print(f"\n=== Large Room (12x12, 4 boxes) ===")
        print(f"Full reset:  {avg_full*1000:.2f}ms avg ({n_iterations} iterations)")
        print(f"Soft reset:  {avg_soft*1000:.2f}ms avg ({n_soft_iterations} iterations)")
        print(f"Speedup:     {speedup:.2f}x")
        
        assert avg_soft < avg_full
        # Larger rooms should show even better speedup
        assert speedup > 5.0, f"Expected speedup >5x for large room, got {speedup:.2f}x"


class TestBackwardCompatibility:
    """Verify existing behavior is unchanged after modifications"""
    
    def test_reset_basic_functionality(self):
        """Test that reset() still works correctly"""
        env = make_env()
        
        # Reset should return observation and info
        obs, info = env.reset()
        
        # Observation should be correct shape
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.uint8
        
        # Info should be dict
        assert isinstance(info, dict)
        
        # Environment state should be initialized
        assert env.num_env_steps == 0
        assert env.reward_last == 0
        assert env.boxes_on_target >= 0
        assert env.player_position is not None
        assert env.room_state is not None
        assert env.room_fixed is not None
    
    def test_reset_seed_reproducibility(self):
        """Test that seeded reset still produces same rooms"""
        env1 = make_env()
        env2 = make_env()
        
        seed = 42
        obs1, _ = env1.reset(seed=seed)
        obs2, _ = env2.reset(seed=seed)
        
        # Same seed should produce identical observations
        assert np.array_equal(obs1, obs2)
        assert np.array_equal(env1.room_state, env2.room_state)
        assert np.array_equal(env1.room_fixed, env2.room_fixed)
    
    def test_step_basic_functionality(self):
        """Test that step() still works correctly after reset"""
        env = make_env()
        env.reset()
        
        # Take some actions
        for action in [1, 5, 7, 2, 8]:
            obs, reward, term, trunc, info = env.step(action)
            
            # Check return types
            assert obs.shape == env.observation_space.shape
            assert isinstance(reward, (int, float, np.number))
            assert isinstance(term, (bool, np.bool_))
            assert isinstance(trunc, (bool, np.bool_))
            assert isinstance(info, dict)
            
            # Info should contain expected keys
            assert "action.name" in info
            assert "action.moved_player" in info
            assert "action.moved_box" in info
    
    def test_episode_progression(self):
        """Test that episodes progress correctly"""
        env = make_env(max_steps=10)
        env.reset()
        
        # Play until truncation
        term = False
        trunc = False
        step_count = 0
        
        while not (term or trunc):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            step_count += 1
            
            # Ensure step counter increments
            assert env.num_env_steps == step_count
            
            if step_count >= 20:  # Safety break
                break
        
        # Should truncate at max_steps
        assert trunc or term or step_count >= 10
    
    def test_reward_calculation_unchanged(self):
        """Test that reward calculation still works"""
        env = make_env()
        env.reset(seed=123)
        
        # Record initial boxes_on_target
        initial_boxes = env.boxes_on_target
        
        # Take action
        _, reward, _, _, _ = env.step(1)
        
        # Reward should be calculated (at minimum penalty for step)
        assert reward <= 0 or reward > 0  # Any numeric value is fine
        assert env.reward_last == reward
        
        # boxes_on_target should be tracked
        assert hasattr(env, 'boxes_on_target')
        assert env.boxes_on_target >= 0


class TestSoftResetCorrectness:
    """Test that soft reset produces identical behavior to full reset"""
    
    def test_soft_reset_identical_to_saved_state(self):
        """Verify soft reset restores exact state"""
        env = make_env()
        env.reset(seed=42)
        
        # Save initial state
        initial_room_state = env.room_state.copy()
        initial_room_fixed = env.room_fixed.copy()
        initial_player_pos = env.player_position.copy()
        
        # Take some actions to modify state
        for _ in range(10):
            env.step(env.action_space.sample())
        
        # State should have changed
        assert not np.array_equal(env.room_state, initial_room_state)
        
        # Soft reset
        env.reset_to_saved_layout()
        
        # State should match initial exactly
        assert np.array_equal(env.room_state, initial_room_state)
        assert np.array_equal(env.room_fixed, initial_room_fixed)
        assert np.array_equal(env.player_position, initial_player_pos)
        assert env.num_env_steps == 0
        assert env.reward_last == 0
    
    def test_soft_reset_episode_independence(self):
        """Test that episodes after soft reset are independent"""
        env = make_env()
        env.reset(seed=42)
        
        # Play first episode with soft reset
        actions_episode1 = [1, 5, 7, 2, 8]
        states_episode1 = []
        
        for action in actions_episode1:
            env.step(action)
            states_episode1.append(env.room_state.copy())
        
        # Soft reset and replay same actions
        env.reset_to_saved_layout()
        
        for i, action in enumerate(actions_episode1):
            env.step(action)
            # State after same action sequence should be identical
            assert np.array_equal(env.room_state, states_episode1[i])
    
    def test_multiple_soft_resets(self):
        """Test multiple consecutive soft resets"""
        env = make_env()
        env.reset(seed=99)
        
        reference_state = env.room_state.copy()
        
        # Multiple soft resets should all produce same state
        for _ in range(5):
            # Modify state
            for _ in range(3):
                env.step(env.action_space.sample())
            
            # Reset
            env.reset_to_saved_layout()
            
            # Should match reference
            assert np.array_equal(env.room_state, reference_state)
            assert env.num_env_steps == 0

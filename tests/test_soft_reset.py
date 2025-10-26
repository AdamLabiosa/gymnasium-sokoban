import time
import numpy as np
import pytest

from gym_sokoban.envs.sokoban_env import SokobanEnv


def make_env(dim=(7, 7), num_boxes=2, max_steps=50):
    # Keep rooms small for speed
    return SokobanEnv(dim_room=dim, num_boxes=num_boxes, max_steps=max_steps, reset=True)


def test_apply_layout_restores_room_state():
    env = make_env()
    # Save initial layout
    layout = env.get_layout()

    # Take a few actions to change the state
    for a in [1, 4, 7, 2, 5]:
        _obs, _r, _t, _tr, _info = env.step(a)

    # Ensure state changed
    assert not np.array_equal(env.room_state, layout['room_state'])

    # Apply layout to reset without regeneration
    obs, info = env.reset_to_layout(layout)

    # room_state should match exactly the saved layout
    assert np.array_equal(env.room_state, layout['room_state'])
    # room_fixed should match as well
    assert np.array_equal(env.room_fixed, layout['room_fixed'])
    # A render('raw') roundtrip returns consistent signals
    walls, goals, boxes, player = env.render('raw')
    # Player must exist
    assert player.sum() == 1
    # Boxes count should equal num_boxes
    assert boxes.sum() == env.num_boxes
    # Info indicates layout reset
    assert info.get('reset_type') == 'layout'


def test_reset_to_saved_layout_roundtrip():
    env = make_env()
    # Save current layout internally
    env.save_layout()

    # Mutate state
    for a in [1, 3, 6, 8]:
        env.step(a)

    # Reset using the saved layout
    _obs, info = env.reset_to_saved_layout()
    assert info.get('reset_type') == 'layout'

    # Compare against freshly captured layout for equality
    saved = env.get_layout()
    assert np.array_equal(env.room_state, saved['room_state'])


def test_soft_reset_is_faster_than_regeneration():
    env = make_env(dim=(9, 9), num_boxes=3)
    layout = env.get_layout()

    # Warm-up a bit
    for _ in range(3):
        env.reset_to_layout(layout)
        env.reset()

    # Measure several runs to reduce noise
    n = 10
    t_soft = 0.0
    t_reset = 0.0

    for _ in range(n):
        t0 = time.perf_counter()
        env.reset_to_layout(layout)
        t_soft += time.perf_counter() - t0

    for _ in range(n):
        t0 = time.perf_counter()
        env.reset()
        t_reset += time.perf_counter() - t0

    # Soft reset should be faster than generating a new level on average
    assert t_soft < t_reset, f"Soft reset slower than full reset: {t_soft:.6f} vs {t_reset:.6f}"

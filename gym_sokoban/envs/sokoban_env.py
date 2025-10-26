import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces import Box
from .room_utils import generate_room
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
import numpy as np


class SokobanEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw'],
        'render_modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=4,
                 num_gen_steps=None,
                 reset=True):

        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_last = 0

        # Other Settings
        self.viewer = None
        self.viewer_ax = None
        self.viewer_img = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        
        if reset:
            # Initialize Room
            _ = self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, observation_mode='rgb_array'):
        assert action in ACTION_LOOKUP
        assert observation_mode in ['rgb_array', 'tiny_rgb_array', 'raw']

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        if action == 0:
            moved_player = False

        # All push actions are in the range of [0, 3]
        elif action < 5:
            moved_player, moved_box = self._push(action)

        else:
            moved_player = self._move(action)

        self._calc_reward()
        
        term = self._check_if_all_boxes_on_target()
        trunc = self._check_if_maxsteps()

        # Convert the observation to RGB frame
        observation = self.render(mode=observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if term or trunc:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return observation, self.reward_last, term, trunc, info

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False


        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True

        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target
        
        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            self.reward_last += self.reward_finished
        
        self.boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.        
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.max_steps == self.num_env_steps)

    def reset(self, seed=None, second_player=False, render_mode='rgb_array', **kwargs):
        # Set the seed if provided
        if seed is not None:
            self.seed(seed)
        
        try:
            self.room_fixed, self.room_state, self.box_mapping = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,
                num_boxes=self.num_boxes,
                second_player=second_player,
                seed=seed
            )
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(seed=seed, second_player=second_player, render_mode=render_mode)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        # Store a snapshot of the freshly generated layout to allow fast resets without regeneration
        self._saved_layout = {
            'room_fixed': self.room_fixed.copy(),
            'room_state': self.room_state.copy(),
            'box_mapping': self.box_mapping.copy() if isinstance(self.box_mapping, dict) else self.box_mapping
        }

        starting_observation = self.render(render_mode)

        info = {}

        return starting_observation, info

    def get_layout(self, include_mapping=True):
        """
        Return a copy of the current layout to allow resetting the room state later without regeneration.
        The returned object can be stored externally by the user.

        Layout schema (dict):
        - 'room_fixed': np.ndarray, immutable room structure (walls/targets/floors)
        - 'room_state': np.ndarray, full current state including player/boxes
        - 'box_mapping': dict (optional), mapping of box targets to current locations
        """
        layout = {
            'room_fixed': self.room_fixed.copy(),
            'room_state': self.room_state.copy()
        }
        if include_mapping and hasattr(self, 'box_mapping'):
            layout['box_mapping'] = self.box_mapping.copy() if isinstance(self.box_mapping, dict) else self.box_mapping
        return layout

    def save_layout(self, layout=None):
        """
        Save a layout snapshot inside the environment for quick reuse.
        If layout is None, saves the current layout.
        """
        if layout is None:
            layout = self.get_layout()
        # Shallow copy keys, ensure arrays are copied
        self._saved_layout = {
            'room_fixed': layout['room_fixed'].copy(),
            'room_state': layout['room_state'].copy(),
            'box_mapping': layout.get('box_mapping', None)
        }

    def reset_to_layout(self, layout, render_mode='rgb_array'):
        """
        Reset the environment to a provided layout without regenerating a new level.
        This applies the given room layout and reinitializes episode variables.

        Args:
            layout (dict): as returned by get_layout(); must contain 'room_state' and 'room_fixed'.
            render_mode (str): observation render mode, defaults to 'rgb_array'.

        Returns:
            observation, info (dict)
        """
        if layout is None:
            raise ValueError("layout must be provided to reset_to_layout")

        if 'room_state' not in layout or 'room_fixed' not in layout:
            raise KeyError("layout must contain 'room_state' and 'room_fixed'")

        # Apply copies to avoid aliasing external buffers
        self.room_fixed = layout['room_fixed'].copy()
        self.room_state = layout['room_state'].copy()

        # Optional: restore box_mapping if available
        if 'box_mapping' in layout and layout['box_mapping'] is not None:
            self.box_mapping = layout['box_mapping'].copy() if isinstance(layout['box_mapping'], dict) else layout['box_mapping']

        # Recompute derived episode state
        player_pos = np.argwhere(self.room_state == 5)
        if player_pos.size == 0:
            raise ValueError("Provided layout has no player (value 5) in room_state")
        self.player_position = player_pos[0]
        self.num_env_steps = 0
        self.reward_last = 0

        # Recalculate current boxes_on_target from state
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target
        self.boxes_on_target = self.num_boxes - np.where(total_targets)[0].shape[0]

        # Return initial observation
        starting_observation = self.render(render_mode)
        info = {"reset_type": "layout"}
        return starting_observation, info

    def reset_to_saved_layout(self, render_mode='rgb_array'):
        """
        Convenience wrapper to reset to the last saved layout via save_layout() or initial reset().
        """
        if not hasattr(self, '_saved_layout') or self._saved_layout is None:
            raise RuntimeError("No saved layout available. Call save_layout() or reset() first.")
        return self.reset_to_layout(self._saved_layout, render_mode=render_mode)

    def render(self, mode='human', close=None, scale=1):
        assert mode in RENDERING_MODES

        img = self.get_image(mode, scale)

        if 'rgb_array' in mode:
            return img

        elif 'human' in mode:
            # Use matplotlib for displaying images in gymnasium
            import matplotlib.pyplot as plt
            
            if self.viewer is None:
                # Try to use an interactive backend
                import matplotlib
                current_backend = matplotlib.get_backend()
                if current_backend == 'agg':
                    # Try different backends in order of preference
                    for backend in ['TkAgg', 'Qt5Agg', 'MacOSX', 'GTK3Agg']:
                        try:
                            matplotlib.use(backend)
                            break
                        except:
                            continue
                
                plt.ion()  # Turn on interactive mode
                self.viewer = plt.figure(figsize=(8, 8))
                self.viewer_ax = self.viewer.add_subplot(111)
                self.viewer_ax.axis('off')
                self.viewer_img = None
            
            if self.viewer_img is None:
                self.viewer_img = self.viewer_ax.imshow(img)
            else:
                self.viewer_img.set_data(img)
            
            self.viewer.canvas.draw()
            self.viewer.canvas.flush_events()
            
            return True

        elif 'raw' in mode:
            arr_walls = (self.room_fixed == 0).view(np.int8)
            arr_goals = (self.room_fixed == 2).view(np.int8)
            arr_boxes = ((self.room_state == 4) + (self.room_state == 3)).view(np.int8)
            arr_player = (self.room_state == 5).view(np.int8)

            return arr_walls, arr_goals, arr_boxes, arr_player

        else:
            super(SokobanEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode, scale=1):
        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)
        else:
            img = room_to_rgb(self.room_state, self.room_fixed)

        return img

    def close(self):
        if self.viewer is not None:
            import matplotlib.pyplot as plt
            plt.close(self.viewer)
            self.viewer = None
            self.viewer_ax = None
            self.viewer_img = None

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP


ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw']

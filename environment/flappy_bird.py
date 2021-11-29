#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Implementation of a Flappy Bird OpenAI Gym environment that yields simple
numerical information about the game's state as observations.
"""
# from flappy_bird_gym.envs.game_logic import FlappyBirdLogic
from enum import IntEnum
from itertools import cycle
from typing import Dict, Tuple, Optional, Union

import gym
import numpy as np
import pygame
from flappy_bird_gym.envs.game_logic import PLAYER_FLAP_ACC, PLAYER_VEL_ROT, PLAYER_MAX_VEL_Y, PLAYER_ACC_Y
from flappy_bird_gym.envs.game_logic import PLAYER_WIDTH, PLAYER_HEIGHT, BASE_WIDTH, BACKGROUND_WIDTH
from flappy_bird_gym.envs.renderer import FlappyBirdRenderer


class FlappyBirdLogic:
    """ Handles the logic of the Flappy Bird game.

    The implementation of this class is decoupled from the implementation of the
    game's graphics. This class implements the logical portion of the game.

    Args:
        screen_size (Tuple[int, int]): Tuple with the screen's width and height.
        pipe_gap_size (int): Space between a lower and an upper pipe.

    Attributes:
        player_x (int): The player's x position.
        player_y (int): The player's y position.
        base_x (int): The base/ground's x position.
        base_y (int): The base/ground's y position.
        score (int): Current score of the player.
        upper_pipes (List[Dict[str, int]): List with the upper pipes. Each pipe
            is represented by a dictionary containing two keys: "x" (the pipe's
            x position) and "y" (the pipe's y position).
        lower_pipes (List[Dict[str, int]): List with the lower pipes. Each pipe
            is represented by a dictionary containing two keys: "x" (the pipe's
            x position) and "y" (the pipe's y position).
        player_vel_y (int): The player's vertical velocity.
        player_rot (int): The player's rotation angle.
        last_action (Optional[FlappyBirdLogic.Actions]): The last action taken
            by the player. If `None`, the player hasn't taken any action yet.
        sound_cache (Optional[str]): Stores the name of the next sound to be
            played. If `None`, then no sound should be played.
        player_idx (int): Current index of the bird's animation cycle.
    """

    def __init__(self,
                 screen_size: Tuple[int, int],
                 pipe_gap_size: int = 100) -> None:
        self._screen_width = screen_size[0]
        self._screen_height = screen_size[1]

        self.player_x = int(self._screen_width * 0.2)
        self.player_y = int((self._screen_height - PLAYER_HEIGHT) / 2)
        self.upper_pipes = []
        self.lower_pipes = []
        self.base_x = 0
        self.base_y = self._screen_height * 0.79
        self._base_shift = BASE_WIDTH - BACKGROUND_WIDTH

        self.score = 0

        # Player's info:
        self.player_vel_y = -9  # player"s velocity along Y
        self.player_rot = 45  # player"s rotation

        self.last_action = None
        self.sound_cache = None

        self._player_flapped = False
        self.player_idx = 0
        self._player_idx_gen = cycle([0, 1, 2, 1])
        self._loop_iter = 0

    class Actions(IntEnum):
        """ Possible actions for the player to take. """
        IDLE, FLAP = 0, 1

    def check_crash(self) -> bool:
        """ Returns True if player collides with the ground (base) or a pipe.
        """
        # if player crashes into ground
        if self.player_y + PLAYER_HEIGHT >= self.base_y - 1:
            return True
        return False

    def update_state(self, action: Union[Actions, int]) -> bool:
        """ Given an action taken by the player, updates the game's state.

        Args:
            action (Union[FlappyBirdLogic.Actions, int]): The action taken by
                the player.

        Returns:
            `True` if the player is alive and `False` otherwise.
        """
        self.sound_cache = None
        if action == FlappyBirdLogic.Actions.FLAP:
            if self.player_y > -2 * PLAYER_HEIGHT:
                self.player_vel_y = PLAYER_FLAP_ACC
                self._player_flapped = True
                self.sound_cache = "wing"

        self.last_action = action
        if self.check_crash():
            self.sound_cache = "hit"
            return False

        # check for score
        player_mid_pos = self.player_x + PLAYER_WIDTH / 2
        self.score += 1

        self.base_x = -((-self.base_x + 100) % self._base_shift)

        # rotate the player
        if self.player_rot > -90:
            self.player_rot -= PLAYER_VEL_ROT

        # player's movement
        if self.player_vel_y < PLAYER_MAX_VEL_Y and not self._player_flapped:
            self.player_vel_y += PLAYER_ACC_Y

        if self._player_flapped:
            self._player_flapped = False

            # more rotation to cover the threshold
            # (calculated in visible rotation)
            self.player_rot = 45

        self.player_y += min(self.player_vel_y,
                             self.base_y - self.player_y - PLAYER_HEIGHT)

        return True


class FlappyBirdEnvSimple_nopipe(gym.Env):
    """ Flappy Bird Gym environment that yields simple observations.

    The observations yielded by this environment are simple numerical
    information about the game's state. Specifically, the observations are:

        * Horizontal distance to the next pipe;
        * Difference between the player's y position and the next hole's y
          position.

    The reward received by the agent in each step is equal to the score obtained
    by the agent in that step. A score point is obtained every time the bird
    passes a pipe.

    Args:
        screen_size (Tuple[int, int]): The screen's width and height.
        normalize_obs (bool): If `True`, the observations will be normalized
            before being returned.
        pipe_gap (int): Space between a lower and an upper pipe.
        bird_color (str): Color of the flappy bird. The currently available
            colors are "yellow", "blue" and "red".
        pipe_color (str): Color of the pipes. The currently available colors are
            "green" and "red".
        background (Optional[str]): Type of background image. The currently
            available types are "day" and "night". If `None`, no background will
            be drawn.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 screen_size: Tuple[int, int] = (288, 512),
                 normalize_obs: bool = True,
                 pipe_gap: int = 100,
                 bird_color: str = "yellow",
                 pipe_color: str = "green",
                 background: Optional[str] = "day") -> None:
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(2,),
                                                dtype=np.float32)
        self._screen_size = screen_size
        self._normalize_obs = normalize_obs
        self._pipe_gap = pipe_gap

        self._game = None
        self._renderer = None

        self._bird_color = bird_color
        self._pipe_color = pipe_color
        self._bg_type = background

    def _get_observation(self):
        # up_pipe = low_pipe = None
        # h_dist = 0
        # for up_pipe, low_pipe in zip(self._game.upper_pipes,
        #                              self._game.lower_pipes):
        #     h_dist = (low_pipe["x"] + PIPE_WIDTH / 2
        #               - (self._game.player_x - PLAYER_WIDTH / 2))
        #     h_dist += 3  # extra distance to compensate for the buggy hit-box
        #     if h_dist >= 0:
        #         break
        #
        # upper_pipe_y = up_pipe["y"] + PIPE_HEIGHT
        # lower_pipe_y = low_pipe["y"]
        player_y = self._game.player_y
        #
        # v_dist = (upper_pipe_y + lower_pipe_y) / 2 - (player_y
        #                                               + PLAYER_HEIGHT / 2)
        #
        # if self._normalize_obs:
        #     h_dist /= self._screen_size[0]
        #     v_dist /= self._screen_size[1]

        return np.array([
            player_y
        ])

    def step(self,
             action: Union[FlappyBirdLogic.Actions, int],
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        """ Given an action, updates the game state.

        Args:
            action (Union[FlappyBirdLogic.Actions, int]): The action taken by
                the agent. Zero (0) means "do nothing" and one (1) means "flap".

        Returns:
            A tuple containing, respectively:

                * an observation (horizontal distance to the next pipe;
                  difference between the player's y position and the next hole's
                  y position);
                * a reward (always 1);
                * a status report (`True` if the game is over and `False`
                  otherwise);
                * an info dictionary.
        """
        alive = self._game.update_state(action)
        obs = self._get_observation()

        reward = 1

        done = not alive
        info = {"score": self._game.score}

        return obs, reward, done, info

    def reset(self):
        """ Resets the environment (starts a new game). """
        self._game = FlappyBirdLogic(screen_size=self._screen_size,
                                     pipe_gap_size=self._pipe_gap)
        if self._renderer is not None:
            self._renderer.game = self._game

        return self._get_observation()

    def render(self, mode='human') -> None:
        """ Renders the next frame. """
        if self._renderer is None:
            self._renderer = FlappyBirdRenderer(screen_size=self._screen_size,
                                                bird_color=self._bird_color,
                                                pipe_color=self._pipe_color,
                                                background=self._bg_type)
            self._renderer.game = self._game
            self._renderer.make_display()

        self._renderer.draw_surface(show_score=True)
        self._renderer.update_display()

    def close(self):
        """ Closes the environment. """
        if self._renderer is not None:
            pygame.display.quit()
            self._renderer = None
        super().close()


if __name__ == '__main__':
    import time

    env = FlappyBirdEnvSimple_nopipe()
    env.reset()
    env.render()
    done = False
    while not done:
        obs, reward, done, _ = env.step(env.action_space.sample())
        env.render()
        time.sleep(1 / 30)

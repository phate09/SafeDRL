import random
import sys
from typing import Tuple

import mpmath
import sympy
from mpmath import iv, pi
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import intervals as I


class BouncingBall(gym.Env):
    def __init__(self):
        self.v = 0  # velocity
        self.c = 0  # cost/hit counter
        self.p = 0  # position
        random.seed = 0

    def reset(self):
        self.p = 7 + random.uniform(0, 3)
        self.v = 0
        return (self.p, self.v)

    def step(self, action):
        done = False
        if self.p >= 4 and action == 1:  # within reach and hit
            if self.v >= 0:  # while going up within reach
                self.v = -(0.9 + random.uniform(0, 0.1)) * self.v - 4
            elif -4 <= self.v < 0:  # while going down within reach
                self.v = -4
        elif self.p == 0 and self.v < 0:  # bounce
            self.v = -(0.85 + random.uniform(0, 0.12)) * self.v
            if self.v <= 1:
                done = True
        else:
            self.v = self.v - 9.81  # speed decay
        self.p = max(self.p + 1 * self.v, 0)  # position update
        cost = 1 if action == 1 else 0
        return (self.p, self.v), cost, done, {}


if __name__ == '__main__':
    env = BouncingBall()
    env.reset()
    env.step(1)

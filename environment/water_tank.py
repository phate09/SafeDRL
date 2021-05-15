import time

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

current_folder = path.abspath(path.dirname(__file__))


class WaterTankEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, conf=None):
        self.dt = .05
        self.l = 50.
        self.viewer = None
        self.discrete_actions = True
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([100, 100, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # action controls the inflow valve
        l, timer, outflow_valve, inflow_valve = self.state  # th := theta
        if action == 0:
            # continue heating what you have
            newl = l,
            newtimer = 0 if inflow_valve != action else min(100, timer + 1)
            newlast_action = action
            pass
        else:  # action ==1
            newl = l + self.np_random.uniform(1, 2),
            newtimer = 0 if inflow_valve != action else min(100, timer + 1)
            newlast_action = action
        self.state = np.array([newl, newtimer, newlast_action])
        costs = -1
        return self.state, costs, False, {}

    def reset(self):
        self.state = self.np_random.uniform(low=np.array([0, 0], dtype=np.float32),
                                            high=np.array([100, 100], dtype=np.float32))
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class MonitoredWaterTank(WaterTankEnv):
    def __init__(self, conf=None):
        super().__init__(conf)
        self.max_steps = 20
        self.failure_step = 0
        self.max_angle = 90

    def step(self, action):
        if self.state[1] < 3 and action != self.state[2]:
            return self.state, -10, False  # just punish wrong behaviour
        obs, cost, done, _ = super().step(action)
        if self.state[0] > 100 or self.state[0] <= 0:
            done = True
            cost -= 1000
        return obs, cost, done, _


if __name__ == '__main__':
    env = MonitoredWaterTank()
    env.reset()

    done = False
    while not done:
        obs, reward, done, _ = env.step(env.action_space.sample())
        env.render()
        # time.sleep(1 / 30)

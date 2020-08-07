from typing import Tuple

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from mpmath import iv, pi
from interval import interval
import interval.imath as imath

from mosaic.hyperrectangle import HyperRectangle


class PendulumEnv_abstract(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.max_angle = np.pi / 4
        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def set_state(self, state: HyperRectangle):
        self.state = tuple([interval([x.left_bound(), x.right_bound()]) for x in state])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = min(max(u, -self.max_torque), self.max_torque)
        self.last_u = u  # for rendering
        costs = angle_normalize_tuple(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * imath.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = clip_interval(newthdot, -self.max_speed, self.max_speed)  # pylint: disable=E1111

        self.state = (newth, newthdot)
        done = newth[0][1] < -self.max_angle or newth[0][0] > self.max_angle
        half_done = done or newth[0][0] < -self.max_angle or newth[0][1] > self.max_angle
        return HyperRectangle.from_numpy(np.array(tuple([make_tuple(x) for x in self.state])).transpose()), -costs, done, half_done

    def is_terminal(self, interval, half=False):
        done = interval[0][1] < -self.max_angle or interval[0][0] > self.max_angle
        if half:
            return done or interval[0][0] < -self.max_angle or interval[0][1] > self.max_angle
        else:
            return done

    def reset(self):
        # high = np.array([np.pi, 1])
        # self.state = self.np_random.uniform(low=-high, high=high)
        self.state = (interval([-self.max_angle, self.max_angle]), interval([-1, 1]))
        self.last_u = None
        return HyperRectangle.from_numpy(np.array(tuple([make_tuple(x) for x in self.state])).transpose())

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class PendulumEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.max_angle = np.pi / 4
        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        # u = min(max(u, -self.max_torque), self.max_torque)
        u = self.max_torque * 1 if u == 1 else -self.max_torque * 1
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)  # pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        done = not -self.max_angle < newth < self.max_angle
        reward = 1 if not done else -100
        return self.state, reward, done, {}

    def reset(self):
        high = np.array([np.pi / 4, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        # self.state = tuple([interval([0, 0.005]) for x in range(2)])
        self.last_u = None
        return self.state

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize_tuple(x):
    tup = make_tuple(x)
    state = tuple(map(lambda j: (j + np.pi) % (2 * np.pi) - np.pi, tup))
    return interval(state)


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


def clip_interval(x: interval, min, max):
    tup = make_tuple(x)
    return interval(clip(tup, min, max))


def clip(interval: tuple, min_x, max_x):
    return tuple([min(max(x, min_x), max_x) for x in interval])


def make_tuple(state):
    return tuple(state[0])


if __name__ == '__main__':
    env = PendulumEnv()
    env.reset()
    env.step(1)

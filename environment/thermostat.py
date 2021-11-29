import logging

import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ThermostatEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        # physical properties of the system
        self.c = 1.  # temperature resistance
        self.inv_c = 1. / self.c
        self.k = 1.
        self.Text = 20.
        self.max_P = 100.

        # state of the system
        self.T0 = 20.
        self.P = 0.

        self.T_target = -10.

        self.dt = 0.005

        self.action_space = spaces.Box(low=0, high=self.max_P, shape=(1,), dtype=np.float32)

        self.viewer = None

        self.temperature_history = []
        self.power_history = []

    def step(self, Pset):
        self.P = np.clip(Pset, 0, self.max_P)
        self.T += self.inv_c * (-self.P * self.dt + (self.Text - self.T) * self.k * self.dt)

        costs = np.abs(self.T_target - self.T)

        self.temperature_history.append(self.T)
        self.power_history.append(self.P)

        self.current_step += 1

        return self._get_obs(), -costs, False, {}

    def reset(self):
        self.T = self.T0
        self.temperature_history = [self.T]
        self.power_history = [0.]
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        return self.T, self.P, self.T_target

    def render(self, mode='human'):

        Tmin_rod = -30.
        Tmax_rod = 100.

        if self.viewer is None:
            self.fig, self.ax_temperature = plt.subplots()
            self.ax_power = self.ax_temperature.twinx()
            plt.show(block=False)

            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            rod_target = rendering.make_capsule(1, .04)
            rod_target.set_color(.2, .7, .7)
            self.transform_rod_target = rendering.Transform()
            rod_target.add_attr(self.transform_rod_target)
            self.transform_rod_target.set_rotation((self.T_target - Tmin_rod) / (Tmax_rod - Tmin_rod) * np.pi * 2)

            self.viewer.add_geom(rod_target)

            rod = rendering.make_capsule(1, .04)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)

        self.viewer.draw_circle(1, filled=False)
        self.pole_transform.set_rotation((self.T - Tmin_rod) / (Tmax_rod - Tmin_rod) * np.pi * 2)

        if self.current_step % 100 == 0:
            self.ax_temperature.clear()
            self.ax_power.clear()
            self.ax_temperature.axhline(self.T_target, color='C0', ls='--')
            self.ax_temperature.axhline(self.Text, color='0.5', ls='--')
            self.ax_temperature.plot(self.temperature_history, 'C0')
            self.ax_power.plot(self.power_history, 'oC1')
            self.ax_temperature.set_xlim(0, 1000)
            self.ax_temperature.set_ylim(-20, 25)
            plt.pause(0.001)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        plt.close()
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = ThermostatEnv()
    env.reset()
    env.render()
    for i in range(1000):
        action = 10  # env.action_space.sample()
        env.step(action)
        env.render()
    env.close()

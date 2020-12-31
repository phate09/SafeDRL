import math
import random

import gym
from gym import spaces
import numpy as np


class StoppingCar(gym.Env):
    def __init__(self, config=None):
        self.x_lead = 0  # position lead vehicle
        self.x_ego = 0  # position ego vehicle
        self.v_lead = 0  # velocity lead vehicle
        self.v_ego = 0  # velocity ego vehicle
        self.y_lead = 0  # acceleration lead vehicle
        self.y_ego = 0  # acceleration ego vehicle
        self.a_ego = 3  # deceleration/acceleration amount
        self.dt = .1  # delta time
        self.d_default = 20  # minimum safe distance
        # self.t_gap = 1.4  # safe distance reaction time
        self.v_set = 30  # speed to drive at if no cars ahead
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        random.seed = 0

    def reset(self):
        self.y_lead = self.y_ego = 0
        self.v_lead = random.uniform(20, 36)
        self.v_ego = random.uniform(20, 36)
        self.x_ego = random.uniform(0, 0)
        self.x_lead = random.uniform(20, 60)
        delta_x = self.x_lead - self.x_ego
        delta_v = self.v_lead - self.v_ego
        return np.array([self.x_lead, self.x_ego, self.v_lead, self.v_ego, self.y_lead, self.y_ego, delta_v, delta_x])

    def step(self, action_ego):
        if action_ego == 0:
            acceleration = -self.a_ego
        else:
            acceleration = self.a_ego
        self.y_ego = acceleration  # -2 * self.y_ego * self.dt + 2 * acceleration
        self.v_ego += self.y_ego * self.dt
        self.v_ego = min(max(self.v_ego, 0),40)
        self.v_lead += self.y_lead * self.dt
        self.x_lead += self.v_lead * self.dt
        self.x_ego += self.v_ego * self.dt
        delta_x = self.x_lead - self.x_ego
        delta_v = self.v_lead - self.v_ego
        cost = 0
        done = False
        # if delta_x < 0:  # crash
        #     # done = True
        #     cost = -1000
        # else:
        # cost -= 0.5 * (((delta_x - self.d_default) ** 2) / (delta_x))
        # if delta_x > self.d_default:
        #     cost -= (-20 * math.log10(delta_x - self.d_default)) + (0.02 * ((delta_x - self.d_default) ** 2)) + 20
        # else:
        #     cost -= 1000
        # cost -= min((delta_v ** 2),1000)
        # cost = max(cost, -200)
        cost -= 0.02*(delta_x - self.d_default) ** 2
        # cost -= (delta_v ** 2)
        # if cost > 0:
        #     print(cost)
        return np.array([self.x_lead, self.x_ego, self.v_lead, self.v_ego, self.y_lead, self.y_ego, delta_v, delta_x]), cost, done, {}

    def perfect_action(self):
        delta_x = self.x_lead - self.x_ego
        target_delta_x = self.d_default  # self.d_default + self.t_gap * self.v_ego
        if delta_x < target_delta_x:
            action = 0
        elif self.v_set - self.v_ego < 0:  # keep target speed
            action = 0
        else:
            action = 1
        # todo we want delta_v to be 0 when delta_x ==target_delta_x
        return action


if __name__ == '__main__':
    env = StoppingCar()
    env.reset()
    env.step(1)

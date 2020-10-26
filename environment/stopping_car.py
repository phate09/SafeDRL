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
        self.a_ego = 1  # deceleration/acceleration amount
        self.dt = .1  # delta time
        self.d_default = 10  # minimum safe distance
        self.t_gap = 1.4  # safe distance reaction time
        self.v_set = 30  # speed to drive at if no cars ahead
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        random.seed = 0

    def reset(self):
        self.y_lead = self.y_ego = 0
        self.v_lead = random.uniform(20, 30)
        self.v_ego = random.uniform(30, 30.5)
        self.x_ego = random.uniform(30, 31)
        self.x_lead = random.uniform(90, 92)
        delta_x = self.x_lead - self.x_ego
        delta_v = self.v_lead - self.v_ego
        return np.array([self.x_lead, self.x_ego, self.v_lead, self.v_ego, self.y_lead, self.y_ego, delta_x, delta_v, self.v_set, self.t_gap])

    def step(self, action_ego):
        if action_ego == 0:
            acceleration = -self.a_ego
        else:
            acceleration = self.a_ego
        self.y_ego += -2 * self.y_ego * self.dt + 2 * acceleration
        self.v_ego += self.y_ego * self.dt
        self.v_lead += self.y_lead * self.dt
        self.x_lead += self.v_lead * self.dt
        self.x_ego += self.v_ego * self.dt
        delta_x = self.x_lead - self.x_ego
        delta_v = self.v_lead - self.v_ego
        cost = 0
        if self.x_ego > self.x_lead:  # crash
            done = True
            cost += 1000
        else:
            done = False
            if delta_x < self.d_default + self.t_gap * self.v_ego:  # keep the safe distance
                cost += 10
            cost += abs(self.v_set - self.v_ego)  # try to match v_set speed

        return np.array([self.x_lead, self.x_ego, self.v_lead, self.v_ego, self.y_lead, self.y_ego, delta_x, delta_v, self.v_set, self.t_gap]), -cost, done, {}


if __name__ == '__main__':
    env = StoppingCar()
    env.reset()
    env.step(1)

import math
import random

import gym
from gym import spaces
import numpy as np
import random
import torch
from gym.utils import seeding


class StoppingCar(gym.Env):
    def __init__(self, config=None):
        self.epsilon_input = 0
        self.cost_function_index = 0
        self.use_reduced_state_space = False
        if config is not None:
            self.cost_function_index = config.get("cost_fn", 0)
            self.epsilon_input = config.get("epsilon_input", 0)
            self.use_reduced_state_space = config.get("reduced", False)
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
        self.action_space = spaces.Box(-3, 3, shape=(1,), dtype=np.float32)
        if self.use_reduced_state_space:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.seed()

    def reset(self):
        self.y_lead = self.y_ego = 0
        self.v_lead = self.np_random.uniform(20, 36)
        # self.v_ego = self.np_random.uniform(20, 36)
        self.v_ego = self.v_lead + self.np_random.uniform(-10, 10)
        self.x_ego = self.np_random.uniform(0, 10)
        self.x_lead = self.np_random.uniform(30, 40)
        delta_x = self.x_lead - self.x_ego
        delta_v = self.v_lead - self.v_ego
        if self.use_reduced_state_space:
            return np.array([delta_v, delta_x])
        else:
            return np.array([self.x_lead, self.x_ego, self.v_lead, self.v_ego, self.y_lead, self.y_ego, delta_v, delta_x])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action_ego):
        acceleration = float(action_ego)
        self.y_ego = acceleration
        self.v_ego += acceleration * self.dt
        self.v_ego = min(max(self.v_ego, 0), 40)
        self.v_lead += self.y_lead * self.dt
        self.x_lead += self.v_lead * self.dt
        self.x_ego += self.v_ego * self.dt
        delta_x = (self.x_lead - self.x_ego) + np.random.uniform(-self.epsilon_input, self.epsilon_input) / 2
        delta_v = (self.v_lead - self.v_ego) + np.random.uniform(-self.epsilon_input, self.epsilon_input) / 2
        cost = 0
        done = False
        if self.cost_function_index == 0:
            cost -= (0.02 * (delta_x - self.d_default)) ** 2
        elif self.cost_function_index == 1:
            cost += 1
            if delta_x < 0:  # crash
                done = True
                cost = -1000
        elif self.cost_function_index == 2:
            cost -= (0.03 * (delta_x - self.d_default)) ** 2
            cost -= (0.01 * (delta_v)) ** 2
            if delta_x < 0:  # crash
                cost -= 1
        elif self.cost_function_index == 3:
            cost -= (0.03 * (delta_x - self.d_default)) ** 2
            cost -= (0.01 * (delta_v)) ** 2
            cost -= (0.02 * max(0, acceleration - 3)) ** 2
            if delta_x < 0:  # crash
                cost -= 10
        if self.use_reduced_state_space:
            return np.array([delta_v, delta_x]), cost, done, {}
        else:
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
        return action

    def get_state(self):
        delta_x = (self.x_lead - self.x_ego) + np.random.uniform(-self.epsilon_input, self.epsilon_input) / 2
        delta_v = (self.v_lead - self.v_ego) + np.random.uniform(-self.epsilon_input, self.epsilon_input) / 2
        if self.use_reduced_state_space:
            return np.array([delta_v, delta_x])
        else:
            return np.array([self.x_lead, self.x_ego, self.v_lead, self.v_ego, self.y_lead, self.y_ego, delta_v, delta_x])


if __name__ == '__main__':
    env = StoppingCar()
    env.reset()
    env.step(1)

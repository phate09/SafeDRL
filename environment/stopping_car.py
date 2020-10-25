import random

import gym


class StoppingCar(gym.Env):
    def __init__(self):
        self.xlead = 0  # position lead vehicle
        self.xego = 0  # position ego vehicle
        self.vlead = 0  # velocity lead vehicle
        self.vego = 0  # velocity ego vehicle
        self.ylead = 0  # acceleration lead vehicle
        self.yego = 0  # acceleration ego vehicle
        self.aego = 1  # deceleration/acceleration amount
        self.dt = .1  # delta time
        random.seed = 0

    def reset(self):
        self.xlead = random.uniform(90, 92)
        self.vlead = random.uniform(20, 30)
        self.ylead = self.yego = 0
        self.vego = random.uniform(30, 30.5)
        self.xego = random.uniform(30, 31)
        return self.xlead, self.xego, self.vlead, self.vego, self.ylead, self.yego

    def step(self, action_ego):
        if action_ego == 0:
            acceleration = -self.aego
        else:
            acceleration = self.aego
        self.yego += -2 * self.yego * self.dt + 2 * acceleration
        self.vego += self.yego * self.dt
        self.vlead += self.ylead * self.dt
        self.xlead += self.vlead * self.dt
        self.xego += self.vego * self.dt
        if self.xego > self.xlead:
            done = True
            cost = 1000
        else:
            done = False
            cost = 1
        return (self.xlead, self.xego, self.vlead, self.vego, self.ylead, self.yego), cost, done, {}


if __name__ == '__main__':
    env = StoppingCar()
    env.reset()
    env.step(1)

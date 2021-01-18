import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class BouncingBall(gym.Env):
    def __init__(self, config=None):
        self.v = 0  # velocity
        self.c = 0  # cost/hit counter
        self.p = 0  # position
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.dt = 0.1
        self.seed()
        if config is not None:
            self.dt = config["tau"]

    def reset(self):
        self.p = 7 + self.np_random.uniform(0, 3)
        self.v = 0
        return np.array((self.p, self.v))

    def step(self, action):
        done = False

        cost = 0
        v_prime = self.v - 9.81 * self.dt
        p_prime = max(self.p + self.dt * v_prime, 0)
        if v_prime <= 0 and p_prime <= 0:
            v_prime = -(0.90) * v_prime
            p_prime = 0
            if v_prime <= 1:
                done = True
                # cost += -1000
        if v_prime <= 0 and p_prime > 4 and action == 1:
            v_prime = v_prime - 4
            p_prime = 4
        if v_prime > 0 and p_prime > 4 and action == 1:
            v_prime = -(0.9) * v_prime - 4
            p_prime = 4
        # v_second = v_prime - 9.81 * dt
        # p_second = p_prime + dt * v_prime
        self.p = p_prime
        self.v = v_prime
        cost += -1 if action == 1 else 0
        if not done:
            cost += 1
        return np.array((self.p, self.v)), cost, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == '__main__':
    env = BouncingBall()
    state = env.reset()
    position_list = [state[0]]
    print(state)
    done = False
    i = 0
    while True:
        state, cost, done, _ = env.step(0)
        position_list.append(state[0])
        print(state)
        i += 1
        if i > 500:
            break
        if done:
            print("done")
            break
    import plotly.graph_objects as go

    fig = go.Figure()
    trace1 = go.Scatter(x=list(range(len(position_list))), y=position_list, mode='markers', )
    fig.add_trace(trace1)
    fig.show()

import datetime
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

from training.dqn.dqn_agent import Agent
from environment.pendulum_abstract import PendulumEnv
from utility.Scheduler import Scheduler

currentDT = datetime.datetime.now()
print(f'Start at {currentDT.strftime("%Y-%m-%d %H:%M:%S")}')
seed = 7
# np.random.seed(seed)
env = PendulumEnv()  # gym.make("CartPole-v0")
env.seed(seed)
np.random.seed(seed)
state_size = 2
action_size = 2
agent = Agent(state_size=state_size, action_size=action_size)
agent.load("/home/edoardo/Development/SafeDRL/save/Pendulum_Apr07_12-17-45_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_final.pth")
state = env.reset()
# for i in range(100):
#     env.reset()
#     env.render()
# env.state = np.array([0, 0])
# state = env.state
for i in range(1000):
    action = agent.act(state)
    next_state, r, done, _ = env.step(action)
    env.render()
    state = next_state
    if done:
        break

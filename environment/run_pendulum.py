import os

from dqn.dqn_agent import Agent
from environment.pendulum_abstract import PendulumEnv
import numpy as np

env = PendulumEnv()
# env.state = np.array([-0.57,0.21]) #guaranteed fail
# env.state = np.array([-0.57,0.21]) #guaranteed fail
# env.state = np.array([-0.22,0.05]) #success right
# env.state = np.array([0.25, -1])  # success left
# env.state = np.array([0.39,0.9]) # fail left
# env.state = np.array([0.834,0.497]) # fail
env.state = np.array([0.43,0.575]) # fail left

state_size = 2
agent = Agent(state_size, 2)
agent.load(os.path.expanduser("~/Development") + "/SafeDRL/save/Pendulum_Apr07_12-17-45_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_final.pth")
action = None
for i in range(7):
    # if action is None or np.random.rand() < 0.8:
    action = agent.act(env.state)
    next_state, reward, done, _ = env.step(action)
    env.render()
    if done:
        print(f"exiting at timestep {i}")
        break
print("finished")

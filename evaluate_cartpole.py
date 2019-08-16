import torch
from gym.envs.classic_control import CartPoleEnv

from dqn_agent import Agent

env = CartPoleEnv()
# number of actions
action_size = 2
state_size = 4
score = 0  # initialize the score
agent = Agent(state_size=state_size, action_size=action_size, alpha=0.6)
agent.qnetwork_local.load_state_dict(torch.load('model.pth'))
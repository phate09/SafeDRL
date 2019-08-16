import torch
from gym.envs.classic_control import CartPoleEnv

from dqn_agent import Agent

env = CartPoleEnv()
# number of actions
action_size = 2
state_size = 4
score = 0  # initialize the score
agent = Agent(state_size=state_size, action_size=action_size, alpha=0.6)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
for i in range(1):
    score = 0
    state = env.reset()  # reset the environment
    while True:
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)  # send the action to the environment
        state = next_state
        score += reward
        if done:
            break

    print("Score: {}".format(score))

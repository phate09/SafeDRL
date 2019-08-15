import gym
import numpy as np
from gym.envs.classic_control import CartPoleEnv

env2 = CartPoleEnv()
np.random.seed(0)
env2.seed(0)
done = False
env2.reset()
reward_sum = 0.0
steps = 0
while not done:
    next_state, reward, done, _ = env2.step(np.random.randint(2))
    steps += 1
    reward_sum += reward
print(reward_sum)

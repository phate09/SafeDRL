import datetime
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.envs.classic_control import CartPoleEnv
from tensorboardX import SummaryWriter

from dqn_agent import Agent
from utility.Scheduler import Scheduler

currentDT = datetime.datetime.now()
print(f'Start at {currentDT.strftime("%Y-%m-%d %H:%M:%S")}')
seed = 5
# np.random.seed(seed)
env = CartPoleEnv()  # gym.make("CartPole-v0")
state_size = 4
action_size = 2
STARTING_BETA = 0.5
ALPHA = 0.4
EPS_DECAY = 0.2
MIN_EPS = 0.01

current_time = currentDT.strftime('%b%d_%H-%M-%S')
comment = f"alpha={ALPHA}, min_eps={MIN_EPS}, eps_decay={EPS_DECAY}"
log_dir = os.path.join('runs', current_time + '_' + comment)
os.mkdir(log_dir)
print(f"logging to {log_dir}")
writer = SummaryWriter(log_dir=log_dir)
agent = Agent(state_size=state_size, action_size=action_size, alpha=ALPHA)


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=MIN_EPS):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    betas = Scheduler(STARTING_BETA, 1.0, n_episodes)
    eps = Scheduler(eps_start, eps_end, n_episodes * EPS_DECAY)
    for i_episode in range(n_episodes):
        state = env.reset()  # reset the environment
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps.get(i_episode))
            next_state, reward, done, _ = env.step(action)  # send the action to the environment
            agent.step(state, action, reward, next_state, done, beta=betas.get(i_episode))
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        writer.add_scalar('data/score', score, i_episode)
        writer.add_scalar('data/score_average', np.mean(scores_window), i_episode)
        writer.add_scalar('data/epsilon', eps.get(i_episode), i_episode)
        writer.add_scalar('data/beta', betas.get(i_episode), i_episode)
        # eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print(
            f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} '
            f'eps={eps.get(i_episode):.3f} beta={betas.get(i_episode):.3f}'
            , end="")
        if i_episode + 1 % 100 == 0:
            print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} '
                  f'eps={eps.get(i_episode):.3f} beta={betas.get(i_episode):.3f}')
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        if np.mean(scores_window) >= 197.0:
            print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores


return_scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(return_scores)), return_scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
finish_T = datetime.datetime.now()
print()
print(f"Finish at {finish_T.strftime('%Y-%m-%d %H:%M:%S')}")

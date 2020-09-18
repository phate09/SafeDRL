import datetime
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

from agents.dqn.dqn_agent import Agent
from environment.pendulum_abstract import PendulumEnv
from utility.Scheduler import Scheduler

currentDT = datetime.datetime.now()
print(f'Start at {currentDT.strftime("%Y-%m-%d %H:%M:%S")}')
seed = 5
# np.random.seed(seed)
env = PendulumEnv()  # gym.make("CartPole-v0")
env.seed(seed)
np.random.seed(seed)
state_size = 2
action_size = 2
STARTING_BETA = 0.6  # the higher the more it decreases the influence of high TD transitions
ALPHA = 0.6  # the higher the more aggressive the sampling towards high TD transitions
EPS_DECAY = 0.2
MIN_EPS = 0.01

current_time = currentDT.strftime('%b%d_%H-%M-%S')
comment = f"alpha={ALPHA}, min_eps={MIN_EPS}, eps_decay={EPS_DECAY}"
log_dir = os.path.join('../runs', current_time + '_' + comment)
os.mkdir(log_dir)
print(f"logging to {log_dir}")
writer = SummaryWriter(log_dir=log_dir)
agent = Agent(state_size=state_size, action_size=action_size, alpha=ALPHA)


# agent.qnetwork_local.load_state_dict(torch.load('model.pth'))
# agent.qnetwork_target.load_state_dict(torch.load('model.pth'))

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=MIN_EPS):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    betas = Scheduler(STARTING_BETA, 1.0, n_episodes)
    eps = Scheduler(eps_start, eps_end, round(n_episodes * EPS_DECAY))
    for i_episode in range(n_episodes):
        state = env.reset()  # reset the environment
        score = 0
        action = None
        for t in range(max_t):
            action = agent.act(state, eps.get(i_episode))
            next_state, reward, done, _ = env.step(action)  # send the action to the environment
            agent.step(state, action, reward, next_state, done, beta=betas.get(i_episode))
            # if np.random.rand() > 0.8 and not done:
            #     next_state, reward, done, _ = env.step(action)
            #     agent.step(state, action, reward, next_state, done, beta=betas.get(i_episode))
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
        print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} eps={eps.get(i_episode):.3f} beta={betas.get(i_episode):.3f}', end="")
        if (i_episode + 1) % 100 == 0:
            print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f} eps={eps.get(i_episode):.3f} beta={betas.get(i_episode):.3f}')
            agent.save(os.path.join(log_dir, f"checkpoint_{i_episode + 1}.pth"), i_episode)
    agent.save(os.path.join(log_dir, f"checkpoint_final.pth"), i_episode)
    return scores


return_scores = dqn(n_episodes=6000)

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

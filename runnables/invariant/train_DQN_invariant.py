import datetime
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

from environment.stopping_car import StoppingCar
from training.dqn.safe_dqn_agent import InvariantAgent
from utility.Scheduler import Scheduler

currentDT = datetime.datetime.now()
print(f'Start at {currentDT.strftime("%Y-%m-%d %H:%M:%S")}')
seed = 5
# np.random.seed(seed)
config = {"cost_fn": 1, "simplified": True}
env = StoppingCar(config)
# env = CartPoleEnv()  # gym.make("CartPole-v0")
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
log_dir = os.path.join('/home/edoardo/Development/SafeDRL/runs', current_time + '_' + comment)
os.mkdir(log_dir)
print(f"logging to {log_dir}")
writer = SummaryWriter(log_dir=log_dir)
agent = InvariantAgent(state_size=state_size, action_size=action_size, alpha=ALPHA)
# agent.load("/home/edoardo/Development/SafeDRL/runs/Aug05_11-16-16_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_6000.pth")

# agent.qnetwork_local.load_state_dict(torch.load('model.pth'))
# agent.qnetwork_target.load_state_dict(torch.load('model.pth'))

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=MIN_EPS):
    scores = []  # list containing scores from each episode
    timesteps = []  # list containing number of timesteps from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    timesteps_window = deque(maxlen=100)  # last 100 timesteps

    betas = Scheduler(STARTING_BETA, 1.0, n_episodes)
    eps = Scheduler(eps_start, eps_end, int(n_episodes * EPS_DECAY))
    # val_data = GridSearchDataset(shuffle=True)
    for i_episode in range(n_episodes):
        # state = env.reset()  # reset the environment
        # state = val_data[i_episode%len(val_data)]
        state = np.array([np.random.uniform(-10,30),np.random.uniform(-10,40)])
        state=state.numpy()
        score = 0
        t = 0
        action = None
        agent_error = None
        done = False
        done_once = False
        for t in range(max_t):
            action = agent.act(state, eps.get(i_episode))

            # next_state, reward, done, _ = env.step(action)  # send the action to the environment
            next_state, reward, done, _ = env.compute_successor(state,action)
            agent_error = agent.step(state, action, reward, next_state, done, beta=betas.get(i_episode))
            if agent_error is not None:
                writer.add_scalar('loss/agent_loss', agent_error.item(), i_episode)
            state = next_state
            score += reward
            if done:
                done_once=True
                # break
        invariant_loss = agent.episode_end(done_once)
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        timesteps.append(t)
        timesteps_window.append(t)
        writer.add_scalar('data/score', score, i_episode)
        writer.add_scalar('data/timesteps', t, i_episode)
        writer.add_scalar('data/score_average', np.mean(scores_window), i_episode)
        writer.add_scalar('data/timesteps_average', np.mean(timesteps_window), i_episode)
        writer.add_scalar('data/epsilon', eps.get(i_episode), i_episode)
        writer.add_scalar('data/beta', betas.get(i_episode), i_episode)
        if invariant_loss is not None:
            writer.add_scalar('loss/invariant_loss', invariant_loss, i_episode)

        # eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f}\tAverage Timesteps: {np.mean(timesteps_window):.2f} eps={eps.get(i_episode):.3f} beta={betas.get(i_episode):.3f}', end="")
        if (i_episode + 1) % 100 == 0:
            print(f'\rEpisode {i_episode + 1}\tAverage Score: {np.mean(scores_window):.2f}\tAverage Timesteps: {np.mean(timesteps_window):.2f} eps={eps.get(i_episode):.3f} beta={betas.get(i_episode):.3f}')
            agent.save(os.path.join(log_dir, f"checkpoint_{i_episode+1}.pth"), i_episode)
    agent.save(os.path.join(log_dir, f"checkpoint_final.pth"), i_episode)
    return scores


return_scores = dqn(n_episodes=8000, max_t=1000)

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

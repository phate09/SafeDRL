import datetime
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.nn
from matplotlib import cm
from matplotlib.colors import Normalize

import utils
from environment.stopping_car import StoppingCar
from runnables.invariant.retrain_agent import GridSearchDataset
from training.dqn.safe_dqn_agent import InvariantAgent

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

agent = InvariantAgent(state_size=state_size, action_size=action_size, alpha=ALPHA)
agent.load(os.path.join(utils.get_save_dir(),"Aug05_16-14-33_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_3000.pth"), invariant=False)
agent2 = InvariantAgent(state_size=state_size, action_size=action_size, alpha=ALPHA)
# agent2.load("/home/edoardo/Development/SafeDRL/runs/Aug20_09-16-25_invariant/checkpoint_1000.pth")
agent2.load(os.path.join(utils.get_save_dir(),"Aug20_11-58-57_invariant/checkpoint_1500.pth"))

agent_model = agent.qnetwork_local
invariant_model = agent2.inetwork_local

agent_model.cpu()
invariant_model.cpu()
val_data = GridSearchDataset()
random.seed(0)
x_data = []
xprime_data = []
y_data = []
for i, data in enumerate(random.sample(val_data.dataset, k=7000)):
    action = torch.argmax(agent_model(data)).item()
    value = invariant_model(data)[action].item()
    next_state_np, reward, done, _ = StoppingCar.compute_successor(data.numpy(), action)
    x_data.append(data.numpy())
    xprime_data.append(next_state_np)
    y_data.append(value)

x_data = np.array(x_data)
xprime_data = np.array(xprime_data)
y_data = np.array(y_data)
x_full = x_data[:, 0]
y_full = x_data[:, 1]
u_full = xprime_data[:, 0] - x_data[:, 0]
v_full = xprime_data[:, 1] - x_data[:, 1]

colors_full = y_data

norm = Normalize(vmax=1.0, vmin=-1.0)
norm.autoscale(colors_full)
# we need to normalize our colors array to match it colormap domain
# which is [0, 1]

colormap = cm.bwr
plt.figure(figsize=(18, 16), dpi=80)
# plt.quiver(x, y, old_u, old_v, color="yellow", angles='xy',
#            scale_units='xy', scale=1, pivot='mid', zorder=1)
plt.quiver(x_full, y_full, u_full, v_full, color=colormap(norm(colors_full)), angles='xy',
           scale_units='xy', scale=1, pivot='mid', zorder=0)  # colormap(norm(colors))
plt.title("New")
plt.xlim([-30, 30])
plt.ylim([-10, 40])
plt.show()

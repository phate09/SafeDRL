import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.colors import Normalize

from agents.dqn.safe_dqn_agent import InvariantAgent
from environment.stopping_car import StoppingCar
from runnables.invariant.retrain_agent import GridSearchDataset

seed = 5
config = {"cost_fn": 1, "simplified": True}
env = StoppingCar(config)
env.seed(seed)
np.random.seed(seed)
state_size = 2
action_size = 2
ALPHA = 0.6  # the higher the more aggressive the sampling towards high TD transitions
agent = InvariantAgent(state_size=state_size, action_size=action_size, alpha=ALPHA)

# agent.load("/home/edoardo/Development/SafeDRL/runs/Aug05_11-16-16_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_6000.pth")
# agent.load("/home/edoardo/Development/SafeDRL/runs/Aug05_14-55-31_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_6000.pth")
agent.load("/home/edoardo/Development/SafeDRL/runs/Aug05_16-14-33_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_3000.pth")
agent_model = agent.qnetwork_local
agent_model.cpu()
invariant_model = agent.inetwork_local
invariant_model.cpu()
val_data = GridSearchDataset()

random.seed(0)
x_data = []
xprime_data = []
old_xprime_data = []
y_data = []
changed_indices = []
for i, data in enumerate(random.sample(val_data.dataset, k=7000)):
    action = torch.argmax(agent_model(data)).item()
    value = invariant_model(data)[action].item()
    next_state_np, reward, done, _ = StoppingCar.compute_successor(data.numpy(), action)
    # old_action = torch.argmax(old_agent_model(data)).item()
    # next_state_np_old, _, _, _ = StoppingCar.compute_successor(data.numpy(), old_action)
    x_data.append(data.numpy())
    xprime_data.append(next_state_np)
    y_data.append(value)
    # old_xprime_data.append(next_state_np_old)
    # if action != old_action:
    #     changed_indices.append(i)

x_data = np.array(x_data)
xprime_data = np.array(xprime_data)
old_xprime_data = np.array(old_xprime_data)
changed_indices = np.array(changed_indices)
y_data = np.array(y_data)
# x = x_data[:, 0][changed_indices]
# y = x_data[:, 1][changed_indices]
# u = xprime_data[:, 0][changed_indices] - x_data[:, 0][changed_indices]
# v = xprime_data[:, 1][changed_indices] - x_data[:, 1][changed_indices]
x_full = x_data[:, 0]
y_full = x_data[:, 1]
u_full = xprime_data[:, 0] - x_data[:, 0]
v_full = xprime_data[:, 1] - x_data[:, 1]
# u_old_full = old_xprime_data[:, 0] - x_data[:, 0]
# v_old_full = old_xprime_data[:, 1] - x_data[:, 1]

# colors = y_data[changed_indices]
colors_full = y_data

norm = Normalize(vmax=1.0, vmin=-1.0)
norm.autoscale(colors_full)
# we need to normalize our colors array to match it colormap domain
# which is [0, 1]

colormap = cm.bwr
# plt.figure(figsize=(18, 16), dpi=80)
# # plt.quiver(x, y, old_u, old_v, color="yellow", angles='xy',
# #            scale_units='xy', scale=1, pivot='mid', zorder=1)
# plt.quiver(x, y, u, v, color=colormap(norm(colors)), angles='xy',
#            scale_units='xy', scale=1, pivot='mid', zorder=0)  # colormap(norm(colors))
# plt.title("Diff")
# plt.xlim([-30, 30])
# plt.ylim([-10, 40])
# plt.show()
plt.figure(figsize=(18, 16), dpi=80)
# plt.quiver(x, y, old_u, old_v, color="yellow", angles='xy',
#            scale_units='xy', scale=1, pivot='mid', zorder=1)
plt.quiver(x_full, y_full, u_full, v_full, color=colormap(norm(colors_full)), angles='xy',
           scale_units='xy', scale=1, pivot='mid', zorder=0)  # colormap(norm(colors))
plt.title("New")
plt.xlim([-30, 30])
plt.ylim([-10, 40])
plt.show()
plt.show()
# plt.figure(figsize=(18, 16), dpi=80)
# plt.quiver(x_full, y_full, u_old_full, v_old_full, color=colormap(norm(colors_full)), angles='xy',
#            scale_units='xy', scale=1, pivot='mid', zorder=0)  # colormap(norm(colors))
# plt.title("Old")
# plt.xlim([-30, 30])
# plt.ylim([-10, 40])
# plt.show()

# %%
import os
import pickle
import time

import gym
import ray
import torch
import importlib
import mosaic.utils as utils
from dqn.dqn_agent import Agent
from environment.pendulum_abstract import PendulumEnv
from mosaic.hyperrectangle import HyperRectangle_action, HyperRectangle
from prism.shared_rtree import SharedRtree
import prism.state_storage
import symbolic.unroll_methods as unroll_methods
import verification_runs.domain_explorers_load
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import mosaic.hyperrectangle_serialisation as serialisation

gym.logger.set_level(40)
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
local_mode = False
allow_compute = True
allow_load = False
if not ray.is_initialized():
    ray.init(local_mode=local_mode, include_dashboard=True, log_to_driver=False)
serialisation.register_serialisers()
n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
storage = prism.state_storage.StateStorage()
storage.reset()
rounding = 2
precision = 10 ** (-rounding)
env = PendulumEnv()
state_size = 2
# param_grid = {'param1': list(np.arange(-env.max_angle, env.max_angle, precision).round(rounding)), 'param2': list(np.arange(-1, 1, precision).round(rounding))}
param_grid = {'param1': list(np.arange(0.45, 0.52, precision).round(rounding)), 'param2': list(np.arange(0.02, 0.18, precision).round(rounding))}
grid = ParameterGrid(param_grid)
agent = Agent(state_size, 2)
agent.load(os.path.expanduser("~/Development") + "/SafeDRL/save/Pendulum_Apr07_12-17-45_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_final.pth")
print(f"Building the tree")
rtree = SharedRtree()
rtree.reset(state_size)
# rtree.load_from_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p", rounding)
print(f"Finished building the tree")
# current_interval = HyperRectangle.from_tuple(tuple([(-0.05, 0.05), (-0.05, 0.05)]))
root = "root"
storage.root = root
storage.graph.add_node(storage.root)
horizon = 4
t = 0
# %%
if allow_load:
    storage.load_state(f"/home/edoardo/Development/SafeDRL/save/nx_graph_concrete_e{rounding}.p")
# %%
if allow_compute:
    current_intervals = []
    for params in grid:
        state = np.array((params['param1'], params['param2']))
        current_intervals.append(state)
    storage.store_successor_multi([(root, tuple(x)) for x in current_intervals])
    print("Stored initial successors")
    for i in range(horizon):
        print(f"Iteration {i}, # states: {len(current_intervals)}")
        new_states = []
        for current_interval in current_intervals:
            env.state = current_interval
            action = agent.act(current_interval)
            next_state, reward, done, _ = env.step(action)
            next_state2, reward2, done2, _ = env.step(action)
            # next_state = next_state.round(rounding)
            # next_state2 = next_state2.round(rounding)
            new_states.append(next_state)
            new_states.append(next_state2)
            storage.store_sticky_successors(tuple(next_state), tuple(next_state2), tuple(current_interval))
            if done:
                storage.mark_as_fail([tuple(next_state)])
            if done2:
                storage.mark_as_fail([tuple(next_state2)])
        current_intervals = new_states
    print("Finished")
    storage.save_state(f"/home/edoardo/Development/SafeDRL/save/nx_graph_concrete_e{rounding}.p")
    storage.recreate_prism(horizon * 2)
    storage.save_state(f"/home/edoardo/Development/SafeDRL/save/nx_graph_concrete_e{rounding}.p")
# %%


# utils.show_heatmap(unroll_methods.get_property_at_timestep(storage, 1, ["lb"]), rounding=2, concrete=True)
storage.recreate_prism(horizon * 2)
utils.show_heatmap(unroll_methods.get_property_at_timestep(storage, 1, ["ub"]), rounding=3, concrete=True, title=f"UB concrete horizon:{horizon}")
utils.show_heatmap(unroll_methods.get_property_at_timestep(storage, 1, ["lb"]), rounding=3, concrete=True, title=f"LB concrete horizon:{horizon}")

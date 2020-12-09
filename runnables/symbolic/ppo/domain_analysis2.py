# %%
import os
import time
import gym
import ray
import torch
import numpy as np
from sklearn.model_selection import ParameterGrid

import mosaic.hyperrectangle_serialisation as serialisation
import mosaic.utils as utils
import prism.state_storage
import symbolic.unroll_methods as unroll_methods
import utility.domain_explorers_load
from agents.dqn.dqn_sequential import TestNetwork, TestNetwork2
from agents.ray_utils import load_sequential_from_ray, get_pendulum_ppo_agent
from mosaic.hyperrectangle import HyperRectangle_action, HyperRectangle
from mosaic.interval import Interval
from plnn.verification_network_sym import SymVerificationNetwork
from prism.shared_rtree import SharedRtree
from symbolic.symbolic_interval import Symbolic_interval, Interval_network

#
gym.logger.set_level(40)
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
local_mode = True
if not ray.is_initialized():
    ray.init(local_mode=local_mode, include_dashboard=False, log_to_driver=False)
n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
storage = prism.state_storage.StateStorage()
storage.reset()
rounding = 2
precision = 10 ** (-rounding)
explorer, verification_model, env, current_interval, state_size, env_class = utility.domain_explorers_load.generatePendulumDomainExplorerPPO(precision, rounding, sym=True)
delta_x = precision * 2
delta_y = precision * 2
param_grid = {'param1': list(np.arange(-0.79, 0.79, delta_x).round(rounding)), 'param2': list(np.arange(-1, 1, delta_y).round(rounding))}
grid = ParameterGrid(param_grid)
print(f"Building the tree")
rtree = SharedRtree()
rtree.reset(state_size)
# rtree.load_from_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p", rounding)
print(f"Finished building the tree")
# current_interval = tuple([(-0.3, -0.2), (-0.7, -0.6)])
current_interval = current_interval.round(rounding)
remainings = [current_interval]
storage.root = HyperRectangle_action.from_hyperrectangle(current_interval, None)
storage.graph.add_node(storage.root)
horizon = 3
t = 0
# %%
# storage.load_state(f"/home/edoardo/Development/SafeDRL/save/ppo_nx_graph_e{rounding}.p")
# rtree.load_from_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p", rounding)
# %%
iterations = 0
time_from_last_save = time.time()
while True:
    print(f"Iteration {iterations}")
    print(f"Total states:{storage.graph.number_of_nodes()}")
    split_performed = unroll_methods.probability_iteration_PPO(storage, rtree, precision, rounding, env_class, n_workers, explorer, verification_model, state_size, horizon=horizon,
                                                           allow_assign_actions=True)
    if time.time() - time_from_last_save >= 60 * 5:
        storage.save_state(f"/home/edoardo/Development/SafeDRL/save/ppo_nx_graph_e{rounding}.p")
        # rtree.save_to_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p")
        print("Graph Saved - Checkpoint")
        time_from_last_save = time.time()
    if not split_performed or iterations == 1000:
        # utils.save_graph_as_dot(storage.graph)
        if not split_performed:
            print("No more split performed")
        break
    iterations += 1

storage.save_state(f"/home/edoardo/Development/SafeDRL/save/ppo_nx_graph_e{rounding}.p")
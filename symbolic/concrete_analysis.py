import itertools
import os
import pickle
import time

import gym
import ray
import importlib
import mosaic.utils as utils
from prism.shared_rtree import SharedRtree
import prism.state_storage
import symbolic.unroll_methods as unroll_methods
import verification_runs.domain_explorers_load
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import numpy as np

# %%
gym.logger.set_level(40)
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
local_mode = True
if not ray.is_initialized():
    ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
storage = prism.state_storage.StateStorage()
storage.reset()
rounding = 3
precision = 10 ** (-rounding)
explorer, verification_model, env, current_interval, state_size, env_class = verification_runs.domain_explorers_load.generateCartpoleDomainExplorer(precision, rounding)
print(f"Building the tree")
rtree = SharedRtree()
rtree.reset(state_size)
print(f"Finished building the tree")
storage.root = (utils.round_tuple(current_interval, rounding), None)
storage.graph.add_node(storage.root)
horizon = 5
# %% # generate every possible permutation within the boundaries of current_interval
remainings = []
grid = np.mgrid[tuple(slice(current_interval[d][0], current_interval[d][1], precision) for d in range(state_size))]

l = [list(range(x)) for x in grid[0].shape]
permutations = [tuple(x) for x in itertools.product(*l)]
for indices in permutations:
    new_index = (slice(None),) + indices
    values = grid[new_index]
    interval = tuple([(float(round(x, rounding)), float(round(x, rounding))) for x in values])
    remainings.append(interval)

# %% first iteration
rtree.load_from_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p", rounding)
loaded = storage.load_state(f"/home/edoardo/Development/SafeDRL/save/nx_graph_e{rounding}_concrete.p")
if not loaded:
    assigned_intervals, ignore_intervals = unroll_methods.assign_action_to_blank_intervals(remainings, explorer, verification_model, n_workers, rounding)  # compute the action in each single state
    storage.graph.add_edges_from([(storage.root, x) for x in assigned_intervals], p=1.0)  # assign single intervals as direct successors of root
    next_to_compute = unroll_methods.compute_successors(env_class, assigned_intervals, n_workers, rounding, storage)  # compute successors and store result in graph
    storage.save_state(f"/home/edoardo/Development/SafeDRL/save/nx_graph_e{rounding}_concrete.p")
# %%
iterations = 0
time_from_last_save = time.time()
while True:
    print(f"Iteration {iterations}")
    split_performed = unroll_methods.probability_iteration(storage, rtree, precision, rounding, env_class, n_workers, explorer, verification_model, state_size, horizon=horizon,
                                                           allow_assign_actions=True, allow_merge=False)
    if time.time() - time_from_last_save >= 60 * 5:
        storage.save_state(f"/home/edoardo/Development/SafeDRL/save/nx_graph_e{rounding}_concrete.p")
        rtree.save_to_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}_concrete.p")
        print("Graph Saved - Checkpoint")
        time_from_last_save = time.time()
    if not split_performed or iterations == 1000:
        # utils.save_graph_as_dot(storage.graph)
        if not split_performed:
            print("No more split performed")
        break
    iterations += 1
# %%
storage.save_state(f"/home/edoardo/Development/SafeDRL/save/nx_graph_e{rounding}_concrete.p")
rtree.save_to_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}_concrete.p")

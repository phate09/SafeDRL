# %%
import os
import pickle
import time

import gym
import ray
import importlib
import mosaic.utils as utils
from mosaic.hyperrectangle import HyperRectangle_action, HyperRectangle
from prism.shared_rtree import SharedRtree
import prism.state_storage
import symbolic.unroll_methods as unroll_methods
import verification_runs.domain_explorers_load
import numpy as np
import pandas as pd
import mosaic.hyperrectangle_serialisation as serialisation

gym.logger.set_level(40)
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
local_mode = False
allow_compute = False
allow_save = False
allow_load = True
if not ray.is_initialized():
    ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
serialisation.register_serialisers()
n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
storage = prism.state_storage.StateStorage()
storage.reset()
rounding = 2
precision = 10 ** (-rounding)
explorer, verification_model, env, current_interval, state_size, env_class = verification_runs.domain_explorers_load.generatePendulumDomainExplorer(precision, rounding, sym=True)
print(f"Building the tree")
rtree = SharedRtree()
rtree.reset(state_size)
# rtree.load_from_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p", rounding)
print(f"Finished building the tree")
# current_interval = HyperRectangle.from_tuple(tuple([(-0.05, 0.05), (-0.05, 0.05)]))
current_interval = current_interval.round(rounding)
remainings = [current_interval]
root = HyperRectangle_action.from_hyperrectangle(current_interval, None)
storage.root = root
storage.graph.add_node(storage.root)
horizon = 7
t = 0
# %%
if allow_load:
    storage.load_state(f"/home/edoardo/Development/SafeDRL/save/nx_graph_e{rounding}.p")
    rtree.load_from_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p", rounding)
# %%
if allow_compute:
    iterations = 0
    time_from_last_save = time.time()
    while True:
        print(f"Iteration {iterations}")
        split_performed = unroll_methods.probability_iteration(storage, rtree, precision, rounding, env_class, n_workers, explorer, verification_model, state_size, horizon=horizon,
                                                               allow_assign_actions=True, allow_refine=False)
        if time.time() - time_from_last_save >= 60 * 5 and allow_save:
            storage.save_state(f"/home/edoardo/Development/SafeDRL/save/nx_graph_e{rounding}.p")
            rtree.save_to_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p")
            print("Graph Saved - Checkpoint")
            time_from_last_save = time.time()
        if not split_performed or iterations == 10:
            # utils.save_graph_as_dot(storage.graph)
            if not split_performed:
                print("No more split performed")
            break
        iterations += 1
# %%
if allow_save:
    storage.save_state(f"/home/edoardo/Development/SafeDRL/save/nx_graph_e{rounding}.p")
    rtree.save_to_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p")
# %%
# unroll_methods.remove_spurious_nodes(storage.graph)
# storage.remove_unreachable()
# storage.recreate_prism()
# utils.save_graph_as_dot(storage.graph)
utils.show_heatmap(unroll_methods.get_property_at_timestep(storage, 1, ["lb"]),rounding=2)
utils.show_heatmap(unroll_methods.get_property_at_timestep(storage, 1, ["ub"]),rounding=2)

# %%
import os
import pickle
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

gym.logger.set_level(40)
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
local_mode = False
if not ray.is_initialized():
    ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
storage = prism.state_storage.StateStorage()
storage.reset()
rounding = 1
precision = 10 ** (-rounding)
explorer, verification_model, env, current_interval, state_size, env_class = verification_runs.domain_explorers_load.generatePendulumDomainExplorer(precision, rounding)
print(f"Building the tree")
rtree = SharedRtree()
rtree.reset(state_size)
# rtree.load_from_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e1.p", rounding)
union_states_total = rtree.tree_intervals()
print(f"Finished building the tree")
# current_interval = tuple([(-0.3, -0.2), (-0.7, -0.6)])
remainings = [current_interval]
storage.root = (utils.round_tuple(current_interval, rounding), None)
storage.graph.add_node(storage.root)
horizon = 4
t = 0
# %%
# for i in range(horizon):
#     remainings = unroll_methods.analysis_iteration(remainings, n_workers, rtree, env_class, explorer, verification_model, state_size, rounding, storage)
#     t = t + 1
#     boundaries = unroll_methods.compute_boundaries([(x, True) for x in remainings])
#     print(boundaries)
# # %%
# storage.save_state("/home/edoardo/Development/SafeDRL/save")
# rtree.save_to_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e1.p")
# pickle.dump(t, open("/home/edoardo/Development/SafeDRL/save/t.p", "wb+"))
# pickle.dump(remainings, open("/home/edoardo/Development/SafeDRL/save/remainings.p", "wb+"))
# print("Checkpoint Saved...")
# %%
# remainings = pickle.load(open("/home/edoardo/Development/SafeDRL/save/remainings.p", "rb"))
# t = pickle.load(open("/home/edoardo/Development/SafeDRL/save/t.p", "rb"))
# storage.load_state("/home/edoardo/Development/SafeDRL/save")
# rtree.load_from_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e1.p", rounding)
# %%
iterations = 0
while True:
    iterations += 1
    split_performed = unroll_methods.probability_iteration(storage, rtree, precision, rounding, env_class, n_workers, explorer, verification_model, state_size, horizon=horizon, max_iteration=-1)
    if not split_performed or iterations == 60:
        utils.save_graph_as_dot(storage.graph)
        break
# %%
# %%
unroll_methods.remove_spurious_nodes(storage.graph)
storage.remove_unreachable()
storage.recreate_prism()
utils.save_graph_as_dot(storage.graph)
utils.show_heatmap([(x, prob) for (x, action), prob in unroll_methods.get_property_at_timestep(storage, 1, "lb")])

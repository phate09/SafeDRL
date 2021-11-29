# %%
import os
import time

import gym
import ray

import mosaic.utils as utils
import prism.state_storage
import symbolic.unroll_methods as unroll_methods
import utility.domain_explorers_load
from prism.shared_rtree import SharedRtree

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
explorer, verification_model, env, current_interval, state_size, env_class = utility.domain_explorers_load.generatePendulumDomainExplorer(precision, rounding,sym=True)
print(f"Building the tree")
rtree = SharedRtree()
rtree.reset(state_size)
# rtree.load_from_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p", rounding)
print(f"Finished building the tree")
# current_interval = tuple([(-0.3, -0.2), (-0.7, -0.6)])
remainings = [current_interval]
storage.root = (utils.round_tuple(current_interval, rounding), None)
storage.graph.add_node(storage.root)
horizon = 9
t = 0
# %%
# storage.load_state(f"/home/edoardo/Development/SafeDRL/save/nx_graph_e{rounding}.p")
# rtree.load_from_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p", rounding)
# %%
iterations = 0
time_from_last_save = time.time()
while True:
    print(f"Iteration {iterations}")
    split_performed = unroll_methods.probability_iteration(storage, rtree, precision, rounding, env_class, n_workers, explorer, verification_model, state_size, horizon=horizon,
                                                           allow_assign_actions=True)
    # if time.time() - time_from_last_save >= 60 * 5:
    #     storage.save_state(f"/home/edoardo/Development/SafeDRL/save/nx_graph_e{rounding}.p")
    #     rtree.save_to_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p")
    #     print("Graph Saved - Checkpoint")
    #     time_from_last_save = time.time()
    if not split_performed or iterations == 1000:
        # utils.save_graph_as_dot(storage.graph)
        if not split_performed:
            print("No more split performed")
        break
    iterations += 1
# %%
# storage.save_state(f"/home/edoardo/Development/SafeDRL/save/nx_graph_e{rounding}.p")
# rtree.save_to_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p")
# %%
# unroll_methods.remove_spurious_nodes(storage.graph)
# storage.remove_unreachable()
# storage.recreate_prism()
# utils.save_graph_as_dot(storage.graph)
utils.show_heatmap([(x, prob) for (x, action), prob in unroll_methods.get_property_at_timestep(storage, 1, ["lb"])])

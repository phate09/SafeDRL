# %%
import os
import pickle
import gym
import ray
from mosaic.utils import round_tuple, show_heatmap, show_plot
from prism.shared_rtree import SharedRtree
from prism.state_storage import StateStorage
from symbolic.unroll_methods import probability_iteration, get_property_at_timestep, analysis_iteration, compute_boundaries
from verification_runs.domain_explorers_load import generatePendulumDomainExplorer

gym.logger.set_level(40)
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
local_mode = False
if not ray.is_initialized():
    ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
storage = StateStorage()
storage.reset()
rounding = 1
precision = 10 ** (-rounding)
explorer, verification_model, env, current_interval, state_size, env_class = generatePendulumDomainExplorer(precision, rounding)
print(f"Building the tree")
rtree = SharedRtree()
rtree.reset(state_size)
rtree.load_from_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e1.p", rounding)
union_states_total = rtree.tree_intervals()
print(f"Finished building the tree")
remainings = [current_interval]
storage.root = round_tuple(current_interval, rounding)
horizon = 4
t = 0
# %%
# for i in range(4):
#     remainings = analysis_iteration(remainings, n_workers, rtree, env_class, explorer, verification_model, state_size, rounding, storage)
#     t = t + 1
#     boundaries = compute_boundaries([(x, True) for x in remainings])
#     print(boundaries)
#     rtree.save_to_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e1.p")
# %%
# storage.save_state("/home/edoardo/Development/SafeDRL/save")
# rtree.save_to_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e1.p")
# pickle.dump(t, open("/home/edoardo/Development/SafeDRL/save/t.p", "wb+"))
# pickle.dump(remainings, open("/home/edoardo/Development/SafeDRL/save/remainings.p", "wb+"))
# print("Checkpoint Saved...")

# %%
remainings = pickle.load(open("/home/edoardo/Development/SafeDRL/save/remainings.p", "rb"))
t = pickle.load(open("/home/edoardo/Development/SafeDRL/save/t.p", "rb"))
storage.load_state("/home/edoardo/Development/SafeDRL/save")
rtree.load_from_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e1.p", rounding)
storage.recreate_prism()
# %%
probability_iteration(storage, rtree, precision, rounding, env_class, n_workers, explorer, verification_model, state_size, horizon=horizon, max_iteration=-1)
# %%
list_to_show = get_property_at_timestep(storage,1,"lb")
show_heatmap(list_to_show)
list_to_show = get_property_at_timestep(storage,2,"lb")
show_heatmap(list_to_show)
list_to_show = get_property_at_timestep(storage,3,"lb")
show_heatmap(list_to_show)
# show_plot(rtree.tree_intervals())

# %%
import os
import pickle
import gym
import ray
from prism.shared_rtree import SharedRtree
from prism.state_storage import StateStorage
from symbolic.unroll_methods import analysis_iteration, compute_boundaries, \
    probability_iteration
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
# rtree.load_from_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e1.p", rounding)
union_states_total = rtree.tree_intervals()
print(f"Finished building the tree")
remainings = [current_interval]

# remainings = pickle.load(open("/home/edoardo/Development/SafeDRL/save/remainings.p", "rb"))
# remainings=remainings[0:5000]
# remainings_overlaps = pickle.load(open("/home/edoardo/Development/SafeDRL/save/remainings_overlaps.p", "rb"))
# remainings_overlaps = remove_overlaps([(x, None) for x in remainings],rounding,n_workers,state_size)
# merged_intervals = merge_supremum3([(x, None) for x in remainings],rounding)
# show_plot([x for x in union_states_total] + [(x[0], "Brown") for x in safe_states_merged] + [(x[0], "Purple") for x in unsafe_states_merged])
t = 0
# %%
# intervals = rtree.tree_intervals()
# show_plot(intervals)
# %%
for i in range(4):
    remainings = analysis_iteration(remainings, t, n_workers, rtree, env_class, explorer, verification_model, state_size, rounding, storage)
    t = t + 1
    boundaries = compute_boundaries([(x, True) for x in remainings])
    print(boundaries)
    rtree.save_to_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e1.p")
# %%
storage.save_state("/home/edoardo/Development/SafeDRL/save")
rtree.save_to_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e1.p")
pickle.dump(t, open("/home/edoardo/Development/SafeDRL/save/t.p", "wb+"))
pickle.dump(remainings, open("/home/edoardo/Development/SafeDRL/save/remainings.p", "wb+"))
print("Checkpoint Saved...")

# %%
remainings = pickle.load(open("/home/edoardo/Development/SafeDRL/save/remainings.p", "rb"))
t = pickle.load(open("/home/edoardo/Development/SafeDRL/save/t.p", "rb"))
storage.load_state("/home/edoardo/Development/SafeDRL/save")
rtree.load_from_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e1.p", rounding)
# %%
probability_iteration(storage,rtree, precision, rounding, env_class, n_workers, explorer, verification_model, state_size,"/home/edoardo/Development/SafeDRL/save/")

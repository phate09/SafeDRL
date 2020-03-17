# %%
import os
import pickle
import gym
import jsonpickle

from mosaic.utils import round_tuples
from prism.shared_rtree import get_rtree
from verification_runs.aggregate_abstract_domain import merge_list_tuple

gym.logger.set_level(40)
from py4j.java_collections import ListConverter
from py4j.java_gateway import JavaGateway

from symbolic.unroll_methods import *

os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
gateway = JavaGateway()
storage: StateStorage = get_storage()
storage.reset()
env = CartPoleEnv_abstract()
s = env.reset()
current_interval = s
explorer, verification_model = generateCartpoleDomainExplorer(1e-1)
# reshape with tuples
current_interval = tuple([(float(x.a), float(x.b)) for i, x in enumerate(current_interval)])
precision = 1e-6
rounding = 6

local_mode = False
if not ray.is_initialized():
    ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1

union_states_total = []
if os.path.exists('save/union_states_total.p'):
    union_states_total = pickle.load(open("/home/edoardo/Development/SafeDRL/save/union_states_total.p", "rb"))
    union_states_total = round_tuples(union_states_total, rounding=rounding)
    union_states_total = merge_list_tuple(union_states_total, n_workers=n_workers)
# else:
#     with open("./save/t_states.json", 'r') as f:
#         t_states = jsonpickle.decode(f.read())
#     safe_states = t_states[0][0]
#     unsafe_states = t_states[0][1]
#     safe_states_total = [tuple([(float(x[0]), float(x[1])) for x in k]) for k in safe_states]
#     unsafe_states_total = [tuple([(float(x[0]), float(x[1])) for x in k]) for k in unsafe_states]
#     union_states_total = [(x, True) for x in safe_states_total] + [(x, False) for x in unsafe_states_total]

if os.path.exists('save/rtree.dat') and os.path.exists('save/rtree.idx'):
    print("Loading the tree")
    os.remove('save/rtree.dat')
    os.remove('save/rtree.idx')
    rtree = get_rtree()
    rtree.load(union_states_total)
    print("Finished loading the tree")
else:
    print(f"Tree is missing, rebuilding it")
    rtree = get_rtree()
    rtree.load(union_states_total)
    print(f"Finished building the tree")
rtree = get_rtree()
remainings = [current_interval]
t = 0
parent_id = storage.store(current_interval, t)
#
last_time_remaining_number = -1

failed = []
failed_area = 0
terminal_states = []

# %%
# merge_list_tuple(union_states_total,n_workers)
# print("finished")
# input("Press key")
# %%
# remainings = [((0.4939272701740265, 0.5060727596282959), (0.47243446111679077, 0.5275655388832092), (0.5377258062362671, 0.5754516124725342), (0.9780327081680298, 1.009901523590088))]
for i in range(6):
    remainings = analysis_iteration(remainings, t, terminal_states, failed, n_workers, rtree, env, explorer, storage, [failed_area], union_states_total, rounding)
    t = t + 1
    storage.save_state("/home/edoardo/Development/SafeDRL/save")
    pickle.dump(union_states_total, open("/home/edoardo/Development/SafeDRL/save/union_states_total.p", "wb+"))
    pickle.dump(terminal_states, open("/home/edoardo/Development/SafeDRL/save/terminal_states.p", "wb+"))
    pickle.dump(t, open("/home/edoardo/Development/SafeDRL/save/t.p", "wb+"))
    pickle.dump(t, open("/home/edoardo/Development/SafeDRL/save/remainings.p", "wb+"))
    print("Checkpoint Saved...")
    boundaries = [[999, 0], [999, 0], [999, 0], [999, 0]]
    for interval in remainings:
        for d in range(len(interval)):
            boundaries[d] = [min(boundaries[d][0], interval[d][0]), max(boundaries[d][0], interval[d][1])]
    print(boundaries)
# %%
union_states_total = pickle.load(open("/home/edoardo/Development/SafeDRL/save/union_states_total.p", "rb"))
terminal_states = pickle.load(open("/home/edoardo/Development/SafeDRL/save/terminal_states.p", "rb"))
remainings = pickle.load(open("/home/edoardo/Development/SafeDRL/save/remainings.p", "rb"))
t = pickle.load(open("/home/edoardo/Development/SafeDRL/save/t.p", "rb"))
storage.load_state("/home/edoardo/Development/SafeDRL/save")
# %%
boundaries = [[999, 0], [999, 0], [999, 0], [999, 0]]
for interval, action in union_states_total:
    for d in range(len(interval)):
        boundaries[d] = [min(boundaries[d][0], interval[d][0]), max(boundaries[d][0], interval[d][1])]
print(boundaries)
# %%
while True:
    if storage.needs_update:
        storage.recreate_prism()
    split_performed = False
    analysis_t = 0  # for now we analyse the layer t=0 after it gets splitted
    terminal_states_java = ListConverter().convert(terminal_states, gateway._gateway_client)
    solution_min = gateway.entry_point.check_state_list(terminal_states_java, True)
    solution_max = gateway.entry_point.check_state_list(terminal_states_java, False)
    t_ids = storage.get_t_layer(f"{analysis_t}.split")
    probabilities = []
    for id in t_ids:
        probabilities.append((solution_min[id], solution_max[id]))  # print(f"Interval {id} ({solution_min[id]},{solution_max[id]})")

    safe_threshold = 0.8
    to_analyse = []
    safe_count = 0
    unsafe_count = 0
    for i, interval_probability in enumerate(probabilities):
        if interval_probability[0] >= 1 - safe_threshold:
            print(f"Interval {t_ids[i]} ({interval_probability[0]},{interval_probability[1]}) is unsafe")
            unsafe_count += 1
        elif interval_probability[1] < 1 - safe_threshold:
            print(f"Interval {t_ids[i]} ({interval_probability[0]},{interval_probability[1]}) is safe")
            safe_count += 1
        else:
            print(f"Splitting interval {t_ids[i]} ({interval_probability[0]},{interval_probability[1]})")  # split
            split_performed = True
            interval_to_split = storage.dictionary_get(t_ids[i])
            dom1, dom2 = DomainExplorer.box_split_tuple(interval_to_split, rounding)
            # storage.purge(0, [t_ids[i]])
            storage.purge_branch(t_ids[i], 0)
            storage.store_successor(dom1, f"{analysis_t}.split", 0)
            storage.store_successor(dom2, f"{analysis_t}.split", 0)
            to_analyse.append(dom1)
            to_analyse.append(dom2)
    t = analysis_t + 1  # start inserting the new values after the current timestep
    print(f"Safe: {safe_count} Unsafe: {unsafe_count} To Analyse:{len(to_analyse)}")
    if len(to_analyse) != 0:
        for i in range(4):
            to_analyse, rtree = analysis_iteration(to_analyse, t, terminal_states, failed, n_workers, rtree, env, explorer, storage, [failed_area], union_states_total)
            t = t + 1
    if not split_performed:
        break  # %%

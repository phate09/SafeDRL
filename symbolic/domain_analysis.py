# %%
import itertools
import os
import pickle

import gym
import progressbar
import ray
from py4j.java_collections import ListConverter
from py4j.java_gateway import JavaGateway

from mosaic.utils import show_plot
from plnn.bab_explore import DomainExplorer
from prism.shared_rtree import SharedRtree
from prism.state_storage import StateStorage
from symbolic.unroll_methods import analysis_iteration, compute_remaining_intervals4_multi, abstract_step_store2
from verification_runs.domain_explorers_load import generatePendulumDomainExplorer

gym.logger.set_level(40)
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
local_mode = False
if not ray.is_initialized():
    ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
storage = StateStorage()
storage.reset()
rounding = 6
explorer, verification_model, env, current_interval, state_size, env_class = generatePendulumDomainExplorer(1e-1, rounding)
precision = 1e-6
print(f"Building the tree")
rtree = SharedRtree()
rtree.reset(state_size)
rtree.load_from_file("/home/edoardo/Development/SafeDRL/save/union_states_total.p", rounding)
union_states_total = rtree.tree_intervals()
print(f"Finished building the tree")
remainings = [current_interval]

# remainings = pickle.load(open("/home/edoardo/Development/SafeDRL/save/remainings.p", "rb"))
# remainings_overlaps = pickle.load(open("/home/edoardo/Development/SafeDRL/save/remainings_overlaps.p", "rb"))
# remainings_overlaps = remove_overlaps([(x, None) for x in remainings],rounding,n_workers,state_size)
# merged_intervals = merge_supremum(remainings,rounding)
# show_plot([x for x in union_states_total] + [(x[0], "Brown") for x in safe_states_merged] + [(x[0], "Purple") for x in unsafe_states_merged])
t = 0
# %%
# for i in range(3):
#     remainings = analysis_iteration(remainings, t, n_workers, rtree, env_class, explorer, verification_model, state_size, rounding,storage)
#     t = t + 1
#     boundaries = [[999, -999], [999, -999], [999, -999], [999, -999]]
#     for interval in remainings:
#         for d in range(len(interval)):
#             boundaries[d] = [min(boundaries[d][0], interval[d][0]), max(boundaries[d][0], interval[d][1])]
#     print(boundaries)  # rtree.save_to_file("/home/edoardo/Development/SafeDRL/save/union_states_total.p")
# # %%
# storage.save_state("/home/edoardo/Development/SafeDRL/save")
# rtree.save_to_file("/home/edoardo/Development/SafeDRL/save/union_states_total.p")
# pickle.dump(t, open("/home/edoardo/Development/SafeDRL/save/t.p", "wb+"))
# pickle.dump(remainings, open("/home/edoardo/Development/SafeDRL/save/remainings.p", "wb+"))
# print("Checkpoint Saved...")

# %%
remainings = pickle.load(open("/home/edoardo/Development/SafeDRL/save/remainings.p", "rb"))
t = pickle.load(open("/home/edoardo/Development/SafeDRL/save/t.p", "rb"))
storage.load_state("/home/edoardo/Development/SafeDRL/save")
# %%
# boundaries = [[999, 0], [999, 0], [999, 0], [999, 0]]
# for interval, action in union_states_total:
#     for d in range(len(interval)):
#         boundaries[d] = [min(boundaries[d][0], interval[d][0]), max(boundaries[d][0], interval[d][1])]
# print(boundaries)
t0 = [storage.dictionary[i] for i in storage.get_t_layer(0)]
t0split = [storage.dictionary[i] for i in storage.get_t_layer(f"{0}.split")]
show_plot(t0, t0split)
# %%
while True:
    mdp, gateway = storage.recreate_prism()
    split_performed = False
    analysis_t = 0  # for now we analyse the layer t=0 after it gets splitted
    terminal_states = storage.get_terminal_states_ids()
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
    intervals_safe = []
    intervals_unsafe = []
    intervals_split = []
    for i, interval_probability in enumerate(probabilities):
        if interval_probability[0] >= 1 - safe_threshold:
            # print(f"Interval {t_ids[i]} ({interval_probability[0]},{interval_probability[1]}) is unsafe")
            unsafe_count += 1
            intervals_safe.append(storage.dictionary[t_ids[i]])
        elif interval_probability[1] < 1 - safe_threshold:
            # print(f"Interval {t_ids[i]} ({interval_probability[0]},{interval_probability[1]}) is safe")
            safe_count += 1
            intervals_unsafe.append(storage.dictionary[t_ids[i]])
        else:
            print(f"Splitting interval {t_ids[i]} ({interval_probability[0]},{interval_probability[1]})")  # split
            intervals_split.append(storage.dictionary[t_ids[i]])
            split_performed = True
            interval_to_split = storage.dictionary_get(t_ids[i])
            dom1, dom2 = DomainExplorer.box_split_tuple(interval_to_split, rounding)
            # storage.purge(0, [t_ids[i]])
            removed_components = storage.purge_branch(t_ids[i], 0)  # todo by removing the interval here, it causes issues when looking for other intervals
            storage.t_dictionary[f"{analysis_t}.split"].remove(t_ids[i])  # remove the index from the t_dictionary
            for key in storage.t_dictionary.keys():
                for item in removed_components:
                    if item in storage.t_dictionary[key]:
                        storage.t_dictionary[key].remove(item)
            # print("Removed old values from storage")
            id1 = storage.store_successor(dom1, 0)
            id2 = storage.store_successor(dom2, 0)
            storage.assign_t(id1, f"{analysis_t}.split")
            storage.assign_t(id2, f"{analysis_t}.split")
            to_analyse.append(dom1)
            to_analyse.append(dom2)
    t = analysis_t + 1  # start inserting the new values after the current timestep
    print(f"Safe: {safe_count} Unsafe: {unsafe_count} To Analyse:{len(to_analyse)}")
    show_plot(intervals_safe, intervals_unsafe, intervals_split)
    # perform one iteration without splitting the interval because the interval is already being splitted
    remainings, intersected_intervals = compute_remaining_intervals4_multi([(x, True) for x in to_analyse], rtree.tree, rounding)
    if len(remainings) != 0:
        print("------------WARNING: at this stage remainings should be 0-----------")
    list_assigned_action = list(itertools.chain.from_iterable([x[1] for x in intersected_intervals]))
    next_states, terminal_states = abstract_step_store2(list_assigned_action, env_class, t + 1, n_workers,
                                                        rounding)  # performs a step in the environment with the assigned action and retrieve the result
    print(f"Sucessors : {len(next_states)} Terminals : {len(terminal_states)}")
    next_to_compute = []
    with progressbar.ProgressBar(prefix="Storing successors ", max_value=len(next_states), is_terminal=True, term_width=200) as bar:
        for interval, successors in next_states:
            parent_id = storage.store(interval[0])
            for successor1, successor2 in successors:
                id1, id2 = storage.store_sticky_successors(successor1, successor2, parent_id)
                storage.assign_t(id1, t + 1)
                storage.assign_t(id2, t + 1)
                next_to_compute.append(successor1)
                next_to_compute.append(successor2)
            bar.update(bar.value + 1)
    storage.mark_as_fail(
        [storage.store(terminal_state) for terminal_state in list(itertools.chain.from_iterable([interval_terminal_states for interval, interval_terminal_states in terminal_states]))])
    print(f"t:{t} Finished")
    t = t + 1
    if len(to_analyse) != 0:
        for i in range(1, 3):
            to_analyse = analysis_iteration(to_analyse, t, n_workers, rtree, env_class, explorer, verification_model, state_size, rounding, storage)
            t = t + 1
    if not split_performed:
        break

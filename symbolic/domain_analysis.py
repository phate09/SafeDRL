# %%
import itertools
import os
import pickle
import gym
import progressbar
import ray
from py4j.java_collections import ListConverter
from mosaic.utils import show_plot
from plnn.bab_explore import DomainExplorer
from prism.shared_rtree import SharedRtree
from prism.state_storage import StateStorage
from symbolic.unroll_methods import analysis_iteration, compute_remaining_intervals4_multi, abstract_step_store2, merge_supremum2, merge_supremum3, is_negligible, is_small, compute_boundaries
from verification_runs.domain_explorers_load import generatePendulumDomainExplorer, generateCartpoleDomainExplorer

gym.logger.set_level(40)
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
local_mode = False
if not ray.is_initialized():
    ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
storage = StateStorage()
storage.reset()
rounding = 2
precision = 10 ** (-rounding)
explorer, verification_model, env, current_interval, state_size, env_class = generatePendulumDomainExplorer(precision, rounding)
print(f"Building the tree")
rtree = SharedRtree()
rtree.reset(state_size)
rtree.load_from_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e1.p", rounding)
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
    rtree.save_to_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e2.p")
# %%
storage.save_state("/home/edoardo/Development/SafeDRL/save")
rtree.save_to_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e2.p")
pickle.dump(t, open("/home/edoardo/Development/SafeDRL/save/t.p", "wb+"))
pickle.dump(remainings, open("/home/edoardo/Development/SafeDRL/save/remainings.p", "wb+"))
print("Checkpoint Saved...")

# %%
remainings = pickle.load(open("/home/edoardo/Development/SafeDRL/save/remainings.p", "rb"))
t = pickle.load(open("/home/edoardo/Development/SafeDRL/save/t.p", "rb"))
storage.load_state("/home/edoardo/Development/SafeDRL/save")
rtree.load_from_file("/home/edoardo/Development/SafeDRL/save/union_states_total_e1.p", rounding)
# %%
# boundaries = [[999, 0], [999, 0], [999, 0], [999, 0]]
# for interval, action in union_states_total:
#     for d in range(len(interval)):
#         boundaries[d] = [min(boundaries[d][0], interval[d][0]), max(boundaries[d][0], interval[d][1])]
# print(boundaries)
# t0 = [storage.dictionary[i] for i in storage.get_t_layer(0)]
# t0split = [storage.dictionary[i] for i in storage.get_t_layer(f"{0}.split")]
# show_plot(t0, t0split)

# %%
iteration = 0
while True:
    mdp, gateway = storage.recreate_prism()
    split_performed = False
    analysis_t = 0  # for now we analyse the layer t=0 after it gets splitted
    terminal_states = storage.get_terminal_states_ids()
    terminal_states_java = ListConverter().convert(terminal_states, gateway._gateway_client)
    solution_min = gateway.entry_point.check_state_list(terminal_states_java, True)
    solution_max = gateway.entry_point.check_state_list(terminal_states_java, False)
    t_ids = list(storage.graph.successors(0))  # storage.get_t_layer(f"{analysis_t}.split")
    intervals_probabilities = []
    for id in t_ids:
        intervals_probabilities.append((id, storage.dictionary.get(id), (solution_min[id], solution_max[id])))
    del t_ids
    unsafe_threshold = 0.8
    to_analyse = []
    safe_count = 0
    unsafe_count = 0
    intervals_safe = []
    intervals_unsafe = []
    intervals_split = []
    intervals_split_ids = []
    for i, (id, interval, interval_probability) in enumerate(intervals_probabilities):
        if interval_probability[0] >= unsafe_threshold:  # high probability of encountering a terminal state
            unsafe_count += 1
            intervals_unsafe.append(interval)
        elif interval_probability[1] < 1 - unsafe_threshold:  # low probability of not encountering a terminal state
            safe_count += 1
            intervals_safe.append(interval)
        elif is_small(interval, precision, rounding):
            print(f"Interval {interval} ({interval_probability[0]},{interval_probability[1]}) is too small, considering it unsafe")
            unsafe_count += 1
            intervals_unsafe.append(interval)
        else:
            print(f"Splitting interval {interval} ({interval_probability[0]},{interval_probability[1]})")  # split
            split_performed = True
            intervals_split.append(interval)
            intervals_split_ids.append(id)
            storage.purge_branch(id, 0)
            dom1, dom2 = DomainExplorer.box_split_tuple(interval, rounding)
            id1 = storage.store_successor(dom1, 0)
            id2 = storage.store_successor(dom2, 0)
            storage.assign_t(id1, f"{analysis_t}.split")
            storage.assign_t(id2, f"{analysis_t}.split")
            to_analyse.append(dom1)
            to_analyse.append(dom2)
    # with progressbar.ProgressBar(prefix="Removing old instances ", max_value=len(intervals_split_ids), is_terminal=True, term_width=200) as bar:
    #     for i, interval_split_id in enumerate(intervals_split_ids):
    #         removed_components = storage.purge_branch(interval_split_id, 0)
    #         # storage.t_dictionary[f"{analysis_t}.split"].remove(interval_split_id)  # remove the index from the t_dictionary
    #         # for key in storage.t_dictionary.keys():
    #         #     for item in removed_components:
    #         #         if item in storage.t_dictionary[key]:
    #         #             storage.t_dictionary[key].remove(item)
    #         bar.update(i)
    t = analysis_t + 1  # start inserting the new values after the current timestep
    print(f"Safe: {safe_count} Unsafe: {unsafe_count} To Analyse:{len(to_analyse)}")
    fig = show_plot(intervals_safe, intervals_unsafe, to_analyse)
    fig.write_html(f"/home/edoardo/Development/SafeDRL/save/fig_{iteration}.html")
    # perform one iteration without splitting the interval because the interval is already being splitted
    remainings, intersected_intervals = compute_remaining_intervals4_multi([(x, None) for x in to_analyse], rtree.tree, rounding)
    assert len(remainings) == 0, "------------WARNING: at this stage remainings should be 0-----------"
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
        for i in range(1, 4):
            to_analyse = analysis_iteration(to_analyse, t, n_workers, rtree, env_class, explorer, verification_model, state_size, rounding, storage)
            t = t + 1
    iteration += 1
    if not split_performed:
        print(f"No more splits performed")
        break
# %% save intervals+ probabilities
pickle.dump(intervals_probabilities, open("/home/edoardo/Development/SafeDRL/save/interval_probability_pairs.p", "wb+"))

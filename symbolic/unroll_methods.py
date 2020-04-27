"""
Collection of methods to use in unroll_abstract_env
"""
import itertools
import math
import pickle
from collections import defaultdict
from contextlib import nullcontext
from itertools import cycle
from math import ceil
from typing import Tuple, List

import networkx as nx
import numpy as np
import progressbar
import ray
from contexttimer import Timer
from py4j.java_collections import ListConverter
from rtree import index
from sympy.combinatorics.graycode import GrayCode
from mosaic.utils import chunks, shrink, interval_contains, flatten_interval, bulk_load_rtree_helper, create_tree, show_plot, show_plot3d, round_tuples, round_tuple
from mosaic.workers.AbstractStepWorker import AbstractStepWorker
from plnn.bab_explore import DomainExplorer
from prism.shared_rtree import SharedRtree
from prism.state_storage import StateStorage
from utility.standard_progressbar import StandardProgressBar
from verification_runs.aggregate_abstract_domain import merge_simple, merge_simple_interval_only, filter_only_connected


def abstract_step_store2(abstract_states_normalised: List[Tuple[Tuple[Tuple[float, float]], bool]], env_class, n_workers: int, rounding: int):
    """
    Given some abstract states, compute the next abstract states taking the action passed as parameter
    :param env:
    :param abstract_states_normalised: the abstract states from which to start, list of tuples of intervals
    :return: the next abstract states after taking the action (array)
    """
    next_states = []
    terminal_states = []
    chunk_size = 1000
    n_chunks = ceil(len(abstract_states_normalised) / chunk_size)
    workers = cycle([AbstractStepWorker.remote(rounding, env_class) for _ in range(min(n_workers, n_chunks))])
    proc_ids = []
    with StandardProgressBar(prefix="Preparing AbstractStepWorkers ", max_value=n_chunks) as bar:
        for i, intervals in enumerate(chunks(abstract_states_normalised, chunk_size)):
            proc_ids.append(next(workers).work.remote(intervals))
            bar.update(i)
    with StandardProgressBar(prefix="Performing abstract step ", max_value=len(proc_ids)) as bar:
        while len(proc_ids) != 0:
            ready_ids, proc_ids = ray.wait(proc_ids, num_returns=min(10, len(proc_ids)), timeout=0.5)
            results: List[Tuple[List[Tuple[Tuple[Tuple[Tuple[float, float]], bool], List[Tuple[Tuple[float, float]]]]], List[
                Tuple[Tuple[Tuple[Tuple[float, float]], bool], List[Tuple[Tuple[float, float]]]]]]] = ray.get(ready_ids)
            bar.update(bar.value + len(results))
            for next_states_local, terminal_states_local in results:
                for next_state_many in next_states_local:
                    next_states.append(next_state_many)
                for terminal_states_many in terminal_states_local:
                    terminal_states.append(terminal_states_many)
    return sorted(next_states), terminal_states


def assign_action_to_blank_intervals(s_array: List[Tuple[Tuple[float, float]]], explorer, verification_model, n_workers: int, rounding: int) -> Tuple[
    List[Tuple[Tuple[Tuple[float, float]], bool]], List[Tuple[Tuple[float, float]]]]:
    """
    Given a list of intervals, calculate the intervals where the agent will take a given action
    :param n_workers: number of worker processes
    :param s_array: the list of intervals
    :return: safe intervals,unsafe intervals, ignore/irrelevant intervals
    """
    # total_area_before = sum([area_tuple(remaining) for remaining in s_array])
    # given the initial states calculate which intervals go left or right
    stats = explorer.explore(verification_model, s_array, n_workers, debug=True)
    print(f"#states: {stats['n_states']} [safe:{stats['safe_relative_percentage']:.3%}, unsafe:{stats['unsafe_relative_percentage']:.3%}, ignore:{stats['ignore_relative_percentage']:.3%}]")
    safe_next = [i.cpu().numpy() for i in explorer.safe_domains]
    unsafe_next = [i.cpu().numpy() for i in explorer.unsafe_domains]
    ignore_next = [i.cpu().numpy() for i in explorer.ignore_domains]
    safe_next = np.stack(safe_next).tolist() if len(safe_next) != 0 else []
    unsafe_next = np.stack(unsafe_next).tolist() if len(unsafe_next) != 0 else []
    ignore_next = np.stack(ignore_next).tolist() if len(ignore_next) != 0 else []
    t_states = ([(tuple([tuple(x) for x in k]), True) for k in safe_next] + [(tuple([tuple(x) for x in k]), False) for k in unsafe_next], ignore_next)  # round_tuples
    # total_area_after = sum([area_tuple(remaining) for remaining, action in t_states[0]])
    # assert math.isclose(total_area_before, total_area_after), f"The areas do not match: {total_area_before} vs {total_area_after}"
    return t_states


def discard_negligibles(intervals: List[Tuple[Tuple[float, float]]]) -> List[Tuple[Tuple[float, float]]]:
    """discards the intervals with area 0"""
    return [x for x in intervals if not is_negligible(x)]


def is_negligible(interval: Tuple[Tuple[float, float]]):
    sizes = [abs(interval[dimension][1] - interval[dimension][0]) for dimension in range(len(interval))]
    return any([math.isclose(x, 0) for x in sizes])


def is_small(interval: Tuple[Tuple[float, float]], min_size: float, rounding):
    sizes = [round(abs(interval[dimension][1] - interval[dimension][0]), rounding) for dimension in range(len(interval))]
    return max(sizes) <= min_size


def analysis_iteration(intervals: List[Tuple[Tuple[float, float]]], n_workers: int, rtree: SharedRtree, env, explorer, verification_model, state_size: int, rounding: int, storage: StateStorage,
                       allow_assign_action=True) -> List[Tuple[Tuple[float, float]]]:
    if len(intervals) == 0:
        return []
    intervals_sorted = [round_tuple(x, rounding) for x in sorted(intervals)]
    remainings = intervals_sorted
    # print(f"Started")
    while True:
        # assign a dummy action to intervals_sorted
        remainings, intersected_intervals = compute_remaining_intervals4_multi(intervals_sorted, rtree.tree)  # checks areas not covered by total intervals
        remainings = sorted(remainings)
        if len(remainings) != 0:
            if allow_assign_action:
                print(f"Found {len(remainings)} remaining intervals, updating the rtree to cover them")
                remainings_merged = merge_supremum3(remainings)  # no need to assign a dummy action
                assigned_intervals, ignore_intervals = assign_action_to_blank_intervals(remainings_merged, explorer, verification_model, n_workers, rounding)
                print(f"Adding {len(assigned_intervals)} states to the tree")
                union_states_total = rtree.tree_intervals()
                union_states_total.extend(assigned_intervals)
                merged1 = [(x, True) for x in merge_supremum3([x[0] for x in union_states_total if x[1] == True])]
                merged2 = [(x, False) for x in merge_supremum3([x[0] for x in union_states_total if x[1] == False])]
                union_states_total_merged = merged1 + merged2
                rtree.load(union_states_total_merged)
            else:
                raise Exception("Remainings is not 0 but allow_assign_action is False")
        else:  # if no more remainings exit
            break
    # show_plot(intersected_intervals, intervals_sorted)
    list_assigned_action = []
    with StandardProgressBar(prefix="Storing intervals with assigned actions ", max_value=len(intersected_intervals)) as bar:
        for interval_noaction, successors in intersected_intervals:
            if allow_assign_action:
                merged1 = [(x, True) for x in merge_supremum2([x[0] for x in successors if x[1] is True], show_bar=False)]
                merged2 = [(x, False) for x in merge_supremum2([x[0] for x in successors if x[1] is False], show_bar=False)]
                successors_merged: List[Tuple[Tuple[Tuple[float, float]], bool]] = merged1 + merged2
                list_assigned_action.extend(successors_merged)
                if len(successors)==1:
                    if successors[0][0] == interval_noaction:
                        print("let's see") #figure out a way to differentiate between split layer and action layer
                storage.store_successor_multi([x[0] for x in successors_merged], interval_noaction)
            else:
                list_assigned_action.extend(successors)
            bar.update(bar.value + 1)
    # list_assigned_action = list(itertools.chain.from_iterable([x[1] for x in intersected_intervals]))
    next_states, terminal_states = abstract_step_store2(list_assigned_action, env, n_workers, rounding)  # performs a step in the environment with the assigned action and retrieve the result
    terminal_states_dict = defaultdict(bool)
    terminal_states_list = list(itertools.chain.from_iterable([interval_terminal_states for interval, interval_terminal_states in terminal_states]))
    storage.mark_as_fail(terminal_states_list)
    for terminal in terminal_states_list:
        terminal_states_dict[terminal] = True
    next_to_compute = []
    n_successors = 0
    with StandardProgressBar(prefix="Storing successors ", max_value=len(next_states)) as bar:
        for interval, successors in next_states:
            for successor1, successor2 in successors:
                n_successors += 2
                storage.store_sticky_successors(successor1, successor2, interval[0])
                if not terminal_states_dict[successor1]:
                    next_to_compute.append(successor1)
                if not terminal_states_dict[successor2]:
                    next_to_compute.append(successor2)
            bar.update(bar.value + 1)
    # store terminal states
    print(f"Sucessors : {n_successors} Terminals : {len(terminal_states_list)} Next States :{len(next_to_compute)}")

    # print(f"Finished")
    return next_to_compute


@ray.remote
def compute_remaining_worker(current_intervals: List[Tuple[Tuple[float, float]]], relevant_intervals_multi: List[List[Tuple[Tuple[Tuple[float, float]], bool]]], rounding: int) -> Tuple[
    List[List[Tuple[Tuple[float, float]]]], List[List[Tuple[Tuple[Tuple[float, float]], bool]]]]:
    remaining_total: List[List[Tuple[Tuple[float, float]]]] = []
    intersection_total: List[List[Tuple[Tuple[Tuple[float, float]], bool]]] = []
    with Timer(factor=1000) as t1:
        for i, interval in enumerate(current_intervals):
            with Timer(factor=1000) as t:
                relevant_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = relevant_intervals_multi[i]
                print(f"Relevant_intervals:{len(relevant_intervals)}")
                remaining, intersection_safe, intersection_unsafe = compute_remaining_intervals3(interval, relevant_intervals, False)
                intersection_safe = [(x, True) for x in intersection_safe]
                intersection_unsafe = [(x, False) for x in intersection_unsafe]
                remaining_total.append(remaining)
                intersection_total.append(intersection_safe + intersection_unsafe)
                print(f"Timer1:{t.elapsed} ")
        print(f"Timer chunk:{t1.elapsed}")
        intersection_total = [merge_simple(x, rounding) if len(x) > 1 else x for x in intersection_total]  # merging
        remaining_total = [merge_simple_interval_only(x, rounding) if len(x) > 1 else x for x in remaining_total]
        print(f"Timer merge:{t1.elapsed}")
    return remaining_total, intersection_total  # , remaining_ids_total


def compute_remaining_intervals3(current_interval: Tuple[Tuple[float, float]], intervals_to_fill: List[Tuple[Tuple[Tuple[float, float]], bool]], debug=True):
    """
    Computes the intervals which are left blank from the subtraction of intervals_to_fill from current_interval
    :param debug:
    :param current_interval:
    :param intervals_to_fill:
    :return: the blank intervals and the union intervals
    """
    "Optimised version of compute_remaining_intervals"
    examine_intervals: List[Tuple[Tuple[float, float]]] = []
    remaining_intervals = [current_interval]
    union_safe_intervals = []  # this list will contains the union between intervals_to_fill and current_interval
    union_unsafe_intervals = []  # this list will contains the union between intervals_to_fill and current_interval
    dimensions = len(current_interval)
    if len(intervals_to_fill) == 0:
        return remaining_intervals, [], []
    if debug:
        bar = StandardProgressBar(prefix="Computing remaining intervals...", max_value=len(intervals_to_fill)).start()
    for i, (interval, action) in enumerate(intervals_to_fill):

        examine_intervals.extend(remaining_intervals)
        remaining_intervals = []
        while len(examine_intervals) != 0:
            examine_interval = examine_intervals.pop(0)
            if not is_negligible(examine_interval):
                state = shrink(interval, examine_interval)
                contains = interval_contains(state, examine_interval)  # check it is partially contained in every dimension
                if contains:
                    points_of_interest = []
                    for dimension in range(dimensions):
                        points_of_interest.append({examine_interval[dimension][0], examine_interval[dimension][1]})  # adds the upper and lower bounds from the starting interval as points of interest
                        points_of_interest[dimension].add(max(examine_interval[dimension][0], state[dimension][0]))  # lb
                        points_of_interest[dimension].add(min(examine_interval[dimension][1], state[dimension][1]))  # ub
                    points_of_interest = [sorted(list(points_of_interest[dimension])) for dimension in range(dimensions)]  # sorts the points
                    point_of_interest_indices = [list(range(len(points_of_interest[dimension]) - 1)) for dimension in
                                                 range(dimensions)]  # we subtract 1 because we want to index the intervals not the points
                    permutations_indices = list(itertools.product(*point_of_interest_indices))
                    permutations_of_interest = []
                    for j, permutation_idx in enumerate(permutations_indices):
                        permutations_of_interest.append(tuple(
                            [(list(points_of_interest[dimension])[permutation_idx[dimension]], list(points_of_interest[dimension])[permutation_idx[dimension] + 1]) for dimension in
                             range(dimensions)]))
                    for j, permutation in enumerate(permutations_of_interest):
                        if permutation != state:
                            if permutation != examine_interval:
                                examine_intervals.append(permutation)
                            else:
                                remaining_intervals.append(permutation)
                        else:
                            if action:
                                union_safe_intervals.append(permutation)
                            else:
                                union_unsafe_intervals.append(permutation)
                else:
                    remaining_intervals.append(examine_interval)
            else:
                print("Discarded negligible in compute_remaining_intervals3")
        if debug:
            bar.update(i)
    if debug:
        bar.finish()
    return remaining_intervals, union_safe_intervals, union_unsafe_intervals


def compute_remaining_intervals4_multi(current_intervals: List[Tuple[Tuple[float, float]]], tree: index.Index, debug=True) -> Tuple[
    List[Tuple[Tuple[float, float]]], List[Tuple[Tuple[Tuple[Tuple[float, float]], bool], List[Tuple[Tuple[Tuple[float, float]], bool]]]]]:
    intervals_with_relevants = []
    dict_intervals = defaultdict(list)
    for i, interval in enumerate(current_intervals):
        relevant_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = filter_relevant_intervals3(tree, interval)
        intervals_with_relevants.append((interval, relevant_intervals))
    remain_list: List[Tuple[Tuple[float, float]]] = []
    proc_ids = []
    chunk_size = 200
    for i, chunk in enumerate(chunks(intervals_with_relevants, chunk_size)):
        proc_ids.append(compute_remaining_intervals_remote.remote(chunk, False))  # if debug:  #     bar.update(i)
    with StandardProgressBar(prefix="Computing remaining intervals", max_value=len(proc_ids)) if debug else nullcontext() as bar:
        while len(proc_ids) != 0:
            ready_ids, proc_ids = ray.wait(proc_ids)
            results = ray.get(ready_ids[0])
            for result in results:
                if result is not None:
                    (remain, safe, unsafe), (previous_interval, action) = result
                    remain_list.extend(remain)
                    dict_intervals[(previous_interval, action)].extend([(x, True) for x in safe])
                    dict_intervals[(previous_interval, action)].extend([(x, False) for x in unsafe])
            if debug:
                bar.update(bar.value + 1)
    intersection_list: List[Tuple[Tuple[Tuple[Tuple[float, float]], bool], List[Tuple[Tuple[Tuple[float, float]], bool]]]] = []  # list with intervals and associated intervals with action assigned
    for key in dict_intervals.keys():
        if len(dict_intervals[key]) != 0:
            intersection_list.append((key, dict_intervals[key]))
    return remain_list, intersection_list


@ray.remote
def compute_remaining_intervals_remote(intervals_with_relevants: List[Tuple[Tuple[Tuple[float, float]], List[Tuple[Tuple[Tuple[float, float]], bool]]]], debug=True):
    return [(compute_remaining_intervals3(current_interval, intervals_to_fill, debug), current_interval) for current_interval, intervals_to_fill in intervals_with_relevants]


def filter_relevant_intervals3(tree, current_interval: Tuple[Tuple[float, float]]) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    """Filter the intervals relevant to the current_interval"""
    # current_interval = inflate(current_interval, rounding)
    results = list(tree.intersection(flatten_interval(current_interval), objects='raw'))
    total = []
    for result in results:  # turn the intersection in an intersection with intervals which are closed only on the left
        suitable = all([x[1] != y[0] and x[0] != y[1] if y[0] != y[1] else True for x, y in zip(result[0], current_interval)])
        if suitable:
            total.append(result)
    return sorted(total)


def filter_relevant_intervals_action(tree, current_interval: Tuple[Tuple[Tuple[float, float]], bool]) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    action = current_interval[1]
    relevants = filter_relevant_intervals3(tree, current_interval[0])
    relevants = [x for x in relevants if x[1] == action]  # filter only intervals with matching action
    return relevants


def merge_supremum2(starting_intervals: List[Tuple[Tuple[float, float]]], show_bar=True) -> List[Tuple[Tuple[float, float]]]:
    """merge all the intervals provided, assumes they all have the same action"""
    if len(starting_intervals) <= 1:
        return starting_intervals
    intervals: List[Tuple[Tuple[float, float]]] = [x for x in starting_intervals if not is_negligible(x)]  # remove size 0 intervals
    if len(intervals) <= 1:
        return intervals
    state_size = len(intervals[0])
    merged_list = []

    with StandardProgressBar(prefix="Merging the intervals ", max_value=len(starting_intervals)) if show_bar else nullcontext()  as bar:
        i = 0
        while True:
            if show_bar:
                bar.update(max(len(starting_intervals) - len(intervals), 0))
            tree = create_tree([(x, True) for x in intervals])
            # find leftmost interval
            boundaries = []
            for i in range(state_size):
                boundaries.append((float(tree.get_bounds()[i * 2]), float(tree.get_bounds()[i * 2 + 1])))
            boundaries = tuple(boundaries)
            a = GrayCode(state_size)
            codes = list(a.generate_gray())
            for c, code in enumerate(codes):
                new_boundaries = merge_iteration(boundaries, codes, c, tree, intervals)
                boundaries = new_boundaries
            # show_plot(intervals, [(boundaries, True)])
            new_group_tree = create_tree([(boundaries, True)])  # add dummy action
            remainings, intersection_intervals = compute_remaining_intervals4_multi(intervals, new_group_tree, debug=False)
            merged_list.append(boundaries)
            if len(remainings) == 0:
                break
            intervals = remainings
            i += 1  # bar.update(i)
    return merged_list


def merge_iteration(bounds: Tuple[Tuple[float, float]], codes, iteration_n, tree, intervals) -> Tuple[Tuple[float, float]]:
    flattened_bounds = []
    starting_coordinate = []
    dimensions = len(bounds)
    previous_codes = []
    for i in range(dimensions):
        previous_codes.append(codes[(iteration_n - i + len(codes)) % len(codes)])  # find all the previous codes in order to check which dimension doesn't change
    same_dimension = -1
    for d in range(dimensions):
        if all(x[d] == previous_codes[0][d] for x in previous_codes):
            same_dimension = d
            break
    if same_dimension == -1:
        raise Exception("No dimension is different from the previous iteration")

    for d in range(dimensions):
        if d == same_dimension:
            direction = int(previous_codes[0][d])  # either squash left (0) or right(1)
            flattened_bounds.append((float(bounds[d][direction]), float(bounds[d][direction])))
        else:
            flattened_bounds.append(bounds[d])  # starting_coordinate.append(float(bounds[d][int(previous_codes[len(previous_codes)-1][d])]))#len(previous_codes)-1

    relevant_intervals = filter_relevant_intervals3(tree, tuple(flattened_bounds))
    # filter_not in bounds
    filtered = []
    for x, action in relevant_intervals:
        suitable = all(x[d][1] > bounds[d][0] and x[d][0] < bounds[d][1] for d in range(len(x)))
        if suitable:
            filtered.append((x, action))
        else:
            pass
    for d in range(dimensions):

        direction = int(previous_codes[1][d])  # determine upper or lower bound
        if direction == 1:
            coordinate = float(min(max([x[0][d][direction] for x in filtered]), bounds[d][direction]))
        else:
            coordinate = float(max(min([x[0][d][direction] for x in filtered]), bounds[d][direction]))
        starting_coordinate.append(coordinate)
    starting_coordinate = tuple(starting_coordinate)
    connected_relevant = filter_only_connected(filtered, starting_coordinate)  # todo define strategy for "connected", what is the starting point?
    if len(connected_relevant) == 0:
        if len(filtered) == 0:
            return bounds  # nothing we can do at this iteration
        else:
            connected_relevant = filtered  # if connected relevant fails to find a relevant cluster than take any result
    new_bounds = []
    for d in range(dimensions):
        if d != same_dimension:
            ub = float(min(max([x[0][d][1] for x in connected_relevant]), bounds[d][1]))
            lb = float(max(min([x[0][d][0] for x in connected_relevant]), bounds[d][0]))
            new_bounds.append((lb, ub))  #
        else:
            new_bounds.append(bounds[d])
    return tuple(new_bounds)


def merge_supremum3(starting_intervals: List[Tuple[Tuple[float, float]]], positional_method=False) -> List[Tuple[Tuple[float, float]]]:
    if len(starting_intervals) <= 1:
        return starting_intervals
    dimensions = len(starting_intervals[0])
    # generate tree
    intervals_dummy_action = [(x, True) for x in starting_intervals]
    tree = create_tree(intervals_dummy_action)
    # find bounds
    boundaries = compute_boundaries(intervals_dummy_action)
    if positional_method:
        # split
        split_list = [boundaries]
        n_splits = 10
        for i in range(n_splits):
            domain = split_list.pop(0)
            splitted_domains = DomainExplorer.box_split_tuple(domain, 0)
            split_list.extend(splitted_domains)

        # find relevant intervals
        working_list = []
        for domain in split_list:
            relevant_list = list(tree.intersection(flatten_interval(domain), objects='raw'))
            local_working_list = []
            # resize intervals
            for relevant, action in relevant_list:
                resized = shrink(relevant, domain)
                local_working_list.append((resized, action))
            working_list.append(local_working_list)
    else:
        working_list = chunks(starting_intervals, 1000)
    # intervals = starting_intervals
    merged_list: List[Tuple[Tuple[float, float]]] = []
    merge_remote = ray.remote(merge_supremum2)
    proc_ids = []
    for i, intervals in enumerate(working_list):
        proc_ids.append(merge_remote.remote(intervals))
    with StandardProgressBar(prefix="Merging intervals", max_value=len(proc_ids)) as bar:
        while len(proc_ids) != 0:
            ready_ids, proc_ids = ray.wait(proc_ids, num_returns=min(len(proc_ids), 5), timeout=0.5)
            results = ray.get(ready_ids)
            for result in results:
                if result is not None:
                    merged_list.extend(result)
            bar.update(bar.value + len(ready_ids))
    new_merged_list = merge_supremum2(merged_list)
    # show_plot(merged_list, new_merged_list)
    return new_merged_list


def compute_boundaries(starting_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]):
    dimensions = len(starting_intervals[0][0])
    boundaries = [(float("inf"), float("-inf")) for _ in range(dimensions)]
    for interval, action in starting_intervals:
        for d in range(dimensions):
            boundaries[d] = (min(boundaries[d][0], interval[d][0]), max(boundaries[d][1], interval[d][1]))
    boundaries = tuple(boundaries)
    return boundaries


def compute_predecessors(storage, ids):
    result = []
    for id in ids:
        predecessors = storage.graph.predecessors(id)
        result.append((id, predecessors))
    return result


def probability_iteration(storage: StateStorage, rtree: SharedRtree, precision, rounding, env_class, n_workers, explorer, verification_model, state_size, horizon, safe_threshold=0.2,
                          unsafe_threshold=0.8, max_iteration=-1):
    iteration = 0
    while True:
        storage.recreate_prism()
        # get the furthest nodes that have a maximum probability less than safe_threshold
        candidates_ids = [(id, x.get('lb'), x.get('ub')) for id, x in storage.graph.nodes.data() if
                          (x.get('lb') is not None and x.get('lb') < unsafe_threshold and x.get('ub') > safe_threshold and not x.get('ignore') and not x.get('fail'))]
        # terminal_states_dict = storage.get_terminal_states_dict()
        path_length = nx.shortest_path_length(storage.graph, source=storage.root)
        candidate_length_dict = defaultdict(list)
        for id, lb, ub in candidates_ids:
            # if not terminal_states_dict[id]:  # ignore terminal states
            candidate_length = path_length[id]
            candidate_length_dict[candidate_length].append((id, lb, ub))
        max_length = 100
        for length in [x for x in candidate_length_dict.keys() if x % 2 == 1]:  # only odd numbered distances
            max_length = min(length, max_length)
        if max_length == 100 or (max_iteration != -1 and iteration >= max_iteration):
            break
        t_ids = [x[0] for x in candidate_length_dict[max_length]]
        split_performed, to_analyse = perform_split(t_ids, storage, safe_threshold, unsafe_threshold, precision, rounding)
        # fig = show_plot(intervals_safe, intervals_unsafe, to_analyse)
        # fig.write_html(f"{save_folder}fig_{iteration}.html")
        # perform one iteration without splitting the interval because the interval is already being splitted
        # to_analyse = analysis_iteration(to_analyse, n_workers, rtree, env_class, explorer, verification_model, state_size, rounding, storage, allow_assign_action=False)
        # remainings, intersected_intervals = compute_remaining_intervals4_multi(to_analyse, rtree.tree)
        # assert len(remainings) == 0, "------------WARNING: at this stage remainings should be 0-----------"
        # list_assigned_action = list(itertools.chain.from_iterable([x[1] for x in intersected_intervals]))
        # next_states, terminal_states = abstract_step_store2(list_assigned_action, env_class, n_workers, rounding)  # performs a step in the environment with the assigned action and retrieve the result
        # print(f"Sucessors : {len(next_states)} Terminals : {len(terminal_states)}")
        # terminal_states_dict = defaultdict(bool)
        # terminal_states_list = list(itertools.chain.from_iterable([interval_terminal_states for interval, interval_terminal_states in terminal_states]))
        # storage.mark_as_fail(terminal_states_list)
        # for terminal in terminal_states_list:
        #     terminal_states_dict[terminal] = True
        # next_to_compute = []
        # with StandardProgressBar(prefix="Storing successors ", max_value=len(next_states)) as bar:
        #     for interval, successors in next_states:
        #         for successor1, successor2 in successors:
        #             storage.store_sticky_successors(successor1, successor2, interval[0])
        #             if not terminal_states_dict[successor1]:
        #                 next_to_compute.append(successor1)
        #             if not terminal_states_dict[successor2]:
        #                 next_to_compute.append(successor2)
        #         bar.update(bar.value + 1)
        # # print(f"t:{t} Finished")
        # to_analyse = next_to_compute
        iterations_needed = horizon - max_length // 2
        for i in range(iterations_needed):
            allow_assign_action = i != 0
            to_analyse = analysis_iteration(to_analyse, n_workers, rtree, env_class, explorer, verification_model, state_size, rounding, storage, allow_assign_action=allow_assign_action)
        iteration += 1


def perform_split(ids_split: List[Tuple[Tuple[float, float]]], storage: StateStorage, safe_threshold: float, unsafe_threshold: float, precision: float, rounding: int):
    split_performed = False
    to_analyse = []
    safe_count = 0
    unsafe_count = 0
    intervals_safe = []
    intervals_unsafe = []
    for interval in ids_split:

        interval_probability = (storage.graph.nodes[interval]['lb'], storage.graph.nodes[interval]['ub'])
        if interval_probability[0] >= unsafe_threshold:  # high probability of encountering a terminal state
            unsafe_count += 1
            intervals_unsafe.append(interval)
        elif interval_probability[1] <= safe_threshold:  # low probability of not encountering a terminal state
            safe_count += 1
            intervals_safe.append(interval)
        elif is_small(interval, precision, rounding):
            if not storage.graph.nodes[interval].get('ignore'):
                print(f"Interval {interval} ({interval_probability[0]},{interval_probability[1]}) is too small, considering it unsafe")
                unsafe_count += 1
                intervals_unsafe.append(interval)
                storage.graph.nodes[interval]['ignore'] = True
        else:
            # print(f"Splitting interval {interval} ({interval_probability[0]},{interval_probability[1]})")  # split
            split_performed = True
            predecessors = list(storage.graph.predecessors(interval))
            dom1, dom2 = DomainExplorer.box_split_tuple(interval, rounding)
            for parent_id in predecessors:
                storage.store_successor_multi([dom1, dom2], parent_id)
            storage.graph.remove_node(interval)
            to_analyse.append(dom1)
            to_analyse.append(dom2)
    print(f"Safe: {safe_count} Unsafe: {unsafe_count} To Analyse:{len(to_analyse)}")
    return split_performed, to_analyse


def get_property_at_timestep(storage: StateStorage, t: int, property: str):
    path_length = nx.shortest_path_length(storage.graph, source=storage.root)
    candidate_length_dict = defaultdict(list)
    for id in path_length.keys():
        # if not terminal_states_dict[id]:  # ignore terminal states
        candidate_length = path_length[id]
        candidate_length_dict[candidate_length].append(id)
    list_to_show = []
    for x in candidate_length_dict[t]:
        attr = storage.graph.nodes[x]
        list_to_show.append((x, attr.get(property, 0)))
    return list_to_show

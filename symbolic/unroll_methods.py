"""
Collection of methods to use in unroll_abstract_env
"""
import itertools
import math
from collections import defaultdict
from contextlib import nullcontext
from itertools import cycle
from math import ceil
from typing import Tuple, List
import numpy as np
import progressbar
import ray
from contexttimer import Timer
from rtree import index
from sympy.combinatorics.graycode import GrayCode
from mosaic.utils import chunks, shrink, interval_contains, flatten_interval, bulk_load_rtree_helper, create_tree, show_plot
from mosaic.workers.AbstractStepWorker import AbstractStepWorker
from prism.shared_rtree import SharedRtree
from prism.state_storage import StateStorage
from verification_runs.aggregate_abstract_domain import merge_simple, merge_simple_interval_only, filter_only_connected


def abstract_step_store2(abstract_states_normalised: List[Tuple[Tuple[Tuple[float, float]], bool]], env_class, t: int, n_workers: int, rounding: int):
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
    workers = cycle([AbstractStepWorker.remote(t, rounding, env_class) for _ in range(min(n_workers, n_chunks))])
    proc_ids = []
    with progressbar.ProgressBar(prefix="Preparing AbstractStepWorkers ", max_value=n_chunks, is_terminal=True, term_width=200) as bar:
        for i, intervals in enumerate(chunks(abstract_states_normalised, chunk_size)):
            proc_ids.append(next(workers).work.remote(intervals))
            bar.update(i)
    with progressbar.ProgressBar(prefix="Performing abstract step ", max_value=len(proc_ids), is_terminal=True, term_width=200) as bar:
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


def analysis_iteration(intervals: List[Tuple[Tuple[float, float]]], t, n_workers: int, rtree: SharedRtree, env, explorer, verification_model, state_size: int, rounding: int, storage: StateStorage) -> \
        List[Tuple[Tuple[float, float]]]:
    intervals_sorted = sorted(intervals)
    remainings = intervals_sorted
    print(f"t:{t} Started")
    while True:
        # assign a dummy action to intervals_sorted
        remainings, intersected_intervals = compute_remaining_intervals4_multi([(x, True) for x in intervals_sorted], rtree.tree, rounding)  # checks areas not covered by total intervals

        remainings = sorted(remainings)
        if len(remainings) != 0:
            print(f"Found {len(remainings)} remaining intervals, updating the rtree to cover them")
            remainings_merged = merge_supremum2(remainings, rounding)  # no need to assign a dummy action
            # show_plot(remainings,remainings_merged)
            remainings_merged_noaction = [x[0] for x in remainings_merged]  # remove dummy action
            assigned_intervals, ignore_intervals = assign_action_to_blank_intervals(remainings_merged_noaction, explorer, verification_model, n_workers, rounding)
            # show_plot(remainings_merged,assigned_intervals)
            # assigned_intervals_no_overlaps = remove_overlaps(assigned_intervals, rounding, n_workers, state_size)
            print(f"Adding {len(assigned_intervals)} states to the tree")
            union_states_total = rtree.tree_intervals()
            union_states_total.extend(assigned_intervals)
            # union_states_total_merged = merge_with_condition(union_states_total, rounding, max_iter=100)
            merged1 = [(x[0], True) for x in merge_supremum2([x for x in union_states_total if x[1] == True], rounding)]
            merged2 = [(x[0], False) for x in merge_supremum2([x for x in union_states_total if x[1] == False], rounding)]
            union_states_total_merged = merged1 + merged2
            # show_plot([x for x in assigned_intervals] + [(x[0], "Brown") for x in merged1] + [(x[0], "Purple") for x in merged2])
            rtree.load(union_states_total_merged)
        else:  # if no more remainings exit
            break
    # show_plot(intersected_intervals, intervals_sorted)
    with progressbar.ProgressBar(prefix="Storing intervals with assigned actions ", max_value=len(intersected_intervals), is_terminal=True, term_width=200) as bar:
        for interval, successors in intersected_intervals:
            parent_id = storage.store(interval[0])
            ids = storage.store_successor_multi2([x[0] for x in successors], parent_id)
            storage.assign_t_multi(ids, f"{t}.split")
            bar.update(bar.value + 1)

    list_assigned_action = list(itertools.chain.from_iterable([x[1] for x in intersected_intervals]))
    next_states, terminal_states = abstract_step_store2(list_assigned_action, env, t + 1, n_workers, rounding)  # performs a step in the environment with the assigned action and retrieve the result
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
    return next_to_compute


def remove_overlaps(current_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int, n_workers: int, state_size: int) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    """
    Ensures there are no overlaps between the current intervals
    :param current_intervals:
    :param rounding:
    :return:
    """
    print("Removing overlaps")
    no_overlaps = compute_no_overlaps2(current_intervals, rounding, n_workers, state_size)
    print("Removed overlaps")

    return no_overlaps


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
        bar = progressbar.ProgressBar(prefix="Computing remaining intervals...", max_value=len(intervals_to_fill), is_terminal=True).start()
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


def compute_remaining_intervals4_multi(current_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], tree: index.Index, rounding: int, debug=True) -> Tuple[
    List[Tuple[Tuple[Tuple[float, float]], bool]], List[Tuple[Tuple[Tuple[Tuple[float, float]], bool], List[Tuple[Tuple[Tuple[float, float]], bool]]]]]:
    intervals_with_relevants = []
    dict_intervals = defaultdict(list)
    for i, interval in enumerate(current_intervals):
        relevant_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = filter_relevant_intervals3(tree, interval[0])
        intervals_with_relevants.append((interval, relevant_intervals))
    remain_list: List[Tuple[Tuple[Tuple[float, float]], bool]] = []

    old_size = len(current_intervals)
    proc_ids = []
    chunk_size = 200
    # with progressbar.ProgressBar(prefix="Starting workers", max_value=ceil(old_size / chunk_size), is_terminal=True, term_width=200) if debug else nullcontext() as bar:
    for i, chunk in enumerate(chunks(intervals_with_relevants, chunk_size)):
        proc_ids.append(compute_remaining_intervals_remote.remote(chunk, False))  # if debug:  #     bar.update(i)
    with progressbar.ProgressBar(prefix="Computing remaining intervals", max_value=len(proc_ids), is_terminal=True, term_width=200) if debug else nullcontext() as bar:
        while len(proc_ids) != 0:
            ready_ids, proc_ids = ray.wait(proc_ids)
            results = ray.get(ready_ids[0])
            for result in results:
                if result is not None:
                    (remain, safe, unsafe), (previous_interval, action) = result
                    remain_list.extend([(x, action) for x in remain])
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
def compute_remaining_intervals_remote(intervals_with_relevants: List[Tuple[Tuple[Tuple[Tuple[float, float]], bool], List[Tuple[Tuple[Tuple[float, float]], bool]]]], debug=True):
    return [(compute_remaining_intervals3(current_interval[0], intervals_to_fill, debug), current_interval) for current_interval, intervals_to_fill in intervals_with_relevants]


def compute_no_overlaps2(starting_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int, n_workers, state_size: int, max_iter: int = -1) -> List[
    Tuple[Tuple[Tuple[float, float]], bool]]:
    intervals = starting_intervals
    while True:
        handled_intervals = defaultdict(bool)
        p = index.Property(dimension=state_size)
        helper = bulk_load_rtree_helper(intervals)
        if len(intervals) != 0:
            tree = index.Index(helper, interleaved=False, properties=p, overwrite=True)
        else:
            tree = index.Index(interleaved=False, properties=p, overwrite=True)
        intervals_with_relevants = []
        found_list = [False] * len(intervals)
        for i, interval in enumerate(intervals):
            if not handled_intervals.get(interval):
                relevant_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = filter_relevant_intervals3(tree, interval[0])
                relevant_intervals = [x for x in relevant_intervals if x != interval]
                found = len(relevant_intervals) != 0
                found_list[i] = found
                relevant_intervals = [x for x in relevant_intervals if not handled_intervals.get(x, False)]
                for relevant in relevant_intervals:
                    handled_intervals[relevant] = True
                handled_intervals[interval] = True
                intervals_with_relevants.append((interval, relevant_intervals))
            else:
                intervals_with_relevants.append((interval, []))
        old_size = len(intervals)
        aggregated_list = []
        proc_ids = []
        chunk_size = 200
        with progressbar.ProgressBar(prefix="Starting workers", max_value=ceil(old_size / chunk_size), is_terminal=True, term_width=200) as bar:
            for i, chunk in enumerate(chunks([x for i, x in enumerate(intervals_with_relevants) if found_list[i]], chunk_size)):
                proc_ids.append(compute_remaining_intervals_remote.remote(chunk, False))
                bar.update(i)
        aggregated_list.extend([interval for i, interval in enumerate(intervals) if not found_list[i]])
        with progressbar.ProgressBar(prefix="Computing no overlap intervals", max_value=len(proc_ids), is_terminal=True, term_width=200) as bar:
            while len(proc_ids) != 0:
                ready_ids, proc_ids = ray.wait(proc_ids)
                results = ray.get(ready_ids[0])
                for result in results:
                    if result is not None:
                        (remain, _, _), action = result
                        aggregated_list.extend([(x, action) for x in remain])
                bar.update(bar.value + 1)
        n_founds = sum(x is True for x in found_list)
        new_size = len(aggregated_list)
        if n_founds != 0:
            print(f"Reduced overlaps to {n_founds / new_size:.2%}")
        else:
            print("Finished!")
        if not any(found_list):
            break
        else:
            intervals = aggregated_list
    return aggregated_list


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


def merge_supremum(starting_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    """merge all the intervals provided, assumes they all have the same action"""
    if len(starting_intervals) == 0:
        return starting_intervals
    intervals = starting_intervals
    state_size = len(intervals[0][0])
    merged_list = []
    with progressbar.ProgressBar(prefix="Merging the intervals ", max_value=len(starting_intervals), is_terminal=True, term_width=200) as bar:
        i = 0
        while True:
            bar.update(max(len(starting_intervals) - len(intervals), 0))
            tree = create_tree(intervals)
            # find leftmost interval
            left_boundary: float = tree.bounds[0]
            right_boundary: float = tree.bounds[1]
            bottom_boundary: float = tree.bounds[2]
            top_boundary: float = tree.bounds[3]
            items_left_boundary = filter_only_connected(filter_relevant_intervals3(tree, tuple([(left_boundary, left_boundary), (bottom_boundary, top_boundary)])))  # start from the left boundary
            top_boundary: float = min(top_boundary, max([x[0][1][1] for x in items_left_boundary]) if len(items_left_boundary) != 0 else float("inf"))
            bottom_boundary: float = max(bottom_boundary, min([x[0][1][0] for x in items_left_boundary]) if len(items_left_boundary) != 0 else float("-inf"))
            items_top_boundary = filter_only_connected(filter_relevant_intervals3(tree, tuple([(left_boundary, right_boundary), (top_boundary, top_boundary)])), (left_boundary, top_boundary))
            left_boundary: float = max(left_boundary, min([x[0][0][0] for x in items_top_boundary]) if len(items_top_boundary) != 0 else float("-inf"))
            right_boundary: float = min(right_boundary, max([x[0][0][1] for x in items_top_boundary]) if len(items_top_boundary) != 0 else float("inf"))
            items_bottom_boundary = filter_only_connected(filter_relevant_intervals3(tree, tuple([(left_boundary, right_boundary), (bottom_boundary, bottom_boundary)])),
                                                          (left_boundary, bottom_boundary))
            left_boundary: float = max(left_boundary, min([x[0][0][0] for x in items_bottom_boundary]) if len(items_bottom_boundary) != 0 else float("-inf"))
            right_boundary: float = min(right_boundary, max([x[0][0][1] for x in items_bottom_boundary]) if len(items_bottom_boundary) != 0 else float("inf"))
            items_right_boundary = filter_only_connected(filter_relevant_intervals3(tree, tuple([(right_boundary, right_boundary), (bottom_boundary, top_boundary)])),
                                                         (right_boundary, bottom_boundary))
            top_boundary: float = min(top_boundary, max([x[0][1][1] for x in items_right_boundary]) if len(items_right_boundary) != 0 else float("inf"))
            bottom_boundary: float = max(bottom_boundary, min([x[0][1][0] for x in items_right_boundary]) if len(items_right_boundary) != 0 else float("-inf"))
            new_group = (tuple([(left_boundary, right_boundary), (bottom_boundary, top_boundary)]), True)
            # show_plot(intervals,[(tuple([(left_boundary, right_boundary), (bottom_boundary, top_boundary)]), True)])
            new_group_tree = create_tree([new_group])
            remainings, intersection_intervals = compute_remaining_intervals4_multi(intervals, new_group_tree, rounding, debug=False)
            merged_list.append(new_group)
            if len(remainings) == 0:
                break
            intervals = remainings
            i += 1  # bar.update(i)
    return merged_list


def merge_supremum2(starting_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    """merge all the intervals provided, assumes they all have the same action"""
    if len(starting_intervals) == 0:
        return starting_intervals
    intervals = starting_intervals
    state_size = len(intervals[0][0])
    merged_list = []

    with progressbar.ProgressBar(prefix="Merging the intervals ", max_value=len(starting_intervals), is_terminal=True, term_width=200) as bar:
        i = 0
        while True:
            bar.update(max(len(starting_intervals) - len(intervals), 0))
            tree = create_tree(intervals)
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
            remainings, intersection_intervals = compute_remaining_intervals4_multi(intervals, new_group_tree, rounding, debug=False)
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
    for d in range(dimensions):

        direction = int(previous_codes[1][d])  # determine upper or lower bound
        if direction == 1:
            coordinate = float(min(max([x[0][d][direction] for x in relevant_intervals]), bounds[d][direction]))
        else:
            coordinate = float(max(min([x[0][d][direction] for x in relevant_intervals]), bounds[d][direction]))
        starting_coordinate.append(coordinate)
    starting_coordinate = tuple(starting_coordinate)
    connected_relevant = filter_only_connected(relevant_intervals, starting_coordinate)  # todo define strategy for "connected", what is the starting point?
    new_bounds = []
    for d in range(dimensions):
        # if d == same_dimension:
        ub = float(min(max([x[0][d][1] for x in connected_relevant]), bounds[d][1]))
        lb = float(max(min([x[0][d][0] for x in connected_relevant]), bounds[d][0]))
        new_bounds.append((lb, ub))
        # else:
        #     new_bounds.append(bounds[d])
    return tuple(new_bounds)

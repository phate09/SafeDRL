#!python
# cython: embedsignature=True, binding=True, profile=False
"""
Collection of methods to use in unroll_abstract_env
"""
import math
from collections import defaultdict
from contextlib import nullcontext
from itertools import cycle
from math import ceil
from typing import Tuple, List, Set

import networkx as nx
import numpy as np
import progressbar
import ray
import sympy
from rtree import index
from sympy.combinatorics.graycode import GrayCode
import torch
import mosaic.utils as utils
import prism.state_storage
import runnables.verification_runs.aggregate_abstract_domain
from mosaic.hyperrectangle import HyperRectangle, HyperRectangle_action
from mosaic.workers.AbstractStepWorker import AbstractStepWorker
from plnn.bab_explore import DomainExplorer
from plnn.verification_network import VerificationNetwork
from prism.shared_rtree import SharedRtree
from prism.state_storage import StateStorage
from symbolic.symbolic_interval import Interval_network, Symbolic_interval
from utility.standard_progressbar import StandardProgressBar


def abstract_step(abstract_states_normalised: List[HyperRectangle_action], env_class, n_workers: int, rounding: int, probabilistic=False):
    """
    Given some abstract states, compute the next abstract states taking the action passed as parameter
    :param env:
    :param abstract_states_normalised: the abstract states from which to start, list of tuples of intervals
    :return: the next abstract states after taking the action (array)
    """
    next_states = []
    terminal_states = defaultdict(bool)
    half_terminal_states = defaultdict(bool)
    chunk_size = 1000
    n_chunks = ceil(len(abstract_states_normalised) / chunk_size)
    workers = cycle([AbstractStepWorker.remote(rounding, env_class, probabilistic) for _ in range(min(n_workers, n_chunks))])
    proc_ids = []
    with StandardProgressBar(prefix="Preparing AbstractStepWorkers ", max_value=n_chunks) as bar:
        for i, intervals in enumerate(utils.chunks(abstract_states_normalised, chunk_size)):
            proc_ids.append(next(workers).work.remote(intervals))
            bar.update(i)
    with StandardProgressBar(prefix="Performing abstract step ", max_value=len(proc_ids)) as bar:
        while len(proc_ids) != 0:
            ready_ids, proc_ids = ray.wait(proc_ids, num_returns=min(10, len(proc_ids)), timeout=0.5)
            results = ray.get(ready_ids)  #: Tuple[List[Tuple[HyperRectangle_action, List[HyperRectangle]]], dict, dict]
            bar.update(bar.value + len(results))
            for next_states_local, half_terminal_states_local, terminal_states_local in results:
                for next_state_key in next_states_local:
                    next_states.append((next_state_key, next_states_local[next_state_key]))
                terminal_states.update(terminal_states_local)
                half_terminal_states.update(half_terminal_states_local)
    return next_states, half_terminal_states, terminal_states


def assign_action_to_blank_intervals(s_array: List[HyperRectangle], explorer, verification_model, n_workers: int, rounding: int) -> Tuple[List[HyperRectangle_action], List[HyperRectangle]]:
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
    t_states = []
    for k in safe_next:
        box = HyperRectangle_action.from_tuple((tuple([tuple(x) for x in k]), True))
        box = box.round(rounding)
        t_states.append(box)
    for k in unsafe_next:
        box = HyperRectangle_action.from_tuple((tuple([tuple(x) for x in k]), False))
        box = box.round(rounding)
        t_states.append(box)
    return t_states, ignore_next


def discard_negligibles(intervals: List[HyperRectangle]) -> List[HyperRectangle]:
    """discards the intervals with area 0"""
    return [x for x in intervals if not is_negligible(x)]


def is_negligible(interval: HyperRectangle):
    sizes = [abs(interval[dimension].width()) for dimension in range(len(interval.intervals))]
    return any([math.isclose(x, 0) for x in sizes])


def is_small(interval: HyperRectangle, min_size: float, rounding):
    sizes = [round(abs(interval[dimension].width()), rounding) for dimension in range(len(interval.intervals))]
    return max(sizes) <= min_size


# def remove_spurious_nodes(graph: nx.DiGraph):
#     candidates_ids = [id for id, x in graph.nodes.data() if (x.get('fail') is not None and x.get('fail'))]
#     for node_id in candidates_ids:
#         edges = list(graph.edges(node_id))
#         if len(edges) != 0:
#             print(f"removed edges from {node_id}")
#             graph.remove_edges_from(edges)


def analysis_iteration(intervals: List[HyperRectangle], n_workers: int, rtree: SharedRtree, env, explorer, verification_model, state_size: int, rounding: int, storage: StateStorage,
                       allow_assign_action=True, allow_merge=True):
    if len(intervals) == 0:
        return []
    intersected_intervals = check_tree_coverage(allow_assign_action, allow_merge, explorer, intervals, n_workers, rounding, rtree, verification_model)
    list_assigned_action = store_subregions(intersected_intervals, storage)
    compute_successors(env, list_assigned_action, n_workers, rounding, storage)


def get_interval_action_probability(intervals: List[HyperRectangle], action: int, verification_model: VerificationNetwork):
    interval_to_numpy = np.stack([interval.to_numpy() for interval in intervals])
    sequential_nn = verification_model.base_network
    ix2 = Symbolic_interval(lower=torch.tensor(interval_to_numpy[:,0]), upper=torch.tensor(interval_to_numpy[:,1]))
    inet = Interval_network(sequential_nn.double(), None)
    result_interval = inet(ix2)
    upper_bound = result_interval.u
    lower_bound = result_interval.l
    return upper_bound, lower_bound


def exploration_PPO(intervals: List[HyperRectangle], n_workers: int, rtree: SharedRtree, env, explorer, verification_model, state_size: int, rounding: int, storage: StateStorage,
                    allow_assign_action=True, allow_merge=True):
    if len(intervals) == 0:
        return []
    # todo get interval probabilities of choosing actions
    # intersected_intervals = check_tree_coverage(allow_assign_action, allow_merge, explorer, intervals, n_workers, rounding, rtree, verification_model)
    # list_assigned_action = store_subregions(intersected_intervals, storage)
    actions = [0, 1]
    upper_bound,lower_bound =get_interval_action_probability(intervals, 0, verification_model)
    for action in actions:
        list_assigned_action = [x.assign(action) for x in intervals]
        parent_successor_intervals = [(x.assign(None), x.assign(action), {"p_ub": upper_bound[i][action].item(), "p_lb": lower_bound[i][action].item()}) for i,x in enumerate(intervals)]
        storage.store_successor_prob(parent_successor_intervals)
        # compute_successors(env, list_assigned_action, n_workers, rounding, storage, probabilistic=True)
        compute_successors_PPO(env, list_assigned_action, n_workers, rounding, storage)


def compute_successors_PPO(env, list_assigned_action, n_workers, rounding, storage):
    next_states, half_terminal_states_dict, terminal_states_dict = abstract_step(list_assigned_action, env, n_workers, rounding, probabilistic=True)
    next_states_prob = [(x, y[0].assign(None), {"p_ub": 1.0, "p_lb": 1.0}) for x, y in next_states]  # todo assign proper probabilities
    terminal_states_list = []
    half_terminal_states_list = []
    for key in terminal_states_dict:
        if terminal_states_dict[key]:
            terminal_states_list.append(key.assign(None))
    for key in half_terminal_states_dict:
        if half_terminal_states_dict[key]:
            half_terminal_states_list.append(key.assign(None))
    storage.mark_as_fail(terminal_states_list)
    storage.mark_as_half_fail(half_terminal_states_list)
    storage.store_successor_prob(next_states_prob)


def store_subregions(intersected_intervals, storage):
    list_assigned_action = []
    with StandardProgressBar(prefix="Storing intervals with assigned actions ", max_value=len(intersected_intervals)) as bar:
        for interval_noaction, successors in intersected_intervals:
            list_assigned_action.extend(successors)
            parent = HyperRectangle_action.from_hyperrectangle(interval_noaction, None)
            storage.store_successor_multi([(parent, x) for x in successors])  # store also the action
            bar.update(bar.value + 1)
    return list_assigned_action


def check_tree_coverage(allow_assign_action, allow_merge, explorer, intervals, n_workers, rounding, rtree, verification_model):
    """Given a list of intervals, query the rtree for the actions taken by the agent
    If action has not yet been discovered for an interval, compute the action for it and store it
    """
    while True:
        remainings, intersected_intervals = compute_remaining_intervals4_multi(intervals, rtree.tree, rounding=rounding)  # checks areas not covered by total intervals
        # remainings = sorted(remainings)
        if len(remainings) != 0:
            if allow_assign_action:
                print(f"Found {len(remainings)} remaining intervals, updating the rtree to cover them")
                if allow_merge:
                    remainings_merged = merge4(remainings, rounding)  # no need to assign a dummy action
                else:
                    remainings_merged = remainings
                assigned_intervals, ignore_intervals = assign_action_to_blank_intervals(remainings_merged, explorer, verification_model, n_workers, rounding)
                print(f"Adding {len(assigned_intervals)} states to the tree")
                union_states_total = rtree.tree_intervals()
                union_states_total.extend(assigned_intervals)
                if allow_merge:
                    merged1 = [x.assign(True) for x in merge4([x for x in union_states_total if x.action == True], rounding)]
                    merged2 = [x.assign(False) for x in merge4([x for x in union_states_total if x.action == False], rounding)]
                    print("Merged")
                    union_states_total_merged = merged1 + merged2
                else:
                    union_states_total_merged = union_states_total
                rtree.load(union_states_total_merged)
            else:
                raise Exception("Remainings is not 0 but allow_assign_action is False")
        else:  # if no more remainings exit
            break
    # show_plot(intersected_intervals, intervals_sorted)
    if allow_merge:
        intersected_intervals = premerge(intersected_intervals, n_workers, rounding)
    return intersected_intervals


def premerge(intersected_intervals, n_workers, rounding: int, show_bar=True):
    proc_ids = []
    merged_list = [(interval_noaction, successors) for interval_noaction, successors in intersected_intervals if len(successors) <= 1]
    working_list = list(utils.chunks([(interval_noaction, successors) for interval_noaction, successors in intersected_intervals if len(successors) > 1], 200))
    with StandardProgressBar(prefix="Premerging intervals ", max_value=len(working_list)) if show_bar else nullcontext() as bar:
        while len(working_list) != 0 or len(proc_ids) != 0:
            while len(proc_ids) < n_workers and len(working_list) != 0:
                intervals = working_list.pop(0)
                proc_ids.append(merge_successors.remote(intervals, rounding))
            ready_ids, proc_ids = ray.wait(proc_ids, num_returns=len(proc_ids), timeout=0.5)
            results = ray.get(ready_ids)
            for result in results:
                if result is not None:
                    merged_list.extend(result)
            if show_bar:
                bar.update(bar.value + len(ready_ids))

    return merged_list


def merge_successors_ray(intersected_intervals, rounding: int):
    merged_list = []
    for interval_noaction, successors in intersected_intervals:
        if len(successors) != 1:
            merged1 = [x.assign(True) for x in merge4([x for x in successors if x.action is True], rounding)]
            merged2 = [x.assign(False) for x in merge4([x for x in successors if x.action is False], rounding)]
            successors_merged = merged1 + merged2  #: List[HyperRectangle_action]
            merged_list.append((interval_noaction, successors_merged))
        else:
            merged_list.append((interval_noaction, successors))
    return merged_list


merge_successors = ray.remote(merge_successors_ray)


def compute_successors(env_class, list_assigned_action: List[HyperRectangle_action], n_workers, rounding, storage: StateStorage, probabilistic=False):
    # performs a step in the environment with the assigned action and retrieve the result
    next_states, half_terminal_states_dict, terminal_states_dict = abstract_step(list_assigned_action, env_class, n_workers, rounding, probabilistic)
    terminal_states_list = []
    half_terminal_states_list = []
    for key in terminal_states_dict:
        if terminal_states_dict[key]:
            terminal_states_list.append(key.assign(None))
    for key in half_terminal_states_dict:
        if half_terminal_states_dict[key]:
            half_terminal_states_list.append(key.assign(None))
    storage.mark_as_fail(terminal_states_list)
    storage.mark_as_half_fail(half_terminal_states_list)
    next_to_compute = []
    n_successors = 0
    with StandardProgressBar(prefix="Storing successors ", max_value=len(next_states)) as bar:
        for interval, successors in next_states:
            for successor1, successor2 in successors:
                n_successors += 2
                storage.store_sticky_successors(successor1.assign(None), successor2.assign(None), interval)  # interval with action
                if not terminal_states_dict[successor1] and not half_terminal_states_dict[successor1]:
                    next_to_compute.append(successor1.assign(None))
                if not terminal_states_dict[successor2] and not half_terminal_states_dict[successor2]:
                    next_to_compute.append(successor2.assign(None))
            bar.update(bar.value + 1)
    # store terminal states
    print(f"Sucessors : {n_successors} Terminals : {len(terminal_states_list)} Half Terminals:{len(half_terminal_states_list)} Next States :{len(next_to_compute)}")  # return next_to_compute


def compute_remaining_intervals3(current_interval: HyperRectangle, intervals_to_fill: List[HyperRectangle_action], debug=True) -> Tuple[List[HyperRectangle], List[HyperRectangle_action]]:
    """
    Computes the intervals which are left blank from the subtraction of intervals_to_fill from current_interval
    :param debug:
    :param current_interval:
    :param intervals_to_fill:
    :return: the blank intervals and the union intervals
    """
    "Optimised version of compute_remaining_intervals"

    examine_intervals = set()  #: Set[HyperRectangle]
    remaining_intervals = [current_interval]
    union_intervals = []  # this list will contains the union between intervals_to_fill and current_interval
    # union_unsafe_intervals = []  # this list will contains the union between intervals_to_fill and current_interval
    dimensions = current_interval.dimension()
    if len(intervals_to_fill) == 0:
        return remaining_intervals, []
    if debug:
        bar = StandardProgressBar(prefix="Computing remaining intervals...", max_value=len(intervals_to_fill)).start()
    for i, interval_action in enumerate(intervals_to_fill):
        examine_intervals = examine_intervals.union(remaining_intervals)
        remaining_intervals = []
        while len(examine_intervals) != 0:
            examine_interval = examine_intervals.pop()  #: HyperRectangle
            intersection = examine_interval.intersect(interval_action)
            if not intersection.empty():
                set_minus = examine_interval.setminus(intersection)
                examine_intervals = examine_intervals.union(set_minus)
                assert math.isclose(examine_interval.size() - intersection.size(), sum([x.size() for x in set_minus]))
                union_intervals.append(intersection.assign(interval_action.action))
            else:
                remaining_intervals.append(examine_interval)
        if debug:
            bar.update(i)
    if debug:
        bar.finish()
    return remaining_intervals, union_intervals


def compute_remaining_intervals4_multi(current_intervals: List[HyperRectangle], tree: index.Index, rounding: int, debug=True) -> Tuple[
    List[HyperRectangle], List[Tuple[HyperRectangle, List[HyperRectangle]]]]:
    intervals_with_relevants = []  #: List[Tuple[HyperRectangle, HyperRectangle_action]]
    for i, interval in enumerate(current_intervals):
        relevant_intervals = filter_relevant_intervals3(tree, interval)
        intervals_with_relevants.append((interval, relevant_intervals))
    remain_list = []  #: List[HyperRectangle]
    proc_ids = []
    chunk_size = 200
    intersection_list = []  # list with intervals and associated intervals with action assigned: List[Tuple[HyperRectangle, List[HyperRectangle]]]
    for i, chunk in enumerate(utils.chunks(intervals_with_relevants, chunk_size)):
        proc_ids.append(compute_remaining_intervals_remote.remote(chunk, False))  # if debug:  #     bar.update(i)
    with StandardProgressBar(prefix="Computing remaining intervals ", max_value=len(proc_ids)) if debug else nullcontext() as bar:
        while len(proc_ids) != 0:
            ready_ids, proc_ids = ray.wait(proc_ids)
            results = ray.get(ready_ids[0])
            for result in results:
                if result is not None:
                    (remain, intersection), previous_interval = result
                    remain_list.extend(remain)
                    assigned = []  #: List[HyperRectangle]
                    assigned.extend(intersection)
                    intersection_list.append((previous_interval, assigned))
            if debug:
                bar.update(bar.value + 1)

    return remain_list, intersection_list


def compute_remaining_intervals_ray(intervals_with_relevants: List[Tuple[HyperRectangle, HyperRectangle_action]], debug=True):
    return [(compute_remaining_intervals3(current_interval, intervals_to_fill, debug), current_interval) for current_interval, intervals_to_fill in intervals_with_relevants]


compute_remaining_intervals_remote = ray.remote(compute_remaining_intervals_ray)


def filter_relevant_intervals3(tree, current_interval: HyperRectangle) -> List[HyperRectangle_action]:
    """Filter the intervals relevant to the current_interval"""
    # current_interval = inflate(current_interval, rounding)
    results = list(tree.intersection(current_interval.to_coordinates(), objects='raw'))  #: List[HyperRectangle_action]
    total = []  #: List[HyperRectangle_action]
    for result in results:  # turn the intersection in an intersection with intervals which are closed only on the left
        suitable = not result.intersect(current_interval).empty()  # all([x[1] != y[0] and x[0] != y[1] if y[0] != y[1] else True for x, y in zip(result[0], current_interval)])
        if suitable:
            total.append(result)
    return total


def merge_supremum2(starting_intervals: List[HyperRectangle], show_bar=True) -> List[HyperRectangle]:
    """merge all the intervals provided, assumes they all have the same action"""
    if len(starting_intervals) <= 1:
        return starting_intervals
    # intervals: List[HyperRectangle] = [x for x in starting_intervals if not is_negligible(x)]  # remove size 0 intervals
    intervals = starting_intervals
    if len(intervals) <= 1:
        return intervals
    state_size = len(intervals[0])
    merged_list = []

    with StandardProgressBar(prefix="Merging the intervals ", max_value=len(starting_intervals)) if show_bar else nullcontext()  as bar:
        i = 0
        while True:
            if show_bar:
                bar.update(max(len(starting_intervals) - len(intervals), 0))
            tree = utils.create_tree([(x, True) for x in intervals])
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
            if len(merged_list) != 0 and merged_list[len(merged_list) - 1] == boundaries:
                print("endless loop")
            # show_plot(intervals, [(boundaries, True)])
            # new_group_tree = utils.create_tree([(boundaries, True)])  # add dummy action
            # remainings, _ = compute_remaining_intervals4_multi(intervals, new_group_tree, debug=False)
            remainings = []
            for interval in intervals:
                remaining, _, _ = compute_remaining_intervals3(interval, [(boundaries, True)], debug=False)
                remainings.extend(remaining)

            merged_list.append(boundaries)
            if len(remainings) == 0:
                break
            intervals = remainings
            i += 1  # bar.update(i)
    return merged_list


def merge_iteration(bounds: HyperRectangle, codes, iteration_n, tree, intervals) -> HyperRectangle:
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
    # filtered = relevant_intervals
    # filter_not in bounds
    filtered = []
    for x, action in relevant_intervals:
        suitable = True  # all(x[d][1] > bounds[d][0] and x[d][0] < bounds[d][1] for d in range(len(x)))
        for d in range(len(x)):
            accept = x[d][1] == x[d][0] or (x[d][1] > bounds[d][0] and x[d][0] < bounds[d][1])
            if not accept:
                suitable = False
                break
        if suitable:
            filtered.append((x, action))
        else:
            pass
    if len(filtered) == 0:
        relevant_intervals = filter_relevant_intervals3(tree, tuple(flattened_bounds))
        return bounds
    for d in range(dimensions):

        direction = int(previous_codes[1][d])  # determine upper or lower bound
        if direction == 1:
            coordinate = float(min(max([x[0][d][direction] for x in filtered]), bounds[d][direction]))
        else:
            coordinate = float(max(min([x[0][d][direction] for x in filtered]), bounds[d][direction]))
        starting_coordinate.append(coordinate)
    starting_coordinate = tuple(starting_coordinate)
    connected_relevant = runnables.verification_runs.aggregate_abstract_domain.filter_only_connected(filtered, starting_coordinate)  # todo define strategy for "connected", what is the starting point?
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


def merge_supremum2_ray(starting_intervals: List[HyperRectangle], show_bar=True) -> List[HyperRectangle]:
    return merge_supremum2(starting_intervals, show_bar)


merge_supremum2_remote = ray.remote(merge_supremum2_ray)


def merge_supremum3(starting_intervals: List[HyperRectangle], n_workers: int, precision: int, positional_method=False, show_bar=True) -> List[HyperRectangle]:
    if len(starting_intervals) <= 1:
        return starting_intervals
    dimensions = len(starting_intervals[0])
    # generate tree
    intervals_dummy_action = [(x, True) for x in starting_intervals]
    tree = utils.create_tree(intervals_dummy_action)
    # find bounds
    boundaries = compute_boundaries(intervals_dummy_action)
    if positional_method:
        # split
        split_list = [boundaries]
        n_splits = 10
        for i in range(n_splits):
            domain = split_list.pop(0)
            splitted_domains = DomainExplorer.box_split_tuple(domain, precision)
            split_list.extend(splitted_domains)

        # find relevant intervals
        working_list = []
        for domain in split_list:
            relevant_list = list(tree.intersection(utils.flatten_interval(domain), objects='raw'))
            local_working_list = []
            # resize intervals
            for relevant, action in relevant_list:
                resized = utils.shrink(relevant, domain)
                local_working_list.append((resized, action))
            working_list.append(local_working_list)
    else:
        working_list = list(utils.chunks(starting_intervals, min(1000, max(int(len(starting_intervals) / n_workers), 1))))
    # intervals = starting_intervals
    merged_list = []  #: List[HyperRectangle]
    proc_ids = []
    with StandardProgressBar(prefix="Merging intervals", max_value=len(working_list)) if show_bar else nullcontext() as bar:
        while len(working_list) != 0 or len(proc_ids) != 0:
            while len(proc_ids) < n_workers and len(working_list) != 0:
                intervals = working_list.pop()
                proc_ids.append(merge_supremum2_remote.remote(intervals))
            ready_ids, proc_ids = ray.wait(proc_ids, num_returns=len(proc_ids), timeout=0.5)
            results = ray.get(ready_ids)
            for result in results:
                if result is not None:
                    merged_list.extend(result)
            if show_bar:
                bar.update(bar.value + len(ready_ids))
    new_merged_list = merge_supremum2(merged_list)
    # show_plot(merged_list, new_merged_list)
    return new_merged_list


def merge4(starting_intervals: List[HyperRectangle], precision: int) -> List[HyperRectangle]:
    if len(starting_intervals) != 0:
        rtree = SharedRtree()
        state_size = starting_intervals[0].dimension()
        rtree.reset(state_size)
        rtree.load(starting_intervals)
        merged_list = []
        window_tuple = []
        window = rtree.tree.get_bounds()
        while len(window) != 0:
            window_tuple.append((window.pop(0), window.pop(0)))
        window = HyperRectangle.from_tuple(window_tuple)
        # subwindows = DomainExplorer.box_split_tuple(window, precision)
        subwindows = [window]
        with progressbar.ProgressBar(prefix="Merging intervals", max_value=progressbar.UnknownLength) as bar:
            while len(subwindows) != 0:
                bar.update()
                sub = subwindows.pop()
                filtered = rtree.filter_relevant_intervals_multi([sub])[0]
                filtered_shrunk = [x.intersect(sub) for x in filtered]
                if len(filtered) != 0:
                    area_filtered = sum([x.size() for x in filtered_shrunk])
                    area_sub = sub.size()
                    # remainings = [(x, None) for x in unroll_methods.compute_remaining_intervals3(sub, filtered)]
                    if not (area_filtered < area_sub and not math.isclose(area_filtered, area_sub, abs_tol=1e-10)):  # check for empty areas
                        merged_list.append(sub)
                    else:
                        subwindows.extend(sub.split(precision))
        # safe = [x for x in merged_list if x[1] == True]
        # unsafe = [x for x in merged_list if x[1] == False]
        # utils.show_plot(safe, unsafe, merged_list)
        return merged_list
    else:
        return []


def compute_boundaries(starting_intervals: List[HyperRectangle_action]):
    dimensions = len(starting_intervals[0][0])
    boundaries = [(float("inf"), float("-inf")) for _ in range(dimensions)]
    for interval, action in starting_intervals:
        for d in range(dimensions):
            boundaries[d] = (min(boundaries[d][0], interval[d][0]), max(boundaries[d][1], interval[d][1]))
    boundaries = tuple(boundaries)
    return boundaries


def get_layers(graph: nx.DiGraph, root):
    candidates_ids = [(id, x.get('lb'), x.get('ub')) for id, x in graph.nodes.data()]
    path_length = nx.shortest_path_length(graph, source=root)
    candidate_length_dict = defaultdict(list)
    for id, lb, ub in candidates_ids:
        # if not terminal_states_dict[id]:  # ignore terminal states
        if path_length.get(id) is not None:
            candidate_length = path_length[id]
            candidate_length_dict[candidate_length].append((id, lb, ub))
    return candidate_length_dict


def probability_iteration(storage: StateStorage, rtree: SharedRtree, precision, rounding, env_class, n_workers, explorer, verification_model, state_size, horizon, safe_threshold=0.2,
                          unsafe_threshold=0.8, allow_assign_actions=False, allow_merge=True, allow_refine=True):
    iteration = 0
    storage.recreate_prism(horizon * 2)
    shortest_path = nx.shortest_path(storage.graph, source=storage.root)
    leaves = storage.get_leaves(shortest_path, unsafe_threshold, horizon * 2)
    leaves = [x for x in leaves if x[1] % 2 == 0]
    max_path_length = min([x[1] for x in leaves]) if len(leaves) > 0 else horizon * 2  # longest path to a leave, only even number layers
    # storage.remove_unreachable()
    # storage.plot_graph()
    if max_path_length >= horizon * 2:  # REFINE
        if allow_refine:
            print("Refine process")
            # find terminal states and split them
            # half_terminal = [((interval, action), attributes.get('lb'), attributes.get('ub')) for (interval, action), attributes in storage.graph.nodes.data() if attributes.get('half_fail')]

            # refine states which are undecided
            candidate_length_dict = defaultdict(list)
            # get the furthest nodes that have a maximum probability less than safe_threshold
            candidates_ids = [(interval, attributes.get('lb'), attributes.get('ub')) for interval, attributes in storage.graph.nodes.data() if (
                    attributes.get('lb') is not None and attributes.get('ub') > safe_threshold and not attributes.get('ignore') and not attributes.get('half_fail') and not attributes.get(
                'fail') and interval.action is not None and not is_small(interval, precision, rounding) and attributes.get('lb') < unsafe_threshold)]
            for id, lb, ub in candidates_ids:
                if shortest_path.get(id) is not None:
                    if lb < unsafe_threshold:
                        candidate_length = len(shortest_path[id]) - 1
                        candidate_length_dict[candidate_length].append((id, lb, ub))
                    else:
                        lower_bound_exceeded = True
            odd_layers = [x for x in candidate_length_dict.keys() if x % 2 == 1 and x < horizon * 2]
            if len(odd_layers) == 0:
                return False
            max_length = min(odd_layers)  # get only odd numbers
            print(f"Refining layer {max_length}")
            t_ids = [x[0] for x in candidate_length_dict[max_length]]
            split_performed, to_analyse = perform_split(t_ids, storage, safe_threshold, unsafe_threshold, precision, rounding)
            compute_successors(env_class, to_analyse, n_workers, rounding, storage)
        else:
            return False
    else:  # EXPLORE
        print(f"Exploring at timestep {max_path_length}")
        to_analyse = []
        for interval_action, length, lb, ub in leaves:
            if length == max_path_length:
                to_analyse.append(interval_action.remove_action())  # just interval no action
        allow_assign_action = allow_assign_actions or True  # for i in range(iterations_needed):
        # to_analyse_array = np.array(to_analyse)
        analysis_iteration(to_analyse, n_workers, rtree, env_class, explorer, verification_model, state_size, rounding, storage, allow_assign_action=allow_assign_action, allow_merge=allow_merge)
    iteration += 1
    return True


def probability_iteration_PPO(storage: StateStorage, rtree: SharedRtree, precision, rounding, env_class, n_workers, explorer, verification_model, state_size, horizon, safe_threshold=0.2,
                              unsafe_threshold=0.8, allow_assign_actions=False, allow_merge=True, allow_refine=True):
    iteration = 0
    storage.recreate_prism_PPO(horizon * 2)
    shortest_path = nx.shortest_path(storage.graph, source=storage.root)
    leaves = storage.get_leaves(shortest_path, unsafe_threshold, horizon * 2)
    leaves = [x for x in leaves if x[1] % 2 == 0]
    max_path_length = min([x[1] for x in leaves]) if len(leaves) > 0 else horizon * 2  # longest path to a leave, only even number layers
    # storage.remove_unreachable()
    # storage.plot_graph()
    if max_path_length >= horizon * 2:  # REFINE
        if allow_refine:
            print("Refine process")
            # find terminal states and split them
            # half_terminal = [((interval, action), attributes.get('lb'), attributes.get('ub')) for (interval, action), attributes in storage.graph.nodes.data() if attributes.get('half_fail')]

            # refine states which are undecided
            candidate_length_dict = defaultdict(list)
            # get the furthest nodes that have a maximum probability less than safe_threshold
            candidates_ids = [(interval, attributes.get('lb'), attributes.get('ub')) for interval, attributes in storage.graph.nodes.data() if (
                    attributes.get('lb') is not None and attributes.get('ub') > safe_threshold and not attributes.get('ignore') and not attributes.get('half_fail') and not attributes.get(
                'fail') and interval.action is not None and not is_small(interval, precision, rounding) and attributes.get('lb') < unsafe_threshold)]
            for id, lb, ub in candidates_ids:
                if shortest_path.get(id) is not None:
                    if lb < unsafe_threshold:
                        candidate_length = len(shortest_path[id]) - 1
                        candidate_length_dict[candidate_length].append((id, lb, ub))
                    else:
                        lower_bound_exceeded = True
            odd_layers = [x for x in candidate_length_dict.keys() if x % 2 == 1 and x < horizon * 2]
            if len(odd_layers) == 0:
                return False
            max_length = min(odd_layers)  # get only odd numbers
            print(f"Refining layer {max_length}")
            t_ids = [x[0] for x in candidate_length_dict[max_length]]
            split_performed, to_analyse = perform_split(t_ids, storage, safe_threshold, unsafe_threshold, precision, rounding)
            # compute_successors(env_class, to_analyse, n_workers, rounding, storage)
            compute_successors_PPO(env_class, to_analyse, n_workers, rounding, storage)
        else:
            return False
    else:
        # EXPLORE
        print(f"Exploring at timestep {max_path_length}")
        to_analyse = []
        for interval_action, length, lb, ub in leaves:
            if length == max_path_length:
                to_analyse.append(interval_action.remove_action())  # just interval no action
        allow_assign_action = allow_assign_actions or True  # for i in range(iterations_needed):
        exploration_PPO(to_analyse, n_workers, rtree, env_class, explorer, verification_model, state_size, rounding, storage, allow_assign_action=allow_assign_action, allow_merge=allow_merge)
    iteration += 1
    return True


def perform_split(ids_split: List[HyperRectangle_action], storage: StateStorage, safe_threshold: float, unsafe_threshold: float, precision: float, rounding: int):
    split_performed = False
    to_analyse = []
    safe_count = 0
    unsafe_count = 0
    intervals_safe = []
    intervals_unsafe = []
    for interval in ids_split:

        interval_probability = (storage.graph.nodes[interval]['lb'], storage.graph.nodes[interval]['ub'])
        # if interval_probability[0] >= unsafe_threshold:  # high probability of encountering a terminal state
        #     unsafe_count += 1
        #     intervals_unsafe.append(interval)
        if interval_probability[1] <= safe_threshold:  # low probability of not encountering a terminal state
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
            domains = interval.split(rounding)
            for parent_id in predecessors:
                eattr = storage.graph.get_edge_data(parent_id, interval)
                storage.store_successor_prob([(parent_id, domain, eattr) for domain in domains])
                storage.graph.remove_edge(parent_id, interval)
            # storage.graph.remove_node(interval)
            storage.graph.nodes[interval]['ignore'] = True
            to_analyse.extend(domains)
    print(f"Safe: {safe_count} Unsafe: {unsafe_count} To Analyse:{len(to_analyse)}")
    return split_performed, to_analyse


def get_property_at_timestep(storage: StateStorage, t: int, properties: List[str]):
    path_length = nx.single_source_dijkstra_path_length(storage.graph, source=storage.root)
    list_to_show = []
    for x, attr in storage.graph.nodes.data():
        path = path_length.get(x)
        if path is not None and path == t:
            single_result = [x]
            for property in properties:
                single_result.append(attr.get(property))
            list_to_show.append(tuple(single_result))
    return list_to_show


def get_n_states(storage: prism.state_storage.StateStorage, horizon: int):
    """
    Returns the number of states up to horizon timesteps
    :param storage:
    :param horizon:
    :return: a list containing the number of states in the graph
    """
    shortest_path_abstract = nx.shortest_path(storage.graph, source=storage.root)
    n_states = []
    for t in range(1, horizon + 1):
        leaves_abstract = [(interval, len(shortest_path_abstract[interval]) - 1, attributes.get('lb'), attributes.get('ub')) for interval, attributes in storage.graph.nodes.data() if
                           interval in shortest_path_abstract and (len(shortest_path_abstract[interval]) - 1) < t * 2 and (len(shortest_path_abstract[interval]) - 1) % 2 == 0]
        n_states.append(len(leaves_abstract))
    return n_states


def softmax_interval(intervals: List[Tuple]):
    '''
    softmax of a, calculate interval by putting ub of a and lb of everything else. reverese for lb
    :param intervals:
    :return:
    '''
    output_values = []
    variables = sympy.symbols(f'a0:{len(intervals)}')
    exp_sum = sum([sympy.exp(x) for x in variables])
    for i, interval in enumerate(intervals):
        softmax_func = sympy.exp(variables[i]) / exp_sum
        substitutions_ub = {variables[j]: x[1] if j == i else x[0] for j, x in enumerate(intervals)}
        substitutions_lb = {variables[j]: x[0] if j == i else x[1] for j, x in enumerate(intervals)}
        softmax_interval_var = (float(softmax_func.subs(substitutions_lb)), float(softmax_func.subs(substitutions_ub)))
        output_values.append(softmax_interval_var)
    return output_values

"""
Collection of methods to use in unroll_abstract_env
"""
import itertools
import math
import random
from itertools import cycle
from math import ceil
from typing import Tuple, List

import Pyro5.api
import numpy as np
import progressbar
import ray
from contexttimer import Timer
from rtree import index

from mosaic.utils import chunks, round_tuple, round_tuples, area_tuple, shrink, interval_contains, flatten_interval, show_plot
from mosaic.workers.AbstractStepWorker import AbstractStepWorker
from plnn.bab_explore import DomainExplorer
from prism.shared_dictionary import get_shared_dictionary
from prism.shared_rtree import SharedRtree, get_rtree
from prism.state_storage import StateStorage, get_storage
from symbolic.cartpole_abstract import CartPoleEnv_abstract
from verification_runs.aggregate_abstract_domain import merge_simple, merge_simple_interval_only, merge_with_condition


def abstract_step_store2(abstract_states_normalised: List[Tuple[Tuple[Tuple[float, float]], bool]], env_class, explorer: DomainExplorer, t: int, n_workers: int, rounding: int) -> Tuple[
    List[Tuple[Tuple[float, float]]], List[Tuple[Tuple[float, float]]]]:
    """
    Given some abstract states, compute the next abstract states taking the action passed as parameter
    :param explorer:
    :param env:
    :param abstract_states_normalised: the abstract states from which to start, list of tuples of intervals
    :return: the next abstract states after taking the action (array)
    """
    next_states: List[Tuple[Tuple[float, float]]] = []
    terminal_states: List[Tuple[Tuple[float, float]]] = []

    chunk_size = 1000
    n_chunks = ceil(len(abstract_states_normalised) / chunk_size)
    workers = cycle([AbstractStepWorker.remote(explorer, t, rounding, env_class) for _ in range(min(n_workers, n_chunks))])
    proc_ids = []
    with progressbar.ProgressBar(prefix="Preparing AbstractStepWorkers ", max_value=n_chunks, is_terminal=True, term_width=200) as bar:
        for i, intervals in enumerate(chunks(abstract_states_normalised, chunk_size)):
            proc_ids.append(next(workers).work.remote(intervals))
            bar.update(i)
    with progressbar.ProgressBar(prefix="Performing abstract step ", max_value=len(proc_ids), is_terminal=True, term_width=200) as bar:
        while len(proc_ids) != 0:
            ready_ids, proc_ids = ray.wait(proc_ids, num_returns=min(10, len(proc_ids)), timeout=0.5)
            results: Tuple[List[List[Tuple[Tuple[float, float]]]], List[List[Tuple[Tuple[float, float]]]]] = ray.get(ready_ids)
            bar.update(bar.value + len(results))
            for next_states_local, terminal_states_local in results:
                for next_state_many in next_states_local:
                    next_states.extend(next_state_many)
                for terminal_states_many in terminal_states_local:
                    terminal_states.extend(terminal_states_many)
    return sorted(next_states), terminal_states


def generate_points_in_intervals(total_states: np.ndarray, n_points=100) -> np.ndarray:
    """
    :param total_states: 3 dimensional array (n,dimension,interval)
    :return:
    """
    generated_points = []
    for i in range(n_points):
        group = i % total_states.shape[0]
        f = lambda x: [random.uniform(v[0], v[1]) for v in x]
        random_point = f(total_states[group])
        generated_points.append(random_point)
    return np.stack(generated_points)


def assign_action(random_points: np.ndarray, t_states):
    """
    Given some points and a list of domains divided in 3 categories (safe,unsafe,ignore) check which action to pick.
    It also check for duplicates throwing an assertion error
    :param random_points: collection of random points
    :param t_states: collection of states at timestep t
    :return:
    """
    assigned_action = [""] * len(random_points)
    for i in range(len(random_points)):
        point = random_points[i]
        for r in range(3):  # safe,unsafe,ignore
            for g in range(len(t_states[r])):  # group
                matches = 0
                interval = t_states[r][g]
                # if interval[0][0] <= point[0] <= interval[0][1]:
                #     if interval[1][0] <= point[1] <= interval[1][1]:
                #         if interval[2][0] <= point[2] <= interval[2][1]:
                #             if interval[3][0] <= point[3] <= interval[3][1]:
                #                 matches=4
                for d in range(len(interval)):  # dimension
                    if interval[d][0] <= point[d] <= interval[d][1]:
                        matches += 1
                if matches == 4:
                    if r == 0:
                        if assigned_action[i] != "" and assigned_action[i] != "safe":
                            print("something's wrong")
                        assigned_action[i] = "safe"
                    elif r == 1:
                        if assigned_action[i] != "" and assigned_action[i] != "unsafe":
                            print("something's wrong 2")
                        assigned_action[i] = "unsafe"
                    elif r == 2:
                        assigned_action[i] = "ignore"
    return assigned_action


def generate_points(t_states, t):
    """
    Generate random points and determines the action of the agent given a collection of abstract states and a time step
    :param t_states: list of list of arrays
    :param t: timestep
    :return: random points and corresponding actions
    """
    # Generate random points and determines the action of the agent
    random_points = generate_points_in_intervals(t_states[t][0], 500) if len(t_states[t][0]) > 0 else np.array([]).reshape((0, 4))  # safe
    random_points2 = generate_points_in_intervals(t_states[t][1], 500) if len(t_states[t][1]) > 0 else np.array([]).reshape((0, 4))  # unsafe
    random_points3 = generate_points_in_intervals(t_states[t][2], 500) if len(t_states[t][2]) > 0 else np.array([]).reshape((0, 4))  # unsafe
    random_points = np.concatenate((random_points, random_points2, random_points3))
    assigned_actions = assign_action(random_points, t_states[t])
    return random_points, assigned_actions


def calculate_area(state: np.ndarray):
    """Given an interval state calculate the area of the interval"""
    dom_sides: np.ndarray = np.abs(state[:, 0] - state[:, 1])
    dom_area = dom_sides.prod()
    return dom_area


def generate_middle_points(t_states, t):
    """
    Generate the middle points of each inteval and determines the action of the agent given a collection of abstract states and a time step
    :param t_states: list of list of arrays
    :param t: timestep
    :return: random points and corresponding actions
    """
    # Generate random points and determines the action of the agent
    random_points = []
    areas = []
    for s in t_states[t][0]:  # safe
        random_points.append(np.average(s, axis=1))
        areas.append(calculate_area(s))
    for s in t_states[t][1]:  # unsafe
        random_points.append(np.average(s, axis=1))
        areas.append(calculate_area(s))
    for s in t_states[t][2]:  # ignore
        random_points.append(np.average(s, axis=1))
        areas.append(calculate_area(s))
    random_points = np.stack(random_points)
    assigned_actions = assign_action(random_points, t_states[t])
    return random_points, areas, assigned_actions


def assign_action_to_blank_intervals(s_array: List[Tuple[Tuple[float, float]]], explorer, verification_model, n_workers: int, rounding: int) -> Tuple[
    List[Tuple[Tuple[Tuple[float, float]], bool]], List[Tuple[Tuple[float, float]]]]:
    """
    Given a list of intervals, calculate the intervals where the agent will take a given action
    :param n_workers: number of worker processes
    :param s_array: the list of intervals
    :return: safe intervals,unsafe intervals, ignore/irrelevant intervals
    """
    total_area_before = sum([area_tuple(remaining) for remaining in s_array])
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
    total_area_after = sum([area_tuple(remaining) for remaining, action in t_states[0]])
    assert math.isclose(total_area_before, total_area_after), f"The areas do not match: {total_area_before} vs {total_area_after}"
    return t_states


def discard_negligibles(intervals: List[Tuple[Tuple[float, float]]]) -> List[Tuple[Tuple[float, float]]]:
    """discards the intervals with area 0"""
    return [x for x in intervals if not is_negligible(x)]


def is_negligible(interval: Tuple[Tuple[float, float]]):
    sizes = [abs(interval[dimension][1] - interval[dimension][0]) for dimension in range(len(interval))]
    return any([math.isclose(x, 0) for x in sizes])


def list_t_layer(t: int, solution_min: List, solution_max: List) -> List[Tuple[float, float]]:
    with get_storage() as storage:
        t_ids = storage.get_t_layer(t)
        result = []
        for i in t_ids:
            result.append((solution_min[i], solution_max[i]))
        return result


def analysis_iteration(intervals: List[Tuple[Tuple[float, float]]], t, n_workers: int, rtree: SharedRtree, env, explorer, verification_model, state_size: int, rounding: int) -> List[
    Tuple[Tuple[float, float]]]:
    intervals_sorted = sorted(intervals)
    remainings = intervals_sorted
    print(f"t:{t} Started")
    while True:
        remainings, intersected_intervals = compute_remaining_intervals3_multi(intervals_sorted, t, n_workers, rounding)  # checks areas not covered by total intervals

        remainings = sorted(remainings)
        if len(remainings) != 0:
            print(f"Found {len(remainings)} remaining intervals, updating the rtree to cover them")
            assigned_intervals, ignore_intervals = assign_action_to_blank_intervals(remainings, explorer, verification_model, n_workers, rounding)
            assigned_intervals_no_overlaps = remove_overlaps(assigned_intervals, rounding, n_workers, state_size)
            print(f"Adding {len(assigned_intervals_no_overlaps)} states to the tree")
            union_states_total = rtree.tree_intervals()
            union_states_total.extend(assigned_intervals_no_overlaps)
            union_states_total_merged = merge_with_condition(union_states_total, rounding, max_iter=100)
            rtree.load(union_states_total_merged)
        else:  # if no more remainings exit
            break
    show_plot(intersected_intervals, intervals_sorted)
    next_states, terminal_states = abstract_step_store2(intersected_intervals, env, explorer, t + 1, n_workers,
                                                        rounding)  # performs a step in the environment with the assigned action and retrieve the result
    print(f"Sucessors : {len(next_states)} Terminals : {len(terminal_states)}")

    print(f"t:{t} Finished")
    return next_states


def compute_remaining_intervals3_multi(current_intervals: List[Tuple[Tuple[float, float]]], t: int, n_workers: int, rounding: int) -> Tuple[
    List[Tuple[Tuple[float, float]]], List[Tuple[Tuple[Tuple[float, float]], bool]]]:
    """
    Calculates the remaining areas that are not included in the intersection between current_intervals and intervals_to_fill
    :param current_intervals:
    :return: the blank intervals and the intersection intervals
    """
    # no_overlaps = remove_overlaps([(x, True) for x in current_intervals], rounding, n_workers)  # assign a dummy action
    # no_overlaps = [x[0] for x in no_overlaps]  # removes the action
    chunk_size = 1
    n_chunks = ceil(len(current_intervals) / chunk_size)
    # workers = cycle([RemainingWorker.remote(rounding) for _ in range(min(n_workers, n_chunks))])
    proc_ids = []
    tree = get_rtree()
    with Timer(factor=1000) as t:  # sort according to the number of relevant intervals so that the slowest operations are executed early
        relevant_intervals_multi: List[List[Tuple[Tuple[Tuple[float, float]], bool]]] = tree.filter_relevant_intervals_multi(current_intervals, rounding)
        intervals_and_relevant_intervals = list(zip(current_intervals, relevant_intervals_multi))
        # intervals_and_relevant_intervals = sorted(intervals_and_relevant_intervals, key=lambda x: len(x[1]), reverse=True)
        chunks_intervals_and_relevant_intervals = list(chunks(intervals_and_relevant_intervals, chunk_size))
        print(f"Prefilter: {round(t.elapsed)}")
    with progressbar.ProgressBar(prefix="Starting computer remaining workers ", max_value=n_chunks, is_terminal=True, term_width=200) as bar:
        for i, interval_and_relevant_intervals in enumerate(chunks_intervals_and_relevant_intervals):
            intervals, relevant_intervals = map(list, zip(*interval_and_relevant_intervals))
            relevant_intervals_merged = []
            for relevant_intervals_single in relevant_intervals:
                # if len(relevant_intervals_single) < 500:
                relevant_intervals_merged.append(
                    relevant_intervals_single)  # else:  # merge relevant_intervals with a high number of matches  #     list_tuple = merge_with_condition(relevant_intervals_single, rounding, max_iter=100, n_remaining_cutoff=400)  #     relevant_intervals_merged.append(list_tuple)
            proc_ids.append(compute_remaining_worker.remote(intervals, relevant_intervals_merged, rounding))
            bar.update(i)
    remainings = []
    intersection_states = []
    with progressbar.ProgressBar(prefix="Compute remaining intervals ", max_value=len(proc_ids), is_terminal=True, term_width=200) as bar:
        while len(proc_ids) != 0:
            ready_ids, proc_ids = ray.wait(proc_ids, num_returns=min(10, len(proc_ids)), timeout=0.5)
            results_multi: Tuple[List[List[Tuple[Tuple[float, float], bool]]], List[List[Tuple[Tuple[Tuple[float, float]], bool]]]] = ray.get(ready_ids)
            bar.update(bar.value + len(results_multi))
            for remaining_multi, intersection_state_multi in results_multi:
                for remaining in remaining_multi:
                    remainings.extend(remaining)
                for intersection_state in intersection_state_multi:  # todo handle storing of successors
                    intersection_states.extend(intersection_state)
    remainings_left = discard_negligibles(remainings)
    return remainings_left, intersection_states


def remove_overlaps(current_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int, n_workers: int, state_size: int) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    """
    Ensures there are no overlaps between the current intervals
    :param current_intervals:
    :param rounding:
    :return:
    """
    print("Removing overlaps")
    no_overlaps = compute_no_overlaps(current_intervals, rounding, n_workers, state_size)
    # no_overlaps = no_overlaps_tree.tree_intervals()
    # for no_overlap_interval in no_overlaps:  # test there are no overlaps
    #     if len(no_overlaps_tree.filter_relevant_intervals3(no_overlap_interval[0], rounding)) == 0:
    #         assert len(no_overlaps_tree.filter_relevant_intervals3(no_overlap_interval[0], rounding)) == 0
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


def compute_no_overlaps(intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int, n_workers, state_size: int, max_iter: int = -1) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    aggregated_list = intervals
    completed_iterations = 0
    # self_proxy = self
    # self._pyroDaemon.register(self_proxy)
    no_overlap_tree = get_rtree_temp()
    no_overlap_tree.reset(state_size)
    workers = cycle([NoOverlapWorker.remote(no_overlap_tree, rounding) for _ in range(n_workers)])
    with get_shared_dictionary() as shared_dict:
        while True:
            shared_dict.reset()  # reset the dictionary
            no_overlap_tree.load(aggregated_list)
            old_size = len(aggregated_list)
            proc_ids = []
            chunk_size = 200
            with progressbar.ProgressBar(prefix="Starting workers", max_value=ceil(old_size / chunk_size), is_terminal=True, term_width=200) as bar:
                for i, chunk in enumerate(chunks(aggregated_list, chunk_size)):
                    proc_ids.append(next(workers).no_overlap_worker.remote(chunk))
                    bar.update(i)
            aggregated_list = []
            found_list = []
            with progressbar.ProgressBar(prefix="Computing no overlap intervals", max_value=len(proc_ids), is_terminal=True, term_width=200) as bar:
                while len(proc_ids) != 0:
                    ready_ids, proc_ids = ray.wait(proc_ids)
                    result = ray.get(ready_ids[0])
                    if result is not None:
                        aggregated_list.extend(result[0])
                        found_list.extend(result[1])
                    bar.update(bar.value + 1)
            n_founds = sum(x is True for x in found_list)
            new_size = len(aggregated_list)
            if n_founds != 0:
                print(f"Reduced overlaps to {n_founds / new_size:.2%}")
            else:
                print("Finished!")
            if n_founds == 0:
                break
            completed_iterations += 1
            if completed_iterations >= max_iter != -1:
                break
    return aggregated_list


@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode="single")
class NoOverlapRtree:
    def __init__(self):
        self.union_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]] = []  # a list representing the content of the tree

    def reset(self, dimension):
        print("Resetting the tree")
        self.dimension = dimension
        self.p = index.Property(dimension=self.dimension)
        self.tree = index.Index(interleaved=False, properties=self.p, overwrite=True)
        self.union_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]] = []  # a list representing the content of the tree

    def load(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]):
        # with self.lock:
        print("Building the tree")
        helper = bulk_load_rtree_helper(intervals)
        self.tree.close()
        if len(intervals) != 0:
            self.tree = index.Index(helper, interleaved=False, properties=self.p, overwrite=True)
        else:
            self.tree = index.Index(interleaved=False, properties=self.p, overwrite=True)
        self.tree.flush()
        print("Finished building the tree")

    def add_single(self, interval: Tuple[Tuple[Tuple[float, float]], bool], rounding: int):
        id = len(self.union_states_total)
        # interval = (open_close_tuple(interval[0]), interval[1])
        relevant_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = self.filter_relevant_intervals3(interval[0], rounding)
        # relevant_intervals = [x for x in relevant_intervals if x != interval[0]]  # remove itself todo needed?
        remaining, intersection_safe, intersection_unsafe = compute_remaining_intervals3(interval[0], relevant_intervals, False)
        for remaining_interval in [(x, interval[1]) for x in remaining]:
            self.union_states_total.append(remaining_interval)
            coordinates = flatten_interval(remaining_interval[0])
            action = remaining_interval[1]
            self.tree.insert(id, coordinates, (remaining_interval[0], action))

    def tree_intervals(self) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
        return self.union_states_total

    def add_many(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int):
        """
        Store all the intervals in the tree with the same action
        :param intervals:
        :param action: the action to be assigned to all the intervals
        :return:
        """
        with progressbar.ProgressBar(prefix="Add many temp:", max_value=len(intervals), is_terminal=True, term_width=200) as bar:
            for i, interval in enumerate(intervals):
                self.add_single(interval, rounding)
                bar.update(i)

    def filter_relevant_intervals3(self, current_interval: Tuple[Tuple[float, float]], rounding: int) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
        """Filter the intervals relevant to the current_interval"""
        # current_interval = inflate(current_interval, rounding)
        results = list(self.tree.intersection(flatten_interval(current_interval), objects='raw'))
        total = []
        for result in results:  # turn the intersection in an intersection with intervals which are closed only on the left
            suitable = all([x[1] != y[0] and x[0] != y[1] for x, y in zip(result[0], current_interval)])
            if suitable:
                total.append(result)
        return sorted(total)


@ray.remote
class NoOverlapWorker:
    def __init__(self, tree: NoOverlapRtree, rounding):
        self.rounding = rounding
        self.tree: NoOverlapRtree = tree
        self.handled_intervals = get_shared_dictionary()

    def no_overlap_worker(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]) -> Tuple[List[Tuple[Tuple[Tuple[float, float]], bool]], List[bool]]:
        aggregated_list = []
        found_list = [False] * len(intervals)
        for i, interval in enumerate(intervals):
            handled = self.handled_intervals.get(interval, False)
            if not handled:
                relevant_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = self.tree.filter_relevant_intervals3(interval[0], self.rounding)
                relevant_intervals = [x for x in relevant_intervals if x != interval]
                found = len(relevant_intervals) != 0
                found_list[i] = found
                relevant_intervals = [x for x in relevant_intervals if not self.handled_intervals.get(x, False)]
                self.handled_intervals.set_multiple([interval] + relevant_intervals, True)  # mark the interval as handled
                remaining, intersection_safe, intersection_unsafe = compute_remaining_intervals3(interval[0], relevant_intervals, False)
                aggregated_list.extend([(x, interval[1]) for x in remaining])  # todo merge here?
            else:
                # already handled previously
                aggregated_list.append(interval)
        return aggregated_list, found_list


def get_rtree_temp() -> NoOverlapRtree:
    Pyro5.api.config.SERIALIZER = "marshal"
    Pyro5.api.config.SERVERTYPE = "multiplex"
    storage = Pyro5.api.Proxy("PYRONAME:prism.rtreetemp")
    return storage


def bulk_load_rtree_helper(data: List[Tuple[Tuple[Tuple[float, float]], bool]]):
    for i, obj in enumerate(data):
        interval = obj[0]
        yield (i, flatten_interval(interval), obj)

"""
Collection of methods to use in unroll_abstract_env
"""
import math
import random
from itertools import cycle
from math import ceil
from typing import Tuple, List

import numpy as np
import progressbar
import ray

from mosaic.utils import chunks, round_tuple, round_tuples, area_tuple
from mosaic.workers.AbstractStepWorker import AbstractStepWorker
from mosaic.workers.RemainingWorker import RemainingWorker
from plnn.bab_explore import DomainExplorer
from prism.shared_rtree import SharedRtree, get_rtree
from prism.no_overlap_rtree import get_rtree_temp
from prism.state_storage import StateStorage, get_storage
from symbolic.cartpole_abstract import CartPoleEnv_abstract
from verification_runs.cartpole_bab_load import generateCartpoleDomainExplorer


def abstract_step_store2(abstract_states_normalised: List[Tuple[Tuple[Tuple[float, float]], bool]], env: CartPoleEnv_abstract, explorer: DomainExplorer, t: int, n_workers: int, rounding: int) -> \
        Tuple[List[Tuple[Tuple[float, float]]], List[Tuple[Tuple[float, float]]]]:
    """
    Given some abstract states, compute the next abstract states taking the action passed as parameter
    :param explorer:
    :param env:
    :param abstract_states_normalised: the abstract states from which to start, list of tuples of intervals
    :return: the next abstract states after taking the action (array)
    """
    next_states = []
    terminal_states = []

    chunk_size = 1000
    n_chunks = ceil(len(abstract_states_normalised) / chunk_size)
    workers = cycle([AbstractStepWorker.remote(explorer, t, rounding) for _ in range(min(n_workers, n_chunks))])
    proc_ids = []
    with progressbar.ProgressBar(prefix="Preparing AbstractStepWorkers ", max_value=n_chunks, is_terminal=True, term_width=200) as bar:
        for i, intervals in enumerate(chunks(abstract_states_normalised, chunk_size)):
            proc_ids.append(next(workers).work.remote(intervals))
            bar.update(i)
    with progressbar.ProgressBar(prefix="Performing abstract step ", max_value=len(proc_ids), is_terminal=True, term_width=200) as bar:
        while len(proc_ids) != 0:
            ready_ids, proc_ids = ray.wait(proc_ids, num_returns=min(10, len(proc_ids)), timeout=0.5)
            results = ray.get(ready_ids)
            bar.update(bar.value + len(results))
            for next_states_local, terminal_states_local in results:
                next_states.extend(next_states_local)
                terminal_states.extend(terminal_states_local)
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


def assign_action_to_blank_intervals(s_array: List[Tuple[Tuple[float, float]]], n_workers: int, rounding: int) -> Tuple[
    List[Tuple[Tuple[Tuple[float, float]], bool]], List[Tuple[Tuple[float, float]]]]:
    """
    Given a list of intervals, calculate the intervals where the agent will take a given action
    :param n_workers: number of worker processes
    :param s_array: the list of intervals
    :return: safe intervals,unsafe intervals, ignore/irrelevant intervals
    """
    total_area_before = sum([area_tuple(remaining) for remaining in s_array])
    explorer, verification_model = generateCartpoleDomainExplorer(1e-1, rounding)
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


def analysis_iteration(intervals: List[Tuple[Tuple[float, float]]], t, n_workers: int, rtree: SharedRtree, env, explorer, rounding: int) -> List[Tuple[Tuple[float, float]]]:
    intervals_sorted = sorted(intervals)
    print(f"t:{t} Started")
    while True:
        remainings, intersected_intervals = compute_remaining_intervals3_multi(intervals_sorted, t, n_workers, rounding)  # checks areas not covered by total intervals
        remainings = sorted(remainings)
        if len(remainings) != 0:
            print(f"Found {len(remainings)} remaining intervals, updating the rtree to cover them")
            # get max of every dimension
            # boundaries = []
            # for d in range(len(remainings[0])):
            #     dimension_slice = [(x[d][0], x[d][1]) for x in remainings]
            #     maximum = float(max(x[1] for x in dimension_slice))
            #     minimum = float(min(x[0] for x in dimension_slice))
            #     boundaries.append((minimum, maximum))
            # boundaries = tuple(boundaries)
            # todo maybe compute_remaining on the boundary before
            # once you get the boundaries of the area to lookup we execute the algorithm to assign an action to this area(s)
            # remainings = [round_tuple(remaining, rounding) for remaining in remainings]
            # total_area_before = sum([area_tuple(remaining) for remaining in remainings])
            assigned_intervals, ignore_intervals = assign_action_to_blank_intervals(remainings, n_workers, rounding)
            # assigned_intervals = round_tuples(assigned_intervals)
            # total_area_after = sum([area_tuple(remaining) for remaining, action in assigned_intervals])
            # assert math.isclose(total_area_before, total_area_after), f"The areas do not match: {total_area_before} vs {total_area_after}"
            assigned_intervals_no_overlaps = remove_overlaps(assigned_intervals, rounding, n_workers)
            # total_area_after_no_overlaps = sum([area_tuple(remaining) for remaining, action in assigned_intervals_no_overlaps])
            # assert math.isclose(total_area_after, total_area_after_no_overlaps), f"The areas of no overlap do not match: {total_area_after} vs {total_area_after_no_overlaps}"
            print(f"Adding {len(assigned_intervals_no_overlaps)} states to the tree")
            rtree.add_many(assigned_intervals_no_overlaps, rounding)
            rtree.flush()
        else:  # if no more remainings exit
            break
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
    chunk_size = 100
    n_chunks = ceil(len(current_intervals) / chunk_size)
    workers = cycle([RemainingWorker.remote(t, rounding, get_rtree()) for _ in range(min(n_workers, n_chunks))])
    proc_ids = []
    with progressbar.ProgressBar(prefix="Starting computer remaining workers ", max_value=n_chunks, is_terminal=True, term_width=200) as bar:
        for i, intervals in enumerate(chunks(current_intervals, chunk_size)):
            proc_ids.append(next(workers).compute_remaining_worker.remote(intervals))
            bar.update(i)
    remainings = []
    intersection_states = []
    with progressbar.ProgressBar(prefix="Compute remaining intervals ", max_value=len(proc_ids), is_terminal=True, term_width=200) as bar:
        while len(proc_ids) != 0:
            ready_ids, proc_ids = ray.wait(proc_ids, num_returns=min(10, len(proc_ids)), timeout=0.5)
            results = ray.get(ready_ids)
            bar.update(bar.value + len(results))
            for remaining, intersection_state in results:
                remainings.extend(remaining)
                intersection_states.extend(intersection_state)
    remainings_left = discard_negligibles(remainings)
    return remainings_left, intersection_states


def remove_overlaps(current_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int, n_workers: int) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    """
    Ensures there are no overlaps between the current intervals
    :param current_intervals:
    :param rounding:
    :return:
    """
    print("Removing overlaps")
    no_overlaps_tree = get_rtree_temp()
    # no_overlaps_tree.add_many(current_intervals, rounding)
    no_overlaps = no_overlaps_tree.compute_no_overlaps(current_intervals, rounding, n_workers)
    # no_overlaps = no_overlaps_tree.tree_intervals()
    # for no_overlap_interval in no_overlaps:  # test there are no overlaps
    #     if len(no_overlaps_tree.filter_relevant_intervals3(no_overlap_interval[0], rounding)) == 0:
    #         assert len(no_overlaps_tree.filter_relevant_intervals3(no_overlap_interval[0], rounding)) == 0
    print("Removed overlaps")

    return no_overlaps

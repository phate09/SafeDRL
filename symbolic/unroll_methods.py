"""
Collection of methods to use in unroll_abstract_env
"""
import random
from itertools import cycle
from typing import Tuple, List

import numpy as np
import progressbar
import ray
import scipy.spatial

from mosaic.utils import chunks, filter_helper, interval_contains
from mosaic.workers.AbstractStepWorker import AbstractStepWorker
from mosaic.workers.RemainingWorker import RemainingWorker
from plnn.bab_explore import DomainExplorer
from prism.shared_rtree import SharedRtree
from prism.state_storage import StateStorage, get_storage
from symbolic.cartpole_abstract import CartPoleEnv_abstract
from verification_runs.cartpole_bab_load import generateCartpoleDomainExplorer


def abstract_step_store2(abstract_states_normalised: List[Tuple[Tuple[Tuple[float, float]], bool]], env: CartPoleEnv_abstract, explorer: DomainExplorer, t: int, n_workers: int) -> Tuple[
    List[Tuple[Tuple[float, float]]], List[int]]:
    """
    Given some abstract states, compute the next abstract states taking the action passed as parameter
    :param explorer:
    :param env:
    :param abstract_states_normalised: the abstract states from which to start, list of tuples of intervals
    :return: the next abstract states after taking the action (array)
    """
    next_states = []
    terminal_states = []

    workers = cycle([AbstractStepWorker.remote(explorer, t) for _ in range(n_workers)])
    proc_ids = []
    with progressbar.ProgressBar(prefix="Preparing AbstractStepWorkers ", max_value=len(abstract_states_normalised), is_terminal=True) as bar:
        for i, intervals in enumerate(chunks(abstract_states_normalised, 100)):
            proc_ids.append(next(workers).work.remote(intervals))
            bar.update(i)
    with progressbar.ProgressBar(prefix="Performing abstract step ", max_value=len(proc_ids), is_terminal=True) as bar:
        while len(proc_ids) != 0:
            ready_ids, proc_ids = ray.wait(proc_ids)
            next_states_local, terminal_states_local = ray.get(ready_ids[0])
            next_states.extend(next_states_local)
            terminal_states.extend(terminal_states_local)
            bar.update(bar.value + 1)
    return next_states, terminal_states


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


def assign_action_to_blank_intervals(s_array: List[Tuple[Tuple[float, float]]], n_workers: int) -> Tuple[
    List[Tuple[Tuple[float, float]]], List[Tuple[Tuple[float, float]]], List[Tuple[Tuple[float, float]]]]:
    """
    Given a list of intervals, calculate the intervals where the agent will take a given action
    :param n_workers: number of worker processes
    :param s_array: the list of intervals
    :return: safe intervals,unsafe intervals, ignore/irrelevant intervals
    """
    # convert list of tuples to list of arrays
    s_array = [np.array(x) for x in s_array]
    explorer, verification_model = generateCartpoleDomainExplorer()
    # given the initial states calculate which intervals go left or right
    stats = explorer.explore(verification_model, s_array, n_workers, debug=True)
    print(f"#states: {stats['n_states']} [safe:{stats['safe_relative_percentage']:.3%}, unsafe:{stats['unsafe_relative_percentage']:.3%}, ignore:{stats['ignore_relative_percentage']:.3%}]")
    safe_next = [i.cpu().numpy() for i in explorer.safe_domains]
    unsafe_next = [i.cpu().numpy() for i in explorer.unsafe_domains]
    ignore_next = [i.cpu().numpy() for i in explorer.ignore_domains]
    safe_next = np.stack(safe_next) if len(safe_next) != 0 else []
    unsafe_next = np.stack(unsafe_next) if len(unsafe_next) != 0 else []
    ignore_next = np.stack(ignore_next) if len(ignore_next) != 0 else []
    t_states = ([tuple([(float(x.item(0)), float(x.item(1))) for x in k]) for k in safe_next], [tuple([(float(x.item(0)), float(x.item(1))) for x in k]) for k in unsafe_next],
                [tuple([(float(x.item(0)), float(x.item(1))) for x in k]) for k in ignore_next])
    return t_states


def discard_negligibles(intervals: List[Tuple[Tuple[float, float]]], intervals_id: List[int] = None) -> Tuple[List[Tuple[Tuple[float, float]]], List[int]]:
    """discards the intervals with area 0"""
    result = []
    result_ids = []
    for i, interval in enumerate(intervals):
        sizes = [abs(interval[dimension][1] - interval[dimension][0]) for dimension in range(len(interval))]
        if all([x > 1e-6 for x in sizes]):
            # area = functools.reduce(operator.mul, sizes)  # multiplication of each side of the rectangle
            # if area > 1e-10:
            result.append(interval)
            if intervals_id is not None:
                result_ids.append(intervals_id[i])
    return result, result_ids


def list_t_layer(t: int, solution_min: List, solution_max: List) -> List[Tuple[float, float]]:
    with get_storage() as storage:
        t_ids = storage.get_t_layer(t)
        result = []
        for i in t_ids:
            result.append((solution_min[i], solution_max[i]))
        return result


def analysis_iteration(remainings, t, terminal_states: List[int], failed: List[Tuple[Tuple[float, float]]], n_workers: int, rtree: SharedRtree, env, explorer, storage: StateStorage, failed_area: List,
                       union_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]]) -> List[Tuple[Tuple[float, float]]]:
    assigned_action_intervals = []
    while True:
        remainings, safe_intervals_union, unsafe_intervals_union, remainings_id = compute_remaining_intervals3_multi(remainings, t, n_workers)  # checks areas not covered by total intervals
        assigned_action_intervals.extend([(x, True) for x in safe_intervals_union])
        assigned_action_intervals.extend([(x, False) for x in unsafe_intervals_union])
        # assigned_action_intervals = merge_list_tuple(assigned_action_intervals)  # aggregate intervals

        print(f"Remainings before negligibles: {len(remainings)}")
        remainings, remainings_id = discard_negligibles(remainings, remainings_id)  # discard intervals with area 0
        # todo assign an action to remainings (it might be that our tree does not include the given interval)

        if len(remainings) != 0:
            print(f"Found {len(remainings)} remaining intervals, updating the rtree to cover them")
            # get max of every dimension
            boundaries = []
            for d in range(len(remainings[0])):
                dimension_slice = [(x[d][0], x[d][1]) for x in remainings]
                maximum = float(max(x[1] for x in dimension_slice))
                minimum = float(min(x[0] for x in dimension_slice))
                boundaries.append((minimum, maximum))
            boundaries = tuple(boundaries)
            # once you get the boundaries of the area to lookup we execute the algorithm to assign an action to this area(s)
            safe_intervals_union2, unsafe_intervals_union2, ignore_intervals = assign_action_to_blank_intervals([boundaries], n_workers)
            union_states_total.extend([(x, True) for x in safe_intervals_union2])
            union_states_total.extend([(x, False) for x in unsafe_intervals_union2])
            for x in [(x, True) for x in safe_intervals_union2]:
                rtree.add_single(storage.store(x, t), x)
                assert len(rtree.filter_relevant_intervals3(x[0])) != 0
            for x in [(x, False) for x in unsafe_intervals_union2]:
                rtree.add_single(storage.store(x, t), x)
                assert len(rtree.filter_relevant_intervals3(x[0])) != 0
            rtree.flush()

            # rtree, _ = rebuild_tree(union_states_total, n_workers)  # rebuild the tree to cover the areas that weren't covered before
        else:  # if no more remainings exit
            break
    terminal_states.extend(remainings_id)
    area = sum([calculate_area(np.array(remaining)) for remaining in remainings])
    failed.extend(remainings)
    failed_area[0] += area
    print(f"Remainings : {len(remainings)} Area:{area} Total Area:{failed_area[0]}")
    next_states_array, remainings_id = abstract_step_store2(assigned_action_intervals, env, explorer, t + 1,
                                                            n_workers)  # performs a step in the environment with the assigned action and retrieve the result
    terminal_states.extend(remainings_id)
    print(f"Sucessors : {len(next_states_array)} Terminals : {len(remainings_id)}")

    print(f"t:{t} Finished")
    if len(terminal_states) != 0:
        storage.mark_as_fail(terminal_states)
    return next_states_array


def compute_remaining_intervals3_multi(current_intervals: List[Tuple[Tuple[float, float]]], t: int, n_workers: int) -> Tuple[List[Tuple[Tuple]], List[Tuple[Tuple]], List[Tuple[Tuple]], List[int]]:
    """
    Calculates the remaining areas that are not included in the intersection between current_intervals and intervals_to_fill
    :param current_intervals:
    :return: the blank intervals and the intersection intervals
    """
    workers = cycle([RemainingWorker.remote(t) for _ in range(n_workers)])
    proc_ids = []
    with progressbar.ProgressBar(prefix="Starting workers", max_value=len(current_intervals), is_terminal=True) as bar:
        for i, intervals in enumerate(chunks(current_intervals, 200)):
            proc_ids.append(next(workers).compute_remaining_worker.remote(intervals))
            bar.update(i)
    parallel_result = []
    with progressbar.ProgressBar(prefix="Compute remaining intervals", max_value=len(proc_ids), is_terminal=True) as bar:
        while len(proc_ids) != 0:
            ready_ids, proc_ids = ray.wait(proc_ids)
            parallel_result.append(ray.get(ready_ids[0]))
            bar.update(bar.value + 1)
    if len(parallel_result) != 0:
        archived_results, intersection_intervals_safe, intersection_intervals_unsafe, terminal_ids = zip(*parallel_result)
    else:
        archived_results, intersection_intervals_safe, intersection_intervals_unsafe, terminal_ids = [], [], [], []
    return [i for x in archived_results for i in x], [i for x in intersection_intervals_safe for i in x], [i for x in intersection_intervals_unsafe for i in x], [i for x in terminal_ids for i in x]


def filter_relevant_intervals(current_interval, intervals_to_fill):
    """Filter the intervals relevant to the current_interval"""
    if not ray.is_initialized():
        ray.init(log_to_driver=False, include_webui=True)
    # remote_func = ray.remote(filter_helper)
    results = [filter_helper.remote(interval_to_fill, current_interval) for interval_to_fill in intervals_to_fill]
    final_list = [ray.get(result) for result in results if ray.get(result) is not None]
    return final_list


def filter_relevant_intervals2(current_interval: Tuple[Tuple], intervals_to_fill, kdtree: scipy.spatial.cKDTree):
    """Filter the intervals relevant to the current_interval"""
    k = 100
    result, indices = kdtree.query(np.array(current_interval).mean(axis=1), k=k, p=2, n_jobs=-1)
    final_list = [intervals_to_fill[i] for i in indices if interval_contains(intervals_to_fill[i], current_interval)]
    return final_list
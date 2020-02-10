import shelve
import itertools
from collections import Iterable
from functools import partial
from typing import Tuple, List
import multiprocessing as mp
import numpy as np
import progressbar
import psutil
import ray
import intervals as I
import decimal
import ray
import scipy.spatial
import torch
from rtree import index
from scipy.spatial.ckdtree import cKDTreeNode
from prism.state_storage import get_storage
from prism.state_storage import StateStorage
from itertools import cycle


def compute_remaining_intervals(current_interval, intervals_to_fill) -> set:
    """
    Returns the intervals not covered by @intervals_to_fill within @current_interval
    :param current_interval: the intervals to check against for coverage, the base (list of tuples of tuples)
    :param intervals_to_fill: the list of intervals to check membership against (list of tuples of tuples)
    :return: the list of remaining intervals not covered by @intervals_to_fill
    """
    points_of_interest = []
    dimensions = len(current_interval)
    bar = progressbar.ProgressBar(prefix="Generating points of interest...", max_value=len(intervals_to_fill) + 1, is_terminal=True).start()
    for dimension in range(dimensions):  # adds the upper and lower bounds from the starting interval as points of interest
        points_of_interest.append({current_interval[dimension][0], current_interval[dimension][1]})
    bar.update(1)
    for i, interval in enumerate(intervals_to_fill):  # adds the upper and lower bounds from every other interval interval as points of interest
        for dimension in range(dimensions):
            points_of_interest[dimension].add(max(current_interval[dimension][0], interval[dimension][0]))  # lb
            points_of_interest[dimension].add(min(current_interval[dimension][1], interval[dimension][1]))  # ub
        bar.update(i + 1)
    bar.finish()
    point_of_interest_indices = [list(range(len(points_of_interest[dimension]) - 1)) for dimension in range(dimensions)]
    permutations_indices = list(itertools.product(*point_of_interest_indices))
    permutations_of_interest = []
    bar = progressbar.ProgressBar(prefix="Generating permutations...", max_value=len(permutations_indices), is_terminal=True).start()
    for i, permutation_idx in enumerate(permutations_indices):
        permutations_of_interest.append(
            tuple([(list(points_of_interest[dimension])[permutation_idx[dimension]], list(points_of_interest[dimension])[permutation_idx[dimension] + 1]) for dimension in range(dimensions)]))
        bar.update(i)
    bar.finish()
    dictionary_covered = {}  # dictionary to flag a covered section
    # bar = progressbar.ProgressBar(prefix="Filling dictionary...", max_value=len(permutations_of_interest)).start()
    # for i, permutation in enumerate(permutations_of_interest):
    #     dictionary_covered[permutation] = False
    #     bar.update(i)
    # bar.finish()
    bar = progressbar.ProgressBar(prefix="Computing remaining intervals...", max_value=len(intervals_to_fill) * len(permutations_of_interest), is_terminal=True).start()
    for i, interval in enumerate(intervals_to_fill):
        for j, permutation in enumerate(permutations_of_interest):
            if not dictionary_covered.get(permutation):
                contains = all([contained(permutation[dimension], interval[dimension]) for dimension in range(dimensions)])  # check it is partially contained in every dimension
                if contains:
                    dictionary_covered[permutation] = True
            bar.update(i * len(permutations_of_interest) + j)
    bar.finish()
    remaining_intervals = [x for x in permutations_of_interest if not dictionary_covered.get(x)]
    return set(remaining_intervals)  # these are the remaining areas not covered by any of the "intervals_to_check" list


def compute_remaining_intervals_multi(current_intervals, intervals_to_fill) -> set:
    """Parallelised version of @see compute_remaining_intervals """
    if not ray.is_initialized():
        ray.init(log_to_driver=False)
    results = []
    for current_interval in current_intervals:
        remote_func = ray.remote(compute_remaining_intervals)
        results.append(remote_func.remote(current_interval, intervals_to_fill))
    final_list = []
    for result in results:
        final_list.extend(ray.get(result))
    return set(final_list)


def compute_remaining_intervals2(current_interval, intervals_to_fill, debug=True):
    """
    Computes the intervals which are left blank from the subtraction of intervals_to_fill from current_interval
    :param current_interval:
    :param intervals_to_fill:
    :return: the blank intervals and the union intervals
    """
    "Optimised version of compute_remaining_intervals"
    examine_intervals = []
    remaining_intervals = [current_interval]
    union_intervals = []  # this list will contains the union between intervals_to_fill and current_interval
    dimensions = len(current_interval)
    if len(intervals_to_fill) == 0:
        return remaining_intervals, []
    if debug:
        bar = progressbar.ProgressBar(prefix="Computing remaining intervals...", max_value=len(intervals_to_fill), is_terminal=True).start()
    for i, interval in enumerate(intervals_to_fill):

        examine_intervals.extend(remaining_intervals)
        remaining_intervals = []
        while len(examine_intervals) != 0:
            examine_interval = examine_intervals.pop(0)
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
                        [(list(points_of_interest[dimension])[permutation_idx[dimension]], list(points_of_interest[dimension])[permutation_idx[dimension] + 1]) for dimension in range(dimensions)]))
                for j, permutation in enumerate(permutations_of_interest):
                    if permutation != state:
                        if permutation != examine_interval:
                            examine_intervals.append(permutation)
                        else:
                            remaining_intervals.append(permutation)
                    else:
                        union_intervals.append(permutation)
            else:
                remaining_intervals.append(examine_interval)
        if debug:
            bar.update(i)
    if debug:
        bar.finish()
    return remaining_intervals, union_intervals


def compute_remaining_intervals3(current_interval: Tuple[Tuple[float, float]], intervals_to_fill: List[Tuple[Tuple[Tuple[float, float]], bool]], debug=True):
    """
    Computes the intervals which are left blank from the subtraction of intervals_to_fill from current_interval
    :param current_interval:
    :param intervals_to_fill:
    :return: the blank intervals and the union intervals
    """
    "Optimised version of compute_remaining_intervals"
    examine_intervals = []
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
                        [(list(points_of_interest[dimension])[permutation_idx[dimension]], list(points_of_interest[dimension])[permutation_idx[dimension] + 1]) for dimension in range(dimensions)]))
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
        if debug:
            bar.update(i)
    if debug:
        bar.finish()
    return remaining_intervals, union_safe_intervals, union_unsafe_intervals


def truncate(number: float, decimal_points: int):
    decimal.getcontext().rounding = decimal.ROUND_DOWN
    c = decimal.Decimal(number)
    return float(round(c, decimal_points))


def truncate_multi(interval: Tuple[float], decimal_points: int):
    """Truncate the number of digits to a given number of decimal points
    not working at the moment"""
    return tuple([(interval[dimension][0], interval[dimension][1]) for dimension in range(len(interval))])


def custom_rounding(x, prec=3, base=.05):
    """Rounding with custom base"""
    return round(base * round(float(x) / base), prec)


def compute_remaining_intervals2_multi(current_intervals, intervals_to_fill):
    """
    Calculates the remaining areas that are not included in the intersection between current_intervals and intervals_to_fill
    :param current_intervals:
    :param intervals_to_fill:
    :return: the blank intervals and the intersection intervals
    """
    remaining_intervals = current_intervals.copy()
    archived_results = []
    intersection_intervals = []
    while len(remaining_intervals) != 0:
        current_interval = remaining_intervals.pop(0)
        # relevant_intervals = filter_relevant_intervals(current_interval, intervals_to_fill)
        results, intersection = compute_remaining_intervals2(current_interval, intervals_to_fill)
        intersection_intervals.extend(intersection)
        for result in results:
            if result == current_interval:
                archived_results.append(current_interval)
            else:
                remaining_intervals.append(result)
    return list(set(archived_results)), list(set(intersection_intervals))


def area_tensor(domain: torch.Tensor) -> float:
    '''
    Compute the area of the domain
    '''
    dom_sides = domain.select(1, 1) - domain.select(1, 0)
    dom_area = dom_sides.prod()
    return float(dom_area.item())


def area_numpy(domain: np.ndarray) -> float:
    '''
    Compute the area of the domain
    '''
    dom = np.array(domain)
    dom_sides = dom[:, 1] - dom[:, 0]
    dom_area = dom_sides.prod()
    return float(dom_area.item())


def compute_remaining_intervals3_multi(current_intervals, rtree: index.Index,local_mode:bool) -> Tuple[List[Tuple[Tuple]], List[Tuple[Tuple]], List[Tuple[Tuple]], List[int]]:
    """
    Calculates the remaining areas that are not included in the intersection between current_intervals and intervals_to_fill
    :param current_intervals:
    :return: the blank intervals and the intersection intervals
    """
    n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
    workers = cycle([RemainingWorker.remote(rtree) for _ in range(n_workers)])
    proc_ids = []
    with progressbar.ProgressBar(prefix="Starting workers", max_value=len(current_intervals), is_terminal=True) as bar:
        for i, x in enumerate(current_intervals):
            proc_ids.append(next(workers).compute_remaining_worker.remote(x))
            bar.update(i)
    parallel_result = []
    with progressbar.ProgressBar(prefix="Compute remaining intervals", max_value=len(proc_ids), is_terminal=True) as bar:
        for i, x in enumerate(proc_ids):
            parallel_result.append(ray.get(x))
            bar.update(i)
    archived_results, intersection_intervals_safe, intersection_intervals_unsafe, terminal_ids = zip(*parallel_result)
    return [i for x in archived_results for i in x], [i for x in intersection_intervals_safe for i in x], [i for x in intersection_intervals_unsafe for i in x], [i for x in terminal_ids for i in x]


@ray.remote
class RemainingWorker():
    def __init__(self, tree):
        self.tree = tree  # self.storage = get_storage()

    def compute_remaining_worker(self, current_interval):
        storage = get_storage()
        parent_id = storage.store(current_interval)
        relevant_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = filter_relevant_intervals3(current_interval, self.tree)
        remaining, intersection_safe, intersection_unsafe = compute_remaining_intervals3(current_interval, relevant_intervals, False)
        remaining_ids = []
        for interval in intersection_safe:
            storage.store_successor(interval, parent_id)
        for interval in intersection_unsafe:
            storage.store_successor(interval, parent_id)
        for interval in remaining:  # mark as terminal?
            remaining_ids.append(storage.store_successor(interval, parent_id))
        storage.close()
        return remaining, intersection_safe, intersection_unsafe, remaining_ids

    # def __del__(self):  #     self.storage.close()


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


def filter_relevant_intervals3(current_interval: Tuple[Tuple[float, float]], rtree: index.Index) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    """Filter the intervals relevant to the current_interval"""
    result = rtree.intersection(flatten_interval(current_interval), objects='raw')
    return list(result)


def flatten_interval(current_interval: Tuple[Tuple[float, float]]) -> Tuple:
    return (
        current_interval[0][0], current_interval[0][1], current_interval[1][0], current_interval[1][1], current_interval[2][0], current_interval[2][1], current_interval[3][0], current_interval[3][1])


def search_kd_tree(tree: cKDTreeNode, range: Tuple[Tuple]):
    if tree.split_dim == -1:
        return tree.data_points, tree.indices
    else:
        data_points_list = []
        indices_list = []
        if range[tree.split_dim][0] >= tree.split or range[tree.split_dim][1] >= tree.split:
            left_datapoints, left_indices = search_kd_tree(tree.greater, range)
            data_points_list.extend(left_datapoints)
            indices_list.extend(left_indices)
        elif range[tree.split_dim][0] < tree.split or range[tree.split_dim][1] < tree.split:
            left_datapoints, left_indices = search_kd_tree(tree.greater, range)
            data_points_list.extend(left_datapoints)
            indices_list.extend(left_indices)
        return data_points_list, indices_list


@ray.remote
def filter_helper(interval_to_fill, current_interval):
    """Check if the interval_to_fill is overlaps current_interval and returns a trimmed version of it"""
    contains = interval_contains(interval_to_fill, current_interval)
    return shrink(interval_to_fill, current_interval) if contains else None


def shrink(a, b):
    """Shrink interval a to be at max as big as b"""
    dimensions = len(a)
    state = tuple([(max(a[dimension][0], b[dimension][0]), min(a[dimension][1], b[dimension][1])) for dimension in range(dimensions)])
    return state


def interval_contains(a, b):
    """Condition used to check if an a touches b and partially covers it"""
    dimensions = len(a)
    partial = all([(I.closed(*a[dimension]) & I.open(*b[dimension])).is_empty() == False for dimension in range(dimensions)])
    return partial


def contained(a: tuple, b: tuple):
    return b[0] <= a[0] <= b[1] and b[0] <= a[1] <= b[1]


def partially_contained(a: tuple, b: tuple):
    return b[0] <= a[0] <= b[1] or b[0] <= a[1] <= b[1]


def partially_contained_interval(a: tuple, b: tuple):
    return all([b[dimension][0] <= a[dimension][0] <= b[dimension][1] or b[dimension][0] <= a[dimension][1] <= b[dimension][1] for dimension in range(len(a))])


def non_zero_area(a: tuple):
    return all([abs(bounds[0] - bounds[1]) != 0 for bounds in a])


def beep():
    import os
    duration = 0.5  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))


def shelve_variables2():
    my_shelf = shelve.open('/tmp/shelve.out', 'n')  # 'n' for new

    for key in globals():
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
        except:
            print('GENERIC ERROR shelving: {0}'.format(key))
    my_shelf.close()


def unshelve_variables():
    my_shelf = shelve.open('/tmp/shelve.out')
    for key in my_shelf:
        globals()[key] = my_shelf[key]
    my_shelf.close()


def bulk_load_rtree_helper(data: List[Tuple[Tuple[Tuple[float, float]], bool]]):
    for i, obj in enumerate(data):
        yield (i, (obj[0][0][0], obj[0][0][1], obj[0][1][0], obj[0][1][1], obj[0][2][0], obj[0][2][1], obj[0][3][0], obj[0][3][1]), obj)

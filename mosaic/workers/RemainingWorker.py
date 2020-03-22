import itertools
from typing import List, Tuple

import progressbar
import ray

from mosaic.utils import shrink, interval_contains, round_tuple
from prism.shared_rtree import get_rtree
from prism.state_storage import get_storage


@ray.remote
class RemainingWorker():
    def __init__(self, t: int, rounding: int):
        self.tree = get_rtree()
        self.t = t
        self.storage = get_storage()
        self.rounding = rounding

    def compute_remaining_worker(self, current_intervals: List[Tuple[Tuple[float, float]]]):
        remaining_total = []
        intersection_safe_total = []
        intersection_unsafe_total = []
        remaining_ids_total = []
        for interval in current_intervals:
            interval = round_tuple(interval)
            parent_id = self.storage.store(interval, self.t)
            relevant_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = self.tree.filter_relevant_intervals3(interval)
            remaining, intersection_safe, intersection_unsafe = compute_remaining_intervals3(interval, relevant_intervals, False)
            remaining_ids = []
            for interval in intersection_safe:
                self.storage.store_successor(interval, f"{self.t}.split", parent_id)
            for interval in intersection_unsafe:
                self.storage.store_successor(interval, f"{self.t}.split", parent_id)
            for interval in remaining:  # mark as terminal?
                remaining_ids.append(self.storage.store_successor(interval, f"{self.t}.split", parent_id))
            remaining_total.extend(remaining)
            intersection_safe_total.extend(intersection_safe)
            intersection_unsafe_total.extend(intersection_unsafe)
            remaining_ids_total.extend(remaining_ids)
        return remaining_total, intersection_safe_total, intersection_unsafe_total, remaining_ids_total


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

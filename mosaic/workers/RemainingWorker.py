import itertools
from typing import List, Tuple
from contexttimer import Timer
import progressbar
import ray

from mosaic.utils import round_tuple, shrink, interval_contains
from prism.state_storage import get_storage, StateStorage


@ray.remote
class RemainingWorker:
    def __init__(self, t: int, rounding: int, tree):
        self.tree = tree
        self.t = t
        self.storage: StateStorage = get_storage()
        self.rounding = rounding

    def compute_remaining_worker(self, current_intervals: List[Tuple[Tuple[float, float]]], remove_same=False) -> Tuple[
        List[Tuple[Tuple[float, float]]], List[Tuple[Tuple[Tuple[float, float]], bool]]]:
        remaining_total: List[Tuple[Tuple[float, float]], bool] = []
        intersection_total: List[Tuple[Tuple[Tuple[float, float]], bool]] = []
        parent_ids = self.storage.store_multi(current_intervals)
        for i, interval in enumerate(current_intervals):
            with Timer(factor=1) as t:
                # interval = round_tuple(interval, self.rounding)
                parent_id = parent_ids[i]
                self.storage.assign_t(parent_id, self.t)
                # print("compute remaining")
                print(f"Timer1:{t.elapsed}")
                relevant_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = self.tree.filter_relevant_intervals3(interval, self.rounding)
                if remove_same:
                    relevant_intervals = [x for x in relevant_intervals if x != interval[0]]  # remove itself
                print(f"Timer2:{t.elapsed}")
                remaining, intersection_safe, intersection_unsafe = compute_remaining_intervals3(interval, relevant_intervals, False)
                # print("merging")
                # if len(intersection_safe) != 0:
                #     intersection_safe = merge_simple([(x, True) for x in intersection_safe], self.rounding)  # todo check merge
                # if len(intersection_unsafe) != 0:
                #     intersection_unsafe = merge_simple([(x, False) for x in intersection_unsafe], self.rounding)
                # print("storing successors")
                print(f"Timer3:{t.elapsed}")
                intersection_safe = [(x, True) for x in intersection_safe]
                intersection_unsafe = [(x, False) for x in intersection_unsafe]
                successors_id = self.storage.store_successor_multi([x[0] for x in intersection_safe], parent_id)
                self.storage.assign_t_multi(successors_id, f"{self.t}.split")
                successors_id = self.storage.store_successor_multi([x[0] for x in intersection_unsafe], parent_id)
                self.storage.assign_t_multi(successors_id, f"{self.t}.split")
                print(f"Timer4:{t.elapsed}")
                # print("return results")
                remaining_total.extend(remaining)
                intersection_total.extend(intersection_safe)
                intersection_total.extend(intersection_unsafe)  # remaining_ids_total.extend(remaining_ids)
                print(f"Timer5:{t.elapsed}")
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

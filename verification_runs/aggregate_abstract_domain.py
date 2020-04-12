import math
from functools import partial
from itertools import permutations, cycle
from math import ceil
from typing import Tuple, List
import multiprocessing as mp
import numpy as np
import progressbar
import ray
from rtree import index

from mosaic.utils import partially_contained_interval, partially_contained, contained, chunks, flatten_interval, area_tuple
from prism.shared_dictionary import get_shared_dictionary, SharedDict
from prism.shared_rtree import bulk_load_rtree_helper
from symbolic.unroll_methods import compute_no_overlaps2, compute_remaining_intervals3_multi, filter_relevant_intervals3


def merge_list(frozen_safe, sorted_indices) -> np.ndarray:
    shrank = [frozen_safe[sorted_indices[0]]]
    for i in sorted_indices[1:]:
        last_merged = shrank[-1]
        equal_dim_count = 0
        n_fields = frozen_safe.shape[1]  # the number of entries across the second dimension
        current_safe = frozen_safe[i]
        suitable = True
        for dim in range(n_fields):  # todo vectorise?
            if (last_merged[dim] == current_safe[dim]).all():
                equal_dim_count += 1
            elif not (last_merged[dim][0] == current_safe[dim][1] or last_merged[dim][1] == current_safe[dim][0]):
                suitable = False
        if equal_dim_count >= n_fields - 1 and suitable:  # >3 fields have the same mean
            output = merge_single(last_merged, current_safe, n_fields)
            shrank[-1] = output  # replace last one
        else:
            output = current_safe
            shrank.append(output)  # add to last one
    return np.stack(shrank)


def merge_if_adjacent(first: Tuple[Tuple[float, float]], second: Tuple[Tuple[float, float]]) -> Tuple[Tuple[float, float]] or None:
    """
    Check every dimension, if d-1 dimensions are the same and the last one is adjacent returns the merged interval
    :param first: the first interval
    :param second: the second interval
    :return: the merged interval or None
    """
    n_dim = len(first)
    if n_dim != len(second):
        return None
    n_same_dim = 0
    idx_different_dim = -1
    suitable = True
    if completely_inside(first, second):
        return second
    if completely_inside(second, first):
        return first
    for k in range(n_dim):
        if first[k][0] == second[k][0] and first[k][1] == second[k][1]:
            n_same_dim += 1
        elif partially_contained(first[k], second[k]):
            if idx_different_dim == -1:
                idx_different_dim = k
            else:
                suitable = False
                break
        else:  # the dimensions are detached
            suitable = False
            break
    # suitable = partially_contained_interval(first, second)
    if n_same_dim == n_dim - 1 and suitable:
        merged_interval = [(float(min(first[k][0], second[k][0])), float(max(first[k][1], second[k][1]))) for k in range(n_dim)]
        return tuple(merged_interval)
    else:
        return None


def completely_inside(first: Tuple[Tuple[float, float]], second: Tuple[Tuple[float, float]]):
    n_dim = len(first)
    return all([contained(first[k], second[k]) for k in range(n_dim)])


def merge_single(first, second, n_dims) -> np.ndarray:
    output = []
    for dim in range(n_dims):
        if not (first[dim] == second[dim]).all():
            if first[dim][0] == second[dim][1] or first[dim][1] == second[dim][0]:
                lb = min(first[dim][0], second[dim][0])
                ub = max(first[dim][1], second[dim][1])
                output.append([lb, ub])
                continue
            else:
                # don't know
                raise Exception("unspecified behaviour")
        else:
            output.append(first[dim])
    return np.array(output)


def aggregate(aggregate_list: np.ndarray):
    if len(aggregate_list) == 0:
        return aggregate_list
    new_list = aggregate_list.copy()
    perms = list(permutations(range(4)))
    for i in range(len(perms)):
        order = []
        for index in perms[i]:
            order = [new_list[:, index, 1], new_list[:, index, 0]] + order
        new_list_sorted_indices = np.lexsort(order, axis=0)
        new_list = merge_list(new_list, new_list_sorted_indices)
    return new_list


def merge_simple_interval_only(intervals: List[Tuple[Tuple[float, float]]], rounding: int) -> List[Tuple[Tuple[float, float]]]:
    result = merge_simple([(x, True) for x in intervals], rounding)
    result = [x[0] for x in result]
    return result


def merge_simple(intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    if len(intervals) == 0:
        return intervals
    dimension = len(intervals[0][0])
    aggregated_list: List[Tuple[Tuple[Tuple[float, float]], bool]] = []
    handled_intervals = dict()
    p = index.Property(dimension=dimension)
    helper = bulk_load_rtree_helper(intervals)
    tree_global = index.Index(helper, properties=p, interleaved=False)
    for interval in intervals:
        # print(f"starting process {i}")
        handled = handled_intervals.get(interval, False)
        if not handled:
            near_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = tree_global.intersection(flatten_interval(interval[0]), objects='raw')
            found_match = False
            for neighbour in near_intervals:
                neighbour_handled = handled_intervals.get(neighbour, False)
                same_area = interval == neighbour
                same_action = neighbour[1] == interval[1]
                if not neighbour_handled and not same_area and same_action:
                    # area_before = area_tuple(interval[0]) + area_tuple(neighbour[0])
                    new_interval = (merge_if_adjacent(neighbour[0], interval[0]), interval[1])
                    if new_interval[0] is not None:
                        # area_after = area_tuple(new_interval[0])
                        # if not math.isclose(area_before, area_after):
                        #     assert math.isclose(area_before, area_after), f"The areas do not match: {area_before} vs {area_after}" # it's ok they do not match
                        aggregated_list.append(new_interval)
                        handled_intervals[neighbour] = True  # mark the interval as handled
                        handled_intervals[interval] = True  # mark the interval as handled
                        found_match = True
                        break
            if not found_match:
                aggregated_list.append(interval)  # result = interval
        else:
            # already handled previously
            pass
    return aggregated_list


def merge_with_condition(intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int, max_iter=-1, n_remaining_cutoff=-1) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    old_result = intervals.copy()
    iterations = 0
    if len(intervals) == 0:
        return intervals
    while (True):
        len_before = len(old_result)
        new_result = merge_simple(intervals, rounding)
        len_after = len(new_result)
        iterations += 1
        if max_iter != -1 and iterations >= max_iter:
            break
        if n_remaining_cutoff != -1 and len_after <= n_remaining_cutoff:
            break
        if len_before == len_after:
            break

    return new_result


def merge_sync(intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    handled_intervals = dict()
    p = index.Property(dimension=4)
    helper = bulk_load_rtree_helper(intervals)
    tree_global = index.Index(helper, properties=p, interleaved=False)
    for interval in intervals:
        # print(f"starting process {i}")
        handled = handled_intervals.get(interval, False)
        if not handled:
            near_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = tree_global.intersection(flatten_interval(interval[0]), objects='raw')


def merge_supremum(starting_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]],rounding,n_workers,state_size) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    if len(starting_intervals) == 0:
        return starting_intervals
    intervals = starting_intervals
    state_size = len(intervals[0][0])
    while True:
        p = index.Property(dimension=state_size)
        helper = bulk_load_rtree_helper(intervals)
        tree = index.Index(helper, interleaved=False, properties=p, overwrite=True)
        # find leftmost interval
        left_boundary = tree.bounds[0]
        leftmost_items = list(tree.intersection((left_boundary, left_boundary, tree.bounds[2], tree.bounds[3]), objects="raw"))
        leftmost_items = sorted(leftmost_items)
        top_coordinate = max([x[1][1] for x in leftmost_items])  # the coordinate of the highest point along the left_boundary todo what if detached from lowest point?
        down_coordinate = min([x[1][0] for x in leftmost_items])  # the coordinate of the lowest point along the left_boundary
        relevant_items = list(
            tree.intersection((tree.bounds[0], tree.bounds[1], down_coordinate, top_coordinate), objects="raw"))  # todo check reachability from starting point (method), filter away unreachable points
        supremum = min([x[1][1] for x in relevant_items])  # todo need to filter by the highest minimum
        infimum = max([x[1][0] for x in relevant_items])
        infimum_filtered_intervals = [x for x in relevant_items if x[1][1] >= infimum]
        supremum_filtered_intervals = [x for x in relevant_items if x[1][0] <= supremum]
        supremum_above_infimum = min([x[1][1] for x in infimum_filtered_intervals])
        infimum_below_supremum = max([x[1][0] for x in supremum_filtered_intervals])
        supremum_lr = min([x[0][1] for x in relevant_items])
        infimum_lr = max([x[0][0] for x in relevant_items])
        infimum_lr_filtered_intervals = [x for x in relevant_items if x[0][1] >= supremum_lr]
        supremum_lr_filtered_intervals = [x for x in relevant_items if x[0][0] <= infimum_lr]
        supremum_lr_above_infimum = min([x[0][1] for x in infimum_lr_filtered_intervals])
        infimum_lr_below_supremum = max([x[0][0] for x in supremum_lr_filtered_intervals])
        new_group = ((infimum_lr_below_supremum, supremum_lr_above_infimum), (infimum_below_supremum, supremum_above_infimum))
        remainings = compute_remaining_intervals3_multi()
        p = index.Property(dimension=state_size)
        helper = bulk_load_rtree_helper(intervals)
        if len(intervals) != 0:
            tree = index.Index(helper, interleaved=False, properties=p, overwrite=True)
        else:
            tree = index.Index(interleaved=False, properties=p, overwrite=True)
        intervals_with_relevants = []
        found_list = [False] * len(intervals)
        intervals = remainings

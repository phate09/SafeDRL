from itertools import permutations
from typing import Tuple, List

import numpy as np
import progressbar
from rtree import index

from mosaic.utils import flatten_interval, partially_contained_interval, partially_contained


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


def merge_list_tuple(intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], tree: index.Index) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    aggregated_list = []
    handled_ids = dict()
    handled_intervals = dict()
    merged = True
    # while merged:
    #     merged = False
    aggregated_count = 0
    non_aggregated_count = 0
    widgets = ['Processed: ', progressbar.Counter('%(value)05d'), ' aggregated (', progressbar.Variable('aggregated'),' non_aggregated (', progressbar.Variable('non_aggregated'), ')']
    with progressbar.ProgressBar(max_value=len(intervals), redirect_stdout=True) as bar:
        for i, interval in enumerate(intervals):
            if not handled_ids.get(i, False) and not handled_intervals.get(interval, False):
                near_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = tree.nearest(flatten_interval(interval[0]), num_results=4, objects='raw')
                found_match = False
                for neighbour in near_intervals:
                    if not handled_intervals.get(neighbour, False) and neighbour[1] == interval[1]:
                        new_interval = (merge_if_adjacent(neighbour[0], interval[0]), interval[1])
                        if new_interval[0] is not None:
                            aggregated_count += 1
                            aggregated_list.append(new_interval)
                            handled_intervals[neighbour] = True  # mark the interval as handled
                            handled_intervals[interval] = True  # mark the interval as handled
                            merged = True
                            found_match = True
                            break
                if not found_match:
                    aggregated_list.append(interval)
                    non_aggregated_count += 1
                handled_ids[i] = True  # mark the id as handled
            else:
                # already handled previously
                pass
            bar.update(i)#,aggregated=aggregated_count,non_aggregated=non_aggregated_count)
    bar.finish()
    return aggregated_list


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
    for k in range(n_dim):
        if first[k][0] == second[k][0] and first[k][1] == second[k][1]:
            n_same_dim += 1
        elif partially_contained(first[k],second[k]):
            if idx_different_dim == -1:
                idx_different_dim = k
            else:
                suitable = False
                break
        else:  # the dimensions are detatched
            suitable = False
            break
    # suitable = partially_contained_interval(first, second)
    if n_same_dim == n_dim - 1 and suitable :
        merged_interval = [(float(min(first[k][0], second[k][0])), float(max(first[k][1], second[k][1]))) for k in range(n_dim)]
        return tuple(merged_interval)
    else:
        return None


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

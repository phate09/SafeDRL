from itertools import permutations
from typing import Tuple, List

import numpy as np
from rtree import index

from mosaic.utils import flatten_interval


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


def merge_list_tuple(intervals: List[Tuple[Tuple[float, float]]], tree: index.Index) -> List[Tuple[Tuple[float, float]]]:
    aggregated_list = []
    handled_ids = dict()
    for i, interval in enumerate(intervals):
        if not handled_ids.get(i, False):
            tree.nearest(flatten_interval(interval), num_results=10, objects='raw')
            # todo check alignment/adjacency (check d-1 dimensions are the same)
            aligned = False
            if aligned:
                # todo merge the two intervals
                pass
            else:
                aggregated_list.append(interval)
            handled_ids[i] = True  # mark the id as handled
        else:
            # already handled previously
            pass
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

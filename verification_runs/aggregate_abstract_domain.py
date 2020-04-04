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


# def init(bar, union_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]]):
#     global lock_bar_global, lock_handled_global, progress_val, tree_global, progress_bar_global
#     lock_bar_global = mp.Lock()
#     lock_handled_global = mp.Lock()
#     progress_val = mp.Value('i', 0)
#     progress_bar_global = bar
#     # print("Creating a new shared tree")
#     p = index.Property(dimension=4)
#     helper = bulk_load_rtree_helper(union_states_total)
#     tree_global = index.Index(helper, properties=p, interleaved=False)


def merge_list_tuple(intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], n_workers: int = 8, max_iter: int = -1) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    aggregated_list = intervals
    completed_iterations = 0
    with get_shared_dictionary() as shared_dict:
        while True:
            shared_dict.reset()  # reset the dictionary
            old_size = len(aggregated_list)
            # path = 'save/rtree'
            proc_ids = []
            print(f"About to start the merging process of {old_size} elements")
            workers = cycle([MergingWorker.remote(aggregated_list) for _ in range(n_workers)])
            chunk_size = 200
            with progressbar.ProgressBar(prefix="Starting workers", max_value=ceil(old_size / chunk_size), is_terminal=True, term_width=200) as bar:
                for i, intervals in enumerate(chunks(aggregated_list, chunk_size)):
                    proc_ids.append(next(workers).merge_worker.remote(intervals))
                    bar.update(i)
            aggregated_list = []
            with progressbar.ProgressBar(prefix="Merging intervals", max_value=len(proc_ids), is_terminal=True, term_width=200) as bar:
                while len(proc_ids) != 0:
                    ready_ids, proc_ids = ray.wait(proc_ids)
                    result = ray.get(ready_ids[0])
                    if result is not None:
                        aggregated_list.extend(result)
                    bar.update(bar.value + 1)
            print("Finished!")
            new_size = len(aggregated_list)
            print(f"Reduced size from {old_size} to {new_size}")
            if old_size == new_size:
                break
            completed_iterations += 1
            if completed_iterations >= max_iter != -1:
                break
    return aggregated_list


@ray.remote
class MergingWorker():
    def __init__(self, union_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]]):
        p = index.Property(dimension=4)
        helper = bulk_load_rtree_helper(union_states_total)
        self.tree_global = index.Index(helper, properties=p, interleaved=False)
        self.handled_intervals = get_shared_dictionary()

    def merge_worker(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]):
        aggregated_list = []
        for interval in intervals:
            # print(f"starting process {i}")
            result = None
            handled = self.handled_intervals.get(interval, False)
            # p = index.Property(dimension=4)
            # tree = index.Index(path, properties=p, interleaved=False)
            if not handled:
                near_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = self.tree_global.nearest(flatten_interval(interval[0]), num_results=4, objects='raw')
                found_match = False
                for neighbour in near_intervals:
                    neighbour_handled = self.handled_intervals.get(neighbour, False)
                    same_action = neighbour[1] == interval[1]
                    if not neighbour_handled and same_action:
                        new_interval = (merge_if_adjacent(neighbour[0], interval[0]), interval[1])
                        if new_interval[0] is not None:
                            aggregated_list.append(new_interval)
                            # result = new_interval
                            self.handled_intervals.set(neighbour, True)  # mark the interval as handled
                            self.handled_intervals.set(interval, True)  # mark the interval as handled
                            found_match = True
                            break
                if not found_match:
                    aggregated_list.append(interval)  # result = interval
            else:
                # already handled previously
                pass
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
    aggregated_list: List[Tuple[Tuple[Tuple[float, float]], bool]] = []
    handled_intervals = dict()
    p = index.Property(dimension=4)
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

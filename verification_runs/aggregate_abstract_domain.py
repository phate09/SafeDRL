from functools import partial
from itertools import permutations
from typing import Tuple, List
import multiprocessing as mp
import numpy as np
import progressbar
from rtree import index

from mosaic.utils import flatten_interval, partially_contained_interval, partially_contained, bulk_load_rtree_helper


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


def init(lb, lh, val, path, union_states_total):
    global lock_bar_global, lock_handled_global, progress_val, tree_global
    lock_bar_global = lb
    lock_handled_global = lh
    progress_val = val
    print("Creating a new shared tree")
    p = index.Property(dimension=4)
    helper = bulk_load_rtree_helper(union_states_total)
    tree_global = index.Index(helper, properties=p, interleaved=False)


def merge_list_tuple(intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    # path = 'save/rtree'
    widgets = ['Processed: ', progressbar.Counter('%(value)05d'), ' aggregated (', progressbar.Variable('aggregated'), ' non_aggregated (', progressbar.Variable('non_aggregated'), ')']
    # with progressbar.ProgressBar(max_value=len(intervals), redirect_stdout=True) as bar:
    lh = mp.Lock()
    lb = mp.Lock()
    progress = mp.Value('i', 0)
    with mp.Pool(mp.cpu_count(), initializer=init, initargs=(lb, lh, progress, None, intervals)) as pool:
        # manager = mp.Manager()
        handled_intervals = dict()
        max_value = len(intervals)
        print("About to start the merging process")
        func = partial(merge_worker, handled_intervals, max_value)
        # aggregated_list = [x for x in progressbar.progressbar(pool.imap_unordered(func, intervals,100),max_value=len(intervals))]

        aggregated_list = pool.map(func, intervals, chunksize=100)
        print()
        print("Finished!")
    # bar.finish()
    return aggregated_list


def merge_worker(handled_intervals: dict, max_value: int, interval: Tuple[Tuple[Tuple[float, float]], bool]):
    # print(f"starting process {i}")
    result = None
    lock_handled_global.acquire()
    handled = handled_intervals.get(interval, False)
    lock_handled_global.release()
    # p = index.Property(dimension=4)
    # tree = index.Index(path, properties=p, interleaved=False)
    if not handled:
        near_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = tree_global.nearest(flatten_interval(interval[0]), num_results=4, objects='raw')
        found_match = False
        for neighbour in near_intervals:
            lock_handled_global.acquire()
            neighbour_handled = handled_intervals.get(neighbour, False)
            lock_handled_global.release()
            same_action = neighbour[1] == interval[1]
            if not neighbour_handled and same_action:
                new_interval = (merge_if_adjacent(neighbour[0], interval[0]), interval[1])
                if new_interval[0] is not None:
                    # aggregated_list.append(new_interval)
                    result = new_interval
                    lock_handled_global.acquire()
                    handled_intervals[neighbour] = True  # mark the interval as handled
                    handled_intervals[interval] = True  # mark the interval as handled
                    lock_handled_global.release()
                    found_match = True
                    break
        if not found_match:
            # aggregated_list.append(interval)
            result = interval
    else:
        # already handled previously
        pass
    # print(i)
    lock_bar_global.acquire()
    # bar.update(bar.value + 1)
    # i += 1
    progress_val.value += 1
    print(f"\r{progress_val.value / max_value:02%} - {progress_val.value} of {max_value}", end="")
    lock_bar_global.release()
    return result


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
        elif partially_contained(first[k], second[k]):
            if idx_different_dim == -1:
                idx_different_dim = k
            else:
                suitable = False
                break
        else:  # the dimensions are detatched
            suitable = False
            break
    # suitable = partially_contained_interval(first, second)
    if n_same_dim == n_dim - 1 and suitable:
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

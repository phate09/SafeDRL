from collections import defaultdict
from typing import Tuple, List

from mosaic.hyperrectangle import HyperRectangle
from mosaic.utils import partially_contained, contained, flatten_interval, create_tree


def merge_if_adjacent(first: HyperRectangle, second: HyperRectangle) -> HyperRectangle or None:
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


def completely_inside(first: HyperRectangle, second: HyperRectangle):
    n_dim = len(first)
    return all([contained(first[k], second[k]) for k in range(n_dim)])


def filter_only_connected(intervals_to_filter: List[Tuple[HyperRectangle, bool]], coordinate: Tuple[float] = None) -> List[Tuple[HyperRectangle, bool]]:
    if len(intervals_to_filter) == 0:
        return intervals_to_filter
    connected_dict = defaultdict(bool)
    if coordinate is not None:
        first_element = (tuple([(x, x) for x in coordinate]), True)
    else:
        first_element = intervals_to_filter[0]
    connected_list = [first_element]
    tree = create_tree(connected_list)
    while True:
        found_one = False
        for interval in intervals_to_filter:
            if connected_dict[interval]:
                continue  # skip
            else:
                coordinates = flatten_interval(interval[0])
                intersection = tree.intersection(coordinates, objects='raw')
                n_intersected = len(list(intersection))
                found = n_intersected != 0
                found_one = found_one or found
                if found:
                    connected_dict[interval] = True
                    tree.insert(len(connected_list), coordinates, interval)
                    connected_list.append(interval)
        if not found_one:
            break
    return connected_list[1:]  # remove the first element


def is_connected(interval: Tuple[HyperRectangle, bool], intervals_to_connect: List[Tuple[HyperRectangle, bool]]) -> bool:
    if len(intervals_to_connect) == 0:
        return False
    tree = create_tree(intervals_to_connect)
    intersection = tree.intersection(flatten_interval(interval[0]), objects='raw')
    n_intersected = len(list(intersection))
    return n_intersected != 0

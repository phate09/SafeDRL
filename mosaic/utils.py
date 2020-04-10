import decimal
import math
import operator
import shelve
from collections import defaultdict
from functools import reduce
from typing import Tuple, List
import plotly.graph_objects as go
import intervals as I
import numpy as np
import ray
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def array_to_tuple(array: np.ndarray) -> Tuple[Tuple[float, float]]:
    array_of_tuples = map(tuple, array)
    tuple_of_tuples = tuple(array_of_tuples)
    return tuple_of_tuples


def area_tensor(domain: torch.Tensor) -> float:
    '''
    Compute the area of the domain
    '''
    dom_sides = domain.select(1, 1) - domain.select(1, 0)
    dom_area = dom_sides.prod()
    return abs(float(dom_area.item()))


def area_numpy(domain: np.ndarray) -> float:
    '''
    Compute the area of the domain
    '''
    dom = np.array(domain)
    dom_sides = dom[:, 1] - dom[:, 0]
    dom_area = dom_sides.prod()
    return abs(float(dom_area.item()))


def area_tuple(domain: Tuple[Tuple[float, float]]):
    dimensions = [abs(x[1] - x[0]) for x in domain]
    area = reduce(operator.mul, dimensions, 1)
    return area


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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunker_list(seq, size):
    "splits the list is size sublists"
    return (seq[i::size] for i in range(size))


def round_tuples(intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int = 6) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    return [(round_tuple(interval, rounding), action) for interval, action in intervals]


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def round_tuple(interval: Tuple[Tuple[float, float]], rounding: int) -> Tuple[Tuple[float, float]]:
    return interval  # tuple([(float(round(x[0], rounding)), float(round(x[1], rounding))) for x in interval])


# def inflate(current_interval: Tuple[Tuple[float, float]], rounding: int, eps=1e-6, ) -> Tuple[Tuple[float, float]]:
#     current_interval = round_tuple(tuple([(x[0] - eps, x[1] + eps) for x in current_interval]), rounding)  # rounding
#     return current_interval


def flatten_interval(current_interval: Tuple[Tuple[float, float]]) -> Tuple:
    result = []
    for d in range(len(current_interval)):
        result.extend([current_interval[d][0], current_interval[d][1]])
    return tuple(result)


def show_plot(intervals_action: List[Tuple[Tuple[Tuple[float, float]], bool]] = None, intervals: List[Tuple[Tuple[float, float]]] = None, aggregate=True):
    fig = go.Figure()
    x_y_dict = defaultdict(list)
    if intervals_action is None:
        intervals_action = []
    if intervals is None:
        intervals = []
    intervals_with_action = [(x, None) for x in intervals]
    for interval in intervals_with_action + intervals_action:
        if interval[1] is True:
            color = 'Red'
        elif interval[1] is False:
            color = 'Blue'
        elif interval[1] is None:
            color = 'Green'
        else:
            color = interval[1]
        x = [interval[0][0][0], interval[0][0][1], interval[0][0][1], interval[0][0][0], interval[0][0][0]]
        y = [interval[0][1][0], interval[0][1][0], interval[0][1][1], interval[0][1][1], interval[0][1][0]]
        x_y_dict[color].append((x, y))
    if not aggregate:
        for color in x_y_dict.keys():
            for x, y in x_y_dict[color]:
                fig.add_scatter(x=x, y=y, fill="toself", fillcolor=color)
    else:
        for color in x_y_dict.keys():
            x_list = []
            y_list = []
            for x, y in x_y_dict[color]:
                x_list.extend(x)
                x_list.append(None)
                y_list.extend(y)
                y_list.append(None)
            fig.add_scatter(x=x_list, y=y_list, fill="toself", fillcolor=color)
    fig.update_shapes(dict(xref='x', yref='y'))
    fig.show()

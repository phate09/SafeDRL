import decimal
import shelve
from typing import Tuple

import intervals as I
import numpy as np
import ray
import torch


def array_to_tuple(array: np.ndarray) -> Tuple[Tuple[float, float]]:
    array_of_tuples = map(tuple, array)
    tuple_of_tuples = tuple(array_of_tuples)
    return tuple_of_tuples


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

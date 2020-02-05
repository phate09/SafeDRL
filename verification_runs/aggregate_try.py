import math
import os
from unittest import TestCase

import jsonpickle
from mpmath import iv
from mosaic.utils import *
import numpy as np
import intervals as I
from rtree import index

from verification_runs.aggregate_abstract_domain import merge_list_tuple

if __name__ == '__main__':
    os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
    with open("./save/t_states.json", 'r') as f:
        t_states = jsonpickle.decode(f.read())
    safe_states = t_states[0][0]
    unsafe_states = t_states[0][1]
    safe_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]] = [(tuple([(float(x[0]), float(x[1])) for x in k]), True) for k in safe_states]
    unsafe_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]] = [(tuple([(float(x[0]), float(x[1])) for x in k]), False) for k in unsafe_states]
    union_states_total = safe_states_total + unsafe_states_total
    union_states_total = union_states_total[0:10]
    # previous_total_area = 0
    # for x in union_states_total:
    #     x_array = np.array(x[0])
    #     previous_total_area += area_numpy(x_array)
    previous_areas = [area_numpy(np.array(x[0])) for x in union_states_total]
    helper = bulk_load_rtree_helper(union_states_total)
    # print(list(helper))
    # p = index.Property(dimension=4)
    # r = index.Index('/tmp/rtree',helper, interleaved=False, properties=p)
    # r.flush()
    # r.close()
    result = [x for x in merge_list_tuple(union_states_total) if x is not None]  # '/tmp/rtree'
    print(len(result))

    # total_area = 0
    # for x in [x for x in result if x is not None]:
    #     x_array = np.array(x[0])
    #     total_area += area_numpy(x_array)
    after_areas = [area_numpy(np.array(x[0])) for x in result if x is not None]
    before_area = sum(previous_areas)
    print(f"Area before merge:{before_area}")
    after_area = sum(after_areas)
    print(f"Area after merge:{after_area}")
    assert before_area == after_area

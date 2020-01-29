import os
from unittest import TestCase

import jsonpickle
from mpmath import iv
from mosaic.utils import *
import numpy as np
import intervals as I
from rtree import index


class TestR_trees(TestCase):
    def test_simple(self):
        os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
        with open("./save/t_states.json", 'r') as f:
            t_states = jsonpickle.decode(f.read())
        safe_states = t_states[0][0]
        unsafe_states = t_states[0][1]
        safe_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]] = [(tuple([(float(x[0]), float(x[1])) for x in k]), True) for k in safe_states]
        unsafe_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]] = [(tuple([(float(x[0]), float(x[1])) for x in k]), False) for k in unsafe_states]
        union_states_total = safe_states_total + unsafe_states_total
        helper = bulk_load_rtree_helper(union_states_total)
        # print(list(helper))
        p = index.Property(dimension=4)
        r = index.Index('rtree', helper, interleaved=False, properties=p)
        print(list(r.intersection((0.5, 1.0, 0, 0.25, 0.5, 0.75, 0.5, 0.75), objects='raw')))
        r.close()

    def test_simple2(self):
        os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
        p = index.Property(dimension=4)
        r = index.Index('save/rtree', properties=p, interleaved=False)
        print(list(r.intersection((0.5, 1.0, 0, 0.25, 0.5, 0.75, 0.5, 0.75), objects='raw')))
        print(list(r.intersection((0.5, 1.0, 0, 0.25, 0.5, 0.75, 0.6, 0.75))))


def boxes15_stream(boxes15, interleaved=True):
    for i, (minx, miny, maxx, maxy) in enumerate(boxes15):

        if interleaved:
            yield (i, (minx, miny, maxx, maxy), 42)
        else:
            yield (i, (minx, maxx, miny, maxy), 42)

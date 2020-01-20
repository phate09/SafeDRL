from unittest import TestCase
from mpmath import iv
from mosaic.utils import *
import numpy as np
import intervals as I


class TestCheck_contains(TestCase):
    def setUp(self) -> None:
        self.interval_base_list = []
        self.interval_base_list.append(tuple([(0, 3), (0, 3), (0, 3), (0, 3)]))
        self.interval_base_list.append(tuple([(0, 3), (0, 3), (0, 3), (0, 3)]))
        self.interval_list = []
        self.interval_list.append(tuple([(0, 1), (0, 1), (0, 1), (0, 1)]))
        self.interval_list.append(tuple([(1, 2), (1, 2), (1, 2), (1, 2)]))

    def test_membership(self):
        assert iv.mpf([0.5, 0.5]) in iv.mpf([0.5, 0.5])

    def test_check_contains(self):
        result_list = compute_remaining_intervals(self.interval_base_list[0], self.interval_list)
        assert len(result_list) != 0
        assert ((0, 1), (0, 1), (0, 1), (0, 1)) not in result_list
        assert ((1, 2), (1, 2), (1, 2), (1, 2)) not in result_list

    def test_check_contains_2d(self):
        interval_base_list = []
        interval_base_list.append(tuple([(0, 3), (0, 3)]))
        interval_list = []
        interval_list.append(tuple([(0, 1), (0, 1)]))
        interval_list.append(tuple([(1, 2), (1, 2)]))
        result_list = compute_remaining_intervals(interval_base_list[0], interval_list)
        assert len(result_list) != 0
        assert ((0, 1), (0, 1)) not in result_list

    def test_contains_multi(self):
        """checks if the parallel version works the same as the single version"""
        result_list_multi = compute_remaining_intervals2_multi(self.interval_base_list, self.interval_list)[0]
        result_list = compute_remaining_intervals2(self.interval_base_list[0], self.interval_list)[0]
        assert sorted(result_list) == sorted(result_list_multi)

    def test_check_contains2(self):
        result_list = compute_remaining_intervals2(self.interval_base_list[0], self.interval_list)[0]
        assert len(result_list) != 0
        assert ((0, 1), (0, 1), (0, 1), (0, 1)) not in result_list
        assert ((1, 2), (1, 2), (1, 2), (1, 2)) not in result_list

    def test_check_contains2_2d(self):
        interval_base_list = []
        interval_base_list.append(tuple([(0, 3), (0, 3)]))
        interval_list = []
        interval_list.append(tuple([(0, 1), (0, 1)]))
        interval_list.append(tuple([(1, 2), (1, 2)]))
        result_list = compute_remaining_intervals2(interval_base_list[0], interval_list)[0]
        assert len(result_list) != 0
        assert ((0, 1), (0, 1)) not in result_list

    def test_check_contains2_2d2(self):
        interval_base_list = []
        interval_base_list.append(tuple([(0, 3), (0, 3)]))
        interval_list = []
        interval_list.append(tuple([(0, 3), (0, 3)]))
        result_list = compute_remaining_intervals2(interval_base_list[0], interval_list)[0]
        assert len(result_list) == 0
        assert ((0, 1), (0, 1)) not in result_list

    def test_check_contains2_2d3(self):
        interval_base_list = []
        interval_base_list.append(tuple([(0, 3), (0, 3)]))
        interval_list = []
        interval_list.append(tuple([(0, 3), (0, 1)]))
        result_list = compute_remaining_intervals2(interval_base_list[0], interval_list)[0]
        assert len(result_list) != 0
        assert ((0, 1), (0, 1)) not in result_list
        assert ((0, 3), (1, 3)) in result_list

    def test_check_contains2_2d4(self):
        interval_base_list = []
        interval_base_list.append(tuple([(0, 3), (0, 3)]))
        interval_list = []
        interval_list.append(tuple([(1, 2), (1, 2)]))
        interval_list.append(tuple([(0, 3), (0, 3)]))
        interval_list.append(tuple([(0, 3), (1, 2)]))
        interval_list.append(tuple([(0, 3), (-1, 4)]))
        interval_list.append(tuple([(-1, 4), (-1, 4)]))
        result_list = compute_remaining_intervals2(interval_base_list[0], [interval_list[0]])[0]
        assert len(result_list) != 0
        result_list = compute_remaining_intervals2(interval_base_list[0], [interval_list[1]])[0]
        assert len(result_list) == 0
        result_list = compute_remaining_intervals2(interval_base_list[0], [interval_list[2]])[0]
        assert len(result_list) != 0
        result_list = compute_remaining_intervals2(interval_base_list[0], [interval_list[3]])[0]
        assert len(result_list) == 0
        result_list = compute_remaining_intervals2(interval_base_list[0], [interval_list[4]])[0]
        assert len(result_list) == 0

    def test_interval_contains(self):
        interval_base_list = []
        interval_base_list.append(tuple([(0, 3), (0, 3)]))
        interval_list = []
        interval_list.append(tuple([(1, 2), (1, 2)]))
        interval_list.append(tuple([(0, 3), (0, 3)]))
        interval_list.append(tuple([(0, 3), (1, 2)]))
        interval_list.append(tuple([(0, 3), (-1, 4)]))
        interval_list.append(tuple([(-1, 4), (-1, 4)]))
        assert interval_contains(interval_list[0], interval_base_list[0])
        assert interval_contains(interval_list[1], interval_base_list[0])
        assert interval_contains(interval_list[2], interval_base_list[0])
        assert interval_contains(interval_list[3], interval_base_list[0])
        assert interval_contains(interval_list[4], interval_base_list[0])

    def test_intervals_methods(self):
        a = I.closed(0,3)
        result = a & I.closed(-1,4)
        assert not result.is_empty()
        assert not (I.closed(0,3) & I.closed(0,3)).is_empty()
        assert not (I.closed(0,3) & I.open(0,3)).is_empty()
        assert not (I.closed(0,3) & I.open(1,2)).is_empty()
        assert not (I.closed(0,3) & I.open(-1,4)).is_empty()
        assert (I.closed(0,3) & I.open(3,4)).is_empty()
        assert I.closed(0,3) in I.closed(-1,4)
        assert I.closed(0,3) in I.closed(0,3)


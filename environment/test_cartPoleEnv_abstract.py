from unittest import TestCase
import decimal
from mpmath import iv
from interval import interval, imath
import numpy as np


class TestCartPoleEnv_abstract(TestCase):
    def test_mpf(self):
        c1 = decimal.Decimal(34.1499123)
        c2 = decimal.Decimal(35.359283476598499123)
        interval_result = interval([c1, c2])
        assert interval_result is not None
        cosine = imath.cos(interval_result)
        assert cosine is not None

    def test_interval_math(self):
        rand_points = np.random.rand(1000)
        max_point = rand_points.max()
        min_point = rand_points.min()
        interval = iv.mpf([min_point, max_point])
        after_cosine = iv.cos(interval)
        cosine_points = np.cos(rand_points)
        check = all([x in after_cosine for x in cosine_points])
        assert check

import unittest
from math import sin
import math

class MyTestCase(unittest.TestCase):
    def __init__(self):
        self.polemass_length = 1
        self.total_mass = 3
    def test_something(self):
        force = 1
        theta_dot = 2
        theta = 1
        sintheta = sin(theta)
        sintheta_i = math.asin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        temp_i = (force - self.polemass_length / theta_dot / theta_dot / sintheta_i) / self.total_mass
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()

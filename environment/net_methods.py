import torch
import numpy as np


def generate_nn(x, min_speed=20, max_speed=30, min_distance=40, max_distance=50):
    """x: an array with [speed,distance]"""

    A = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    b = np.array([-min_speed, -max_speed, -min_distance, -max_distance])
    z1 = x @ A + b
    # print(z1)
    x2 = np.maximum(0, z1)  # relu
    A2 = -np.eye(4)
    b2 = np.array([[1, 1, 1, 1]])
    z2 = x2 @ A2 + b2
    # print(z2)
    x3 = np.maximum(0, z2)  # relu
    # print(x3)  # (here there are 1s where it should accelerate, 0s where it should decelerate)
    A3 = np.array([[1, 1, 0, 0], [0, 0, 1, 1]]).T
    b3 = np.array([[0, 0]])
    z3 = x3 @ A3 + b3
    # print(z3)  # OR gate (not scaled) for the speed and the distance
    # now i'll try to scale it in the range [0,1]
    x4 = np.maximum(0, z3)  # relu
    A4 = -np.eye(2)
    b4 = np.array([[1, 1]])
    z4 = x4 @ A4 + b4
    # print(z4)  # now it tells whether we are above (1) below(-1) or within range (0)
    x5 = np.maximum(0, z4)  # relu , above (1), within range or below (0)
    A5 = np.array([[-10, -10]]).T
    b5 = np.array([[1]])
    z5 = x5 @ A5 + b5
    # print(z5)
    y = np.maximum(0, z5)  # relu, 0 decelerate, 1 accelerate
    return y


if __name__ == '__main__':
    x = np.array([[25, 45],[18,50.5]])
    print(generate_nn(x))

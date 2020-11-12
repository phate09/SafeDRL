import torch
import numpy as np
from sklearn.model_selection import ParameterGrid
import pandas as pd
import plotly.express as px
import torch
import torch.nn


def generate_nn(x, min_speed=20, max_speed=30, min_distance=40, max_distance=50):
    """x: an array with [speed,distance]"""

    A = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    b = np.array([-min_speed, -max_speed, -min_distance, -max_distance])
    z1 = x @ A + b
    print(z1)
    x2 = np.maximum(0, z1)  # relu
    A2 = -np.eye(4)
    b2 = np.array([[1, 1, 1, 1]])
    z2 = x2 @ A2 + b2
    print(z2)
    x3 = np.maximum(0, z2)  # relu
    print(x3)  # (here there are 1s where it should accelerate, 0s where it should decelerate)
    A3 = np.array([[1, 1, 0, 0], [0, 0, 1, 1]]).T
    b3 = np.array([[0, 0]])
    z3 = x3 @ A3 + b3
    print(z3)  # OR gate (not scaled) for the speed and the distance
    # now i'll try to scale it in the range [0,1]
    x4 = np.maximum(0, z3)  # relu
    A4 = -np.eye(2)
    b4 = np.array([[1, 1]])
    z4 = x4 @ A4 + b4
    print(z4)  # now it tells whether we are above (1) below(-1) or within range (0)
    x5 = np.maximum(0, z4)  # relu , above (1), within range or below (0)
    A5 = np.array([[-10, -10]]).T
    b5 = np.array([[1]])
    z5 = x5 @ A5 + b5
    print(z5)
    y = np.maximum(0, z5)  # relu, 0 decelerate, 1 accelerate
    return y


def generate_nn_torch(min_speed=20, max_speed=30, min_distance=40, max_distance=50):
    layers = []
    l1 = torch.nn.Linear(2, 4)
    l1.weight = torch.nn.Parameter(torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.float64).T)
    l1.bias = torch.nn.Parameter(torch.tensor([-min_speed, -max_speed, -min_distance, -max_distance], dtype=torch.float64))
    layers.append(l1)
    # z1 = l1(x)
    layers.append(torch.nn.ReLU())
    # x2 = torch.nn.functional.relu(z1)
    l2 = torch.nn.Linear(4, 4)
    l2.weight = torch.nn.Parameter(-torch.eye(4, dtype=torch.float64).T)
    l2.bias = torch.nn.Parameter(torch.tensor([1, 1, 1, 1], dtype=torch.float64))
    layers.append(l2)
    # z2 = l2(x2)
    layers.append(torch.nn.ReLU())
    # x3 = torch.nn.functional.relu(z2)
    l3 = torch.nn.Linear(4, 2)
    l3.weight = torch.nn.Parameter(torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.float64))  #
    l3.bias = torch.nn.Parameter(torch.tensor([0, 0], dtype=torch.float64))
    layers.append(l3)
    # z3 = l3(x3)
    layers.append(torch.nn.ReLU())
    # x4 = torch.nn.functional.relu(z3)
    l4 = torch.nn.Linear(2, 2)
    l4.weight = torch.nn.Parameter(-torch.eye(2, dtype=torch.float64).T)
    l4.bias = torch.nn.Parameter(torch.tensor([1, 1], dtype=torch.float64))
    layers.append(l4)
    # z4 =l4(x4)
    layers.append(torch.nn.ReLU())
    # x5 = torch.nn.functional.relu(z4)
    l5 = torch.nn.Linear(2, 1)
    l5.weight = torch.nn.Parameter(torch.tensor([[-10, -10]], dtype=torch.float64))  #
    l5.bias = torch.nn.Parameter(torch.tensor([1], dtype=torch.float64))
    layers.append(l5)
    # z5 = l5(x5)
    layers.append(torch.nn.ReLU())
    # x6 = torch.nn.functional.relu(z5)
    return torch.nn.Sequential(*layers)


def test_nn_numpy():
    x = generate_mock_input()
    y = generate_nn(x)

    print(y)
    fig = px.scatter(x=x[:, 0], y=x[:, 1], color=y.astype(str), color_continuous_scale=px.colors.sequential.Viridis)
    fig.show()


def generate_mock_input():
    precision = 0.1
    rounding = 1
    param_grid = {'speed': list(np.arange(15, 35, precision).round(rounding)), 'distance': list(np.arange(35, 55, precision).round(rounding))}
    # param_grid = {'param1': list(np.arange(0.33, 0.35, precision).round(rounding)), 'param2': list(np.arange(-0.19, -0.15, precision).round(rounding))}
    grid = ParameterGrid(param_grid)
    points = []
    for params in grid:
        state = np.array((params['speed'], params['distance']))
        points.append(state)
    # x = np.array([[25, 45], [18, 50.5]])
    x = np.stack(points)
    return x


def test_nn_torch():
    x_numpy = generate_mock_input()
    x = torch.tensor(x_numpy)
    nn = generate_nn_torch()
    y = nn(x)
    print(y)
    fig = px.scatter(x=x_numpy[:, 0], y=x_numpy[:, 1], color=y[:, 0].detach().numpy().astype(str), color_continuous_scale=px.colors.sequential.Viridis)
    fig.show()


if __name__ == '__main__':
    test_nn_torch()

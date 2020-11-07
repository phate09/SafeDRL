import torch
import torch.nn
import numpy as np
import gurobi as grb
from environment.net_methods import generate_nn_torch, generate_mock_input


def analyse_input(input, nn: torch.nn.Sequential):
    gurobi_model = grb.Model()
    print(input)
    input = gurobi_model.addMVar(shape=(2,))
    gurobi_vars = []
    gurobi_vars.append(input)
    for layer in nn:

        print(layer)
        if type(layer) is torch.nn.Linear:
            v = gurobi_model.addMVar(lb=float("-inf"),shape=(layer.out_features))
            lin_expr = layer.weight.data.numpy() @ gurobi_vars[-1] + layer.bias.data.numpy()
            gurobi_model.addConstr(v == lin_expr)
            gurobi_vars.append(v)
            gurobi_model.update()
            gurobi_model.optimize()
            assert gurobi_model.status == 2, "LP wasn't optimally solved"
        elif type(layer) is torch.nn.ReLU:
            v = gurobi_model.addMVar(lb=float("-inf"),shape=gurobi_vars[-1].shape)  # same shape as previous
            z = gurobi_model.addMVar(lb=0, ub=1, shape=gurobi_vars[-1].shape, vtype=grb.GRB.INTEGER)
            M = 10e6
            # gurobi_model.addConstr(v == grb.max_(0, gurobi_vars[-1]))
            gurobi_model.addConstr(v >= gurobi_vars[-1])
            gurobi_model.addConstr(v <= gurobi_vars[-1]+M*z)
            gurobi_model.addConstr(v >= 0)
            gurobi_model.addConstr(v <= M - M*z)
            gurobi_vars.append(v)
            gurobi_model.update()
            gurobi_model.optimize()
            assert gurobi_model.status == 2, "LP wasn't optimally solved"
            """
            y = Relu(x)
            0 <= z <= 1, z is integer
            y >= x
            y <= x + Mz
            y >= 0
            y <= M - Mz"""
    gurobi_model.update()
    #     print(f'v={v}')
    # gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
    gurobi_model.optimize()
    assert gurobi_model.status == 2, "LP wasn't optimally solved"
    print(gurobi_model)


if __name__ == '__main__':
    nn = generate_nn_torch()
    x = generate_mock_input()[0]
    analyse_input(x, nn)

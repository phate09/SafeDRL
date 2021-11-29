import gurobi as grb
import torch
import torch.nn

from polyhedra.net_methods import generate_nn_torch


def analyse_input(nn: torch.nn.Sequential):
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', False)
    v = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), name="input")
    gurobi_vars = []
    gurobi_vars.append(v)
    for i, layer in enumerate(nn):

        print(layer)
        if type(layer) is torch.nn.Linear:
            v = gurobi_model.addMVar(lb=float("-inf"), shape=(layer.out_features), name=f"layer_{i}")
            lin_expr = layer.weight.data.numpy() @ gurobi_vars[-1] + layer.bias.data.numpy()
            gurobi_model.addConstr(v == lin_expr)
            gurobi_vars.append(v)
            gurobi_model.update()
            gurobi_model.optimize()
            assert gurobi_model.status == 2, "LP wasn't optimally solved"
        elif type(layer) is torch.nn.ReLU:
            v = gurobi_model.addMVar(lb=float("-inf"), shape=gurobi_vars[-1].shape, name=f"layer_{i}")  # same shape as previous
            z = gurobi_model.addMVar(lb=0, ub=1, shape=gurobi_vars[-1].shape, vtype=grb.GRB.INTEGER, name=f"relu_{i}")
            M = 10e6
            # gurobi_model.addConstr(v == grb.max_(0, gurobi_vars[-1]))
            gurobi_model.addConstr(v >= gurobi_vars[-1])
            gurobi_model.addConstr(v <= gurobi_vars[-1] + M * z)
            gurobi_model.addConstr(v >= 0)
            gurobi_model.addConstr(v <= M - M * z)
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
    # gurobi_model.setObjective(v[0].sum(), grb.GRB.MINIMIZE)
    gurobi_model.update()
    gurobi_model.optimize()
    assert gurobi_model.status == 2, "LP wasn't optimally solved"
    # print_gurobi(gurobi_model)
    return gurobi_model, v


def print_gurobi(gurobi_model):
    for var in gurobi_model.getVars():
        print(f"{var.varName}: {var.X}")
    print(f"Objective value: {gurobi_model.objVal}")


if __name__ == '__main__':
    nn = generate_nn_torch()
    model, y = analyse_input(nn)
    model.setObjective(y[0].sum(), grb.GRB.MINIMIZE)  # minimise the output
    model.optimize()
    print_gurobi(model)
    model.setObjective(y[0].sum(), grb.GRB.MAXIMIZE)  # maximise the output
    model.optimize()
    print_gurobi(model)

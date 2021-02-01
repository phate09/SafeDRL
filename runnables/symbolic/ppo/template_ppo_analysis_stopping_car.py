import ray
import torch
from agents.ppo.train_PPO_car import get_PPO_trainer
from agents.ray_utils import convert_DQN_ray_policy_to_sequential, convert_ray_policy_to_sequential
import gurobi as grb

from polyhedra.runnable.experiment.run_experiment_stopping_car import StoppingCarExperiment
from symbolic import unroll_methods

from polyhedra.experiments_nn_analysis import Experiment
import numpy as np


def get_nn():
    global config, l0
    config, trainer = get_PPO_trainer(use_gpu=0)
    trainer.restore("/home/edoardo/ray_results/PPO_StoppingCar_2020-12-30_17-06-3265yz3d63/checkpoint_65/checkpoint-65")
    policy = trainer.get_policy()
    sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
    l0 = torch.nn.Linear(6, 2, bias=False)
    l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0]], dtype=torch.float32))
    layers = [l0]
    for l in sequential_nn:
        layers.append(l)
    nn = torch.nn.Sequential(*layers)
    return nn


def get_range_bounds(input, nn):
    gurobi_vars = []
    gurobi_vars.append(input)
    for i, layer in enumerate(nn):

        # print(layer)
        if type(layer) is torch.nn.Linear:
            v = gurobi_model.addMVar(lb=float("-inf"), shape=(layer.out_features), name=f"layer_{i}")
            lin_expr = layer.weight.data.numpy() @ gurobi_vars[-1]
            if layer.bias is not None:
                lin_expr = lin_expr + layer.bias.data.numpy()
            gurobi_model.addConstr(v == lin_expr, name=f"linear_constr_{i}")
            gurobi_vars.append(v)
        elif type(layer) is torch.nn.ReLU:
            v = gurobi_model.addMVar(lb=float("-inf"), shape=gurobi_vars[-1].shape, name=f"layer_{i}")  # same shape as previous
            z = gurobi_model.addMVar(lb=0, ub=1, shape=gurobi_vars[-1].shape, vtype=grb.GRB.INTEGER, name=f"relu_{i}")
            M = 10e6
            # gurobi_model.addConstr(v == grb.max_(0, gurobi_vars[-1]))
            gurobi_model.addConstr(v >= gurobi_vars[-1], name=f"relu_constr_1_{i}")
            gurobi_model.addConstr(v <= gurobi_vars[-1] + M * z, name=f"relu_constr_2_{i}")
            gurobi_model.addConstr(v >= 0, name=f"relu_constr_3_{i}")
            gurobi_model.addConstr(v <= M - M * z, name=f"relu_constr_4_{i}")
            gurobi_vars.append(v)
            # gurobi_model.update()
            # gurobi_model.optimize()
            # assert gurobi_model.status == 2, "LP wasn't optimally solved"
            """
            y = Relu(x)
            0 <= z <= 1, z is integer
            y >= x
            y <= x + Mz
            y >= 0
            y <= M - Mz"""
    # gurobi_model.update()
    # gurobi_model.optimize()
    # assert gurobi_model.status == 2, "LP wasn't optimally solved"
    last_layer = gurobi_vars[-1]
    ranges = []
    for i in range(last_layer.shape[0]):
        gurobi_model.setObjective(last_layer[i].sum(), grb.GRB.MAXIMIZE)  # maximise the output
        gurobi_model.update()
        gurobi_model.optimize()
        ub = last_layer[i].X[0]
        gurobi_model.setObjective(last_layer[i].sum(), grb.GRB.MINIMIZE)  # maximise the output
        gurobi_model.update()
        gurobi_model.optimize()
        lb = last_layer[i].X[0]
        ranges.append((lb, ub))
    return ranges


def optimise(templates: np.ndarray, gurobi_model: grb.Model, x_prime: tuple, env_input_size: int):
    results = []
    for template in templates:
        gurobi_model.update()
        gurobi_model.setObjective(sum((template[i] * x_prime[i]) for i in range(env_input_size)), grb.GRB.MAXIMIZE)
        gurobi_model.optimize()
        # print_model(gurobi_model)
        if gurobi_model.status != 2:
            return None
        result = gurobi_model.ObjVal
        results.append(result)
    return np.array(results)


if __name__ == '__main__':
    ray.init(local_mode=False)
    nn = get_nn()

    output_flag = False
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', output_flag)
    env_input_size = 6
    input_boundaries = [50, -40, 10, -0, 28, -28, 36, -36, 0, -0, 0, -0, 0]
    x = input_boundaries
    template = Experiment.box(env_input_size)
    ranges_probs = None
    for chosen_action in range(2):
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        input = Experiment.generate_input_region(gurobi_model, template, x, env_input_size)
        x_prime = StoppingCarExperiment.apply_dynamic(input, gurobi_model, action=chosen_action, env_input_size=env_input_size)
        if ranges_probs is None:
            ranges = get_range_bounds(input, nn)
            ranges_probs = unroll_methods.softmax_interval(ranges)
            print(ranges)
        x_prime_results = optimise(template, gurobi_model, x_prime, env_input_size)
        # todo create the tree
        # todo connect the tree to prism

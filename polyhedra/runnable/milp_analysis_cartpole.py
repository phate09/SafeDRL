import math
from collections import defaultdict

import gurobi as grb
import numpy as np
import pypoman
import ray
import torch

from agents.ppo.train_PPO_cartpole import get_PPO_trainer
from agents.ray_utils import convert_ray_policy_to_sequential
from environment.cartpole_ray import CartPoleEnv
from polyhedra.graph_explorer import GraphExplorer
from polyhedra.net_methods import generate_nn_torch
from polyhedra.plot_utils import show_polygon_list2
from interval import interval, imath

env_input_size = 4


def generate_input_region(gurobi_model, templates, boundaries):
    input = gurobi_model.addMVar(shape=env_input_size, lb=float("-inf"), name="input")
    for j, template in enumerate(templates):
        gurobi_model.update()
        multiplication = 0
        for i in range(env_input_size):
            multiplication += template[i] * input[i]
        gurobi_model.addConstr(multiplication <= boundaries[j], name=f"input_constr_{j}")
    return input


def generate_mock_guard(gurobi_model: grb.Model, input, action_ego=0, t=0):
    if action_ego == 0:  # decelerate
        if t == 0:
            gurobi_model.addConstr(input[3] >= 36, name=f"cond2_{t}")
        else:
            gurobi_model.addConstr((input[0] - input[1]) <= 20, name=f"cond1_{t}")
    else:  # accelerate
        epsilon = 1e-4
        gurobi_model.addConstr(input[3] <= 36 - epsilon, name=f"cond2_{t}")
        gurobi_model.addConstr((input[0] - input[1]) >= 20 + epsilon, name=f"cond1_{t}")


def generate_nn_guard(gurobi_model: grb.Model, input, nn: torch.nn.Sequential, action_ego=0):
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
    # gurobi_model.setObjective(v[action_ego].sum(), grb.GRB.MAXIMIZE)  # maximise the output
    last_layer = gurobi_vars[-1]
    if action_ego == 0:
        gurobi_model.addConstr(last_layer[0] >= last_layer[1], name="last_layer")
    else:
        gurobi_model.addConstr(last_layer[1] >= last_layer[0], name="last_layer")
    gurobi_model.update()
    gurobi_model.optimize()
    # assert gurobi_model.status == 2, "LP wasn't optimally solved"
    return gurobi_model.status == 2


def apply_dynamic(input, gurobi_model: grb.Model, action_ego, thetaacc, xacc, case=0):
    '''

    :param costheta: gurobi variable containing the range of costheta values
    :param sintheta: gurobi variable containin the range of sintheta values
    :param input:
    :param gurobi_model:
    :param action_ego:
    :param t:
    :return:
    '''

    tau = 0.02  # seconds between state updates
    x = input[0]
    x_dot = input[1]
    theta = input[2]
    theta_dot = input[3]
    z = gurobi_model.addMVar(shape=(env_input_size,), lb=float("-inf"), name=f"x_prime")
    x_prime = x + tau * x_dot
    x_dot_prime = x_dot + tau * xacc
    theta_prime = theta + tau * theta_dot
    theta_dot_prime = theta_dot + tau * thetaacc
    gurobi_model.addConstr(z[0] == x_prime, name=f"dyna_constr_1")
    gurobi_model.addConstr(z[1] == x_dot_prime, name=f"dyna_constr_2")
    gurobi_model.addConstr(z[2] == theta_prime, name=f"dyna_constr_3")
    gurobi_model.addConstr(z[3] == theta_dot_prime, name=f"dyna_constr_4")
    return z


def create_temp_var(gurobi_model, expression: grb.MLinExpr, name):
    temp_var = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name=f"{name}")
    gurobi_model.addConstr(temp_var == expression, name=f"{name}_constr_1")
    return temp_var


def get_sin_cos_table(n_split=24, action=0):
    sin_cos_table = []
    start_theta = -12
    end_theta = 12
    start_theta_dot = -10
    end_theta_dot = 10
    split_theta, step_theta = np.linspace(start_theta, end_theta, endpoint=False, retstep=True, num=n_split)
    split_theta_dot, step_theta_dot = np.linspace(start_theta_dot, end_theta_dot, endpoint=False, retstep=True, num=n_split)
    env = CartPoleEnv(None)
    force = env.force_mag if action == 1 else -env.force_mag

    for t_dot in split_theta_dot:
        lb_theta_dot = t_dot
        ub_theta_dot = min(t_dot + step_theta_dot, end_theta_dot)
        theta_dot = interval([lb_theta_dot, ub_theta_dot])
        for s in split_theta:
            lb = s * 2 * math.pi / 360
            ub = min(s + step_theta, end_theta) * 2 * math.pi / 360
            theta = interval([lb, ub])
            sintheta = imath.sin(theta)
            costheta = imath.cos(theta)
            temp = (force + env.polemass_length * theta_dot ** 2 * sintheta) / env.total_mass
            thetaacc = (env.gravity * sintheta - costheta * temp) / (env.length * (4.0 / 3.0 - env.masspole * costheta ** 2 / env.total_mass))
            xacc = temp - env.polemass_length * thetaacc * costheta / env.total_mass
            sin_cos_table.append((theta, theta_dot, thetaacc, xacc))
    return sin_cos_table


def optimise(templates, gurobi_model: grb.Model, x_prime):
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
    print(results)
    return np.array(results)


def create_window_boundary(template_input, x_results, template_2d, window_boundaries):
    assert len(window_boundaries) == 4
    window_template = np.vstack([template_input, template_2d, -template_2d])  # max and min
    window_boundaries = np.stack((window_boundaries[0], window_boundaries[1], -window_boundaries[2], -window_boundaries[3]))
    window_boundaries = np.concatenate((x_results, window_boundaries))
    # windowed_projection_max = np.maximum(window_boundaries, projection)
    # windowed_projection_min = np.minimum(window_boundaries, projection)
    # windowed_projection = np.concatenate((windowed_projection_max.squeeze(), windowed_projection_min.squeeze()), axis=0)
    return window_template, window_boundaries


def main():
    output_flag = False
    ray.init(local_mode=True)
    config, trainer = get_PPO_trainer(use_gpu=0)
    trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-03_14-02-48ynltrgiw/checkpoint_8/checkpoint-8")
    policy = trainer.get_policy()
    sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
    layers = []
    for l in sequential_nn:
        layers.append(l)

    nn = torch.nn.Sequential(*layers)
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', output_flag)

    input_boundaries, template = get_template(0)

    input = generate_input_region(gurobi_model, template, input_boundaries)
    # _, template = get_template(1)
    x_results = optimise(template, gurobi_model, input)
    # input_boundaries, template = get_template(1)
    if x_results is None:
        print("Model unsatisfiable")
        return
    root = tuple(x_results)
    template_2d = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    vertices_list = defaultdict(list)
    # vertices = windowed_projection(template, x_results, template2d_speed)
    # vertices_list.append([vertices])

    # show_polygon_list2(vertices_list)
    seen = []
    frontier = [(0, root)]
    max_t = 0
    while len(frontier) != 0:
        t, x = frontier.pop()
        max_t = max(max_t, t)
        if t > 200:
            break
        if any([contained(x, s) for s in seen]):
            continue
        vertices = windowed_projection(template, np.array(x), template_2d)
        vertices_list[t].append(vertices)
        seen.append(x)
        x_primes = post(x, nn, output_flag, t, template)
        for x_prime in x_primes:
            x_prime = tuple(np.array(x_prime).round(4))  # todo should we round to prevent numerical errors?
            frontier = [(u, y) for u, y in frontier if not contained(y, x_prime)]
            if not any([contained(x_prime, y) for u, y in frontier]):
                frontier.append(((t + 1), x_prime))

    print(f"T={max_t}")
    show_polygon_list2(vertices_list, "x_lead-x_ego", "x_ego")  # show_polygon_list2(vertices_list)


def contained(x: tuple, y: tuple):
    # y contains x
    assert len(x) == len(y)
    for i in range(len(x)):
        if x[i] > y[i]:
            return False
    return True


def generate_angle_guard(gurobi_model: grb.Model, input, angle_interval, theta_dot_interval):
    eps = 1e-6
    gurobi_model.addConstr(input[2] >= angle_interval[0].inf, name=f"theta_guard1")
    gurobi_model.addConstr(input[2] <= angle_interval[0].sup, name=f"theta_guard2")
    gurobi_model.addConstr(input[3] >= theta_dot_interval[0].inf, name=f"theta_dot_guard1")
    gurobi_model.addConstr(input[3] <= theta_dot_interval[0].sup, name=f"theta_dot_guard2")
    gurobi_model.update()
    gurobi_model.optimize()
    # assert gurobi_model.status == 2, "LP wasn't optimally solved"
    return gurobi_model.status == 2


def post(x, nn, output_flag, t, template):
    post = []
    for action in range(2):
        sin_cos_table = get_sin_cos_table(action=action)
        for angle_interval, theta_dot, thetaacc_interval, xacc_interval in sin_cos_table:
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            input = generate_input_region(gurobi_model, template, x)
            feasible = generate_angle_guard(gurobi_model, input, angle_interval, theta_dot)
            if feasible:
                # sintheta = gurobi_model.addMVar(shape=(1,), lb=sin_interval[0].inf, ub=sin_interval[0].sup, name="sin_theta")
                # costheta = gurobi_model.addMVar(shape=(1,), lb=cos_interval[0].inf, ub=cos_interval[0].sup, name="cos_theta")
                thetaacc = gurobi_model.addMVar(shape=(1,), lb=thetaacc_interval[0].inf, ub=thetaacc_interval[0].sup, name="thetaacc")
                xacc = gurobi_model.addMVar(shape=(1,), lb=xacc_interval[0].inf, ub=xacc_interval[0].sup, name="xacc")
                feasible = generate_nn_guard(gurobi_model, input, nn, action_ego=action)
                if feasible:
                    # apply dynamic
                    x_prime = apply_dynamic(input, gurobi_model, action, thetaacc=thetaacc, xacc=xacc)
                    found_successor, x_prime_results = h_repr_to_plot(gurobi_model, template, x_prime)
                    if found_successor:
                        post.append(tuple(x_prime_results))
    return post


def windowed_projection(template, x_results, template_2d):
    ub_lb_window_boundaries = np.array([1000, 1000, -100, -100])
    window_A, window_b = create_window_boundary(template, x_results, template_2d, ub_lb_window_boundaries)
    vertices, rays = pypoman.projection.project_polyhedron((template_2d, np.array([0, 0])), (window_A, window_b), canonicalize=False)
    vertices = np.vstack(vertices)
    return vertices


def get_template(mode=0):
    x = e(env_input_size, 0)
    x_dot = e(env_input_size, 1)
    theta = e(env_input_size, 2)
    theta_dot = e(env_input_size, 3)
    if mode == 0:  # box directions with intervals
        input_boundaries = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        # optimise in a direction
        template = []
        for dimension in range(env_input_size):
            template.append(e(env_input_size, dimension))
            template.append(-e(env_input_size, dimension))
        template = np.array(template)  # the 6 dimensions in 2 variables
        return input_boundaries, template
    if mode == 1:  # directions to easily find fixed point
        input_boundaries = [20]
        template = np.array([x, x_dot])
        return input_boundaries, template


def e(n, i):
    result = [0] * n
    result[i] = 1
    return np.array(result)


def h_repr_to_plot(gurobi_model, template, x_prime):
    x_prime_results = optimise(template, gurobi_model, x_prime)  # h representation
    return x_prime_results is not None, x_prime_results


if __name__ == '__main__':
    main()

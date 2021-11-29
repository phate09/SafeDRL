import math
import time
from collections import defaultdict
from typing import Tuple, List

import gurobi as grb
import numpy as np
import progressbar
import ray
import torch
from interval import interval, imath

from environment.cartpole_ray import CartPoleEnv
from polyhedra.plot_utils import show_polygon_list3
from training.ppo.train_PPO_cartpole import get_PPO_trainer
from training.ray_utils import convert_ray_policy_to_sequential

env_input_size = 4


def generate_input_region(gurobi_model, templates, boundaries):
    input = gurobi_model.addMVar(shape=env_input_size, lb=float("-inf"), ub=float("inf"), name="input")
    generate_region_constraints(gurobi_model, templates, input, boundaries)
    return input


def generate_region_constraints(gurobi_model, templates, input, boundaries):
    for j, template in enumerate(templates):
        gurobi_model.update()
        multiplication = 0
        for i in range(env_input_size):
            multiplication += template[i] * input[i]
        gurobi_model.addConstr(multiplication <= boundaries[j], name=f"input_constr_{j}")


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

    tau = 0.001  # seconds between state updates
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


def get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action):
    assert min_theta < max_theta
    assert min_theta_dot < max_theta_dot
    step_theta = 0.1
    step_theta_dot = 0.1
    min_theta = max(min_theta, -math.pi / 2)
    max_theta = min(max_theta, math.pi / 2)
    split_theta1 = np.arange(min(min_theta, 0), min(max_theta, 0), step_theta)
    split_theta2 = np.arange(max(min_theta, 0), max(max_theta, 0), step_theta)
    # if len(split_theta1) == 0:
    #     split_theta = np.array([min_theta])
    split_theta = np.concatenate([split_theta1, split_theta2])
    split_theta_dot1 = np.arange(min(min_theta_dot, 0), min(max_theta_dot, 0), step_theta)
    split_theta_dot2 = np.arange(max(min_theta_dot, 0), max(max_theta_dot, 0), step_theta)
    # if len(split_theta_dot) == 0:
    #     split_theta_dot = np.array([min_theta_dot])
    split_theta_dot = np.concatenate([split_theta_dot1, split_theta_dot2])
    # split_theta, step_theta = np.linspace(start_theta, end_theta, endpoint=False, retstep=True, num=n_split)
    # split_theta_dot, step_theta_dot = np.linspace(start_theta_dot, end_theta_dot, endpoint=False, retstep=True, num=n_split)
    env = CartPoleEnv(None)
    force = env.force_mag if action == 1 else -env.force_mag

    split = []
    for t_dot in split_theta_dot:
        for theta in split_theta:
            lb_theta_dot = t_dot
            ub_theta_dot = min(t_dot + step_theta_dot, max_theta_dot)
            lb = theta
            ub = min(theta + step_theta, max_theta)
            split.append((interval([lb_theta_dot, ub_theta_dot]), interval([lb, ub])))
    sin_cos_table = []
    while (len(split)):
        theta_dot, theta = split.pop()
        sintheta = imath.sin(theta)
        costheta = imath.cos(theta)
        temp = (force + env.polemass_length * theta_dot ** 2 * sintheta) / env.total_mass
        thetaacc: interval = (env.gravity * sintheta - costheta * temp) / (env.length * (4.0 / 3.0 - env.masspole * costheta ** 2 / env.total_mass))
        xacc = temp - env.polemass_length * thetaacc * costheta / env.total_mass
        if thetaacc[0].sup - thetaacc[0].inf > 0.3:
            # split theta theta_dot
            mid_theta = (theta[0].sup + theta[0].inf) / 2
            mid_theta_dot = (theta_dot[0].sup + theta_dot[0].inf) / 2
            theta_1 = interval([theta[0].inf, mid_theta])
            theta_2 = interval([mid_theta, theta[0].sup])
            theta_dot_1 = interval([theta_dot[0].inf, mid_theta_dot])
            theta_dot_2 = interval([mid_theta_dot, theta_dot[0].sup])
            split.append((theta_1, theta_dot_1))
            split.append((theta_1, theta_dot_2))
            split.append((theta_2, theta_dot_1))
            split.append((theta_2, theta_dot_2))
        else:
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
    return np.array(results)


def check_unsafe(template, bnds, unsafe_zone):
    for A, b in unsafe_zone:
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', False)
        input = gurobi_model.addMVar(shape=(env_input_size,), lb=float("-inf"), name="input")
        generate_region_constraints(gurobi_model, template, input, bnds)
        generate_region_constraints(gurobi_model, A, input, b)
        gurobi_model.update()
        gurobi_model.optimize()
        if gurobi_model.status == 2:
            return True
    return False


def main():
    ray.init(local_mode=False)
    config, trainer = get_PPO_trainer(use_gpu=0)
    # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-07_12-49-16sn6s0bd0/checkpoint_19/checkpoint-19")
    # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-07_17-13-476oom2etf/checkpoint_20/checkpoint-20")
    # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-08_16-19-23tg3bxrcz/checkpoint_18/checkpoint-18")
    # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-09_10-42-12ad16ozkq/checkpoint_150/checkpoint-150")
    # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-09_10-42-12ad16ozkq/checkpoint_170/checkpoint-170")
    # trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-09_10-42-12ad16ozkq/checkpoint_200/checkpoint-200")
    trainer.restore("/home/edoardo/ray_results/PPO_CartPoleEnv_2021-01-09_15-34-25f0ld3dex/checkpoint_30/checkpoint-30")

    policy = trainer.get_policy()
    # sequential_nn = convert_ray_simple_policy_to_sequential(policy).cpu()
    sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
    l0 = torch.nn.Linear(4, 2, bias=False)
    l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32))
    layers = [l0]
    for l in sequential_nn:
        layers.append(l)
    ray.shutdown()
    nn = torch.nn.Sequential(*layers)
    template_2d = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    output_flag = False
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', output_flag)
    input_boundaries, template = get_template(0)

    input = generate_input_region(gurobi_model, template, input_boundaries)
    _, template = get_template(1)
    x_results = optimise(template, gurobi_model, input)
    if x_results is None:
        print("Model unsatisfiable")
        return
    root = tuple(x_results)
    max_t, num_already_visited, vertices_list = main_loop(nn, template, output_flag, root, template_2d)
    print(f"T={max_t}")
    print(f"The algorithm skipped {num_already_visited} already visited states")
    # show_polygon_list2(vertices_list, "theta_dot", "theta")  # show_polygon_list2(vertices_list)


def main_loop(nn, template, output_flag, root, template_2d):
    vertices_list = defaultdict(list)
    ray.init(local_mode=False, log_to_driver=False)
    seen = []
    frontier = [(0, root)]
    max_t = 0
    num_already_visited = 0
    safe_angle = 0.5  # 12 * 2 * math.pi / 360
    theta = [e(4, 2)]
    neg_theta = [-e(4, 2)]
    unsafe_zone = [(neg_theta, np.array([-safe_angle]))]  # , (theta, np.array([-safe_angle]))
    widgets = [progressbar.Variable('n_workers'), ', ', progressbar.Variable('frontier'), ', ', progressbar.Variable('seen'), ', ', progressbar.Variable('num_already_visited'), ", ",
               progressbar.Variable('max_t'), ", ", progressbar.Variable('last_visited_state')]
    proc_ids = []
    n_workers = 8
    last_time_plot = None
    with progressbar.ProgressBar(widgets=widgets) as bar:
        while len(frontier) != 0 or len(proc_ids) != 0:
            while len(proc_ids) < n_workers and len(frontier) != 0:
                t, x = frontier.pop(0)
                if max_t > 100:
                    break
                if any([contained(x, s) for s in seen]):
                    num_already_visited += 1
                    continue
                max_t = max(max_t, t)
                vertices_list[t].append(np.array(x))
                if check_unsafe(template, x, unsafe_zone):
                    print(f"Unsafe state found at timestep t={t}")
                    print(x)
                    return max_t, num_already_visited, vertices_list

                seen.append(x)

                proc_ids.append(post_milp_remote.remote(x, nn, output_flag, t, template))
            if last_time_plot is None or time.time() - last_time_plot >= 60 * 2:
                show_polygon_list3(vertices_list, "theta", "theta_dot", template, template_2d)
                last_time_plot = time.time()
            bar.update(value=bar.value + 1, n_workers=len(proc_ids), seen=len(seen), frontier=len(frontier), num_already_visited=num_already_visited, last_visited_state=str(x), max_t=max_t)
            ready_ids, proc_ids = ray.wait(proc_ids, num_returns=len(proc_ids), timeout=0.5)
            x_primes_list = ray.get(ready_ids)
            for x_primes in x_primes_list:
                for x_prime in x_primes:
                    x_prime = tuple(np.ceil(np.array(x_prime) * 256) / 256)  # todo should we round to prevent numerical errors?
                    frontier = [(u, y) for u, y in frontier if not contained(y, x_prime)]
                    if not any([contained(x_prime, y) for u, y in frontier]):
                        frontier.append(((t + 1), x_prime))
                        # print(x_prime)
                    else:
                        num_already_visited += 1
    show_polygon_list3(vertices_list, "theta", "theta_dot", template, template_2d)  # show_polygon_list2(vertices_list)
    return max_t, num_already_visited, vertices_list


@ray.remote
def post_milp_remote(x, nn, output_flag, t, template):
    return post_milp(x, nn, output_flag, t, template)


def contained(x: tuple, y: tuple):
    # y contains x
    assert len(x) == len(y)
    for i in range(len(x)):
        if x[i] > y[i]:
            return False
    return True


def generate_angle_guard(gurobi_model: grb.Model, input, theta_interval, theta_dot_interval):
    eps = 1e-6
    gurobi_model.addConstr(input[2] >= theta_interval[0].inf, name=f"theta_guard1")
    gurobi_model.addConstr(input[2] <= theta_interval[0].sup, name=f"theta_guard2")
    gurobi_model.addConstr(input[3] >= theta_dot_interval[0].inf, name=f"theta_dot_guard1")
    gurobi_model.addConstr(input[3] <= theta_dot_interval[0].sup, name=f"theta_dot_guard2")
    gurobi_model.update()
    gurobi_model.optimize()
    # assert gurobi_model.status == 2, "LP wasn't optimally solved"
    return gurobi_model.status == 2


def generate_angle_milp(gurobi_model: grb.Model, input, sin_cos_table: List[Tuple]):
    """MILP method
    input: theta, thetadot
    output: thetadotdot, xdotdot (edited)
    l_{theta, i}, l_{thatdot,i}, l_{thetadotdot, i}, l_{xdotdot, i}, u_....
    sum_{i=1}^k l_{x,i} - l_{x,i}*z_i <= x <= sum_{i=1}^k u_{x,i} - u_{x,i}*z_i, per ogni variabile x
    sum_{i=1}^k l_{theta,i} - l_{theta,i}*z_i <= theta <= sum_{i=1}^k u_{theta,i} - u_{theta,i}*z_i
    """
    theta = input[2]
    theta_dot = input[3]
    k = len(sin_cos_table)
    zs = []
    thetaacc = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name="thetaacc")
    xacc = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name="xacc")
    for i in range(k):
        z = gurobi_model.addMVar(lb=0, ub=1, shape=(1,), vtype=grb.GRB.INTEGER, name=f"part_{i}")
        zs.append(z)
    gurobi_model.addConstr(k - 1 == sum(zs), name=f"const_milp1")
    theta_lb = 0
    theta_ub = 0
    theta_dot_lb = 0
    theta_dot_ub = 0
    thetaacc_lb = 0
    thetaacc_ub = 0
    xacc_lb = 0
    xacc_ub = 0
    for i in range(k):
        theta_interval, theta_dot_interval, theta_acc_interval, xacc_interval = sin_cos_table[i]
        theta_lb += theta_interval[0].inf - theta_interval[0].inf * zs[i]
        theta_ub += theta_interval[0].sup - theta_interval[0].sup * zs[i]
        theta_dot_lb += theta_dot_interval[0].inf - theta_dot_interval[0].inf * zs[i]
        theta_dot_ub += theta_dot_interval[0].sup - theta_dot_interval[0].sup * zs[i]

        thetaacc_lb += theta_acc_interval[0].inf - theta_acc_interval[0].inf * zs[i]
        thetaacc_ub += theta_acc_interval[0].sup - theta_acc_interval[0].sup * zs[i]

        xacc_lb += xacc_interval[0].inf - xacc_interval[0].inf * zs[i]
        xacc_ub += xacc_interval[0].sup - xacc_interval[0].sup * zs[i]

    gurobi_model.addConstr(theta >= theta_lb, name=f"theta_guard1")
    gurobi_model.addConstr(theta <= theta_ub, name=f"theta_guard2")
    gurobi_model.addConstr(theta_dot >= theta_dot_lb, name=f"theta_dot_guard1")
    gurobi_model.addConstr(theta_dot <= theta_dot_ub, name=f"theta_dot_guard2")

    gurobi_model.addConstr(thetaacc >= thetaacc_lb, name=f"thetaacc_guard1")
    gurobi_model.addConstr(thetaacc <= thetaacc_ub, name=f"thetaacc_guard2")
    gurobi_model.addConstr(xacc >= xacc_lb, name=f"xacc_guard1")
    gurobi_model.addConstr(xacc <= xacc_ub, name=f"xacc_guard2")

    gurobi_model.update()
    gurobi_model.optimize()
    # assert gurobi_model.status == 2, "LP wasn't optimally solved"
    return thetaacc, xacc


#
# def post(x, nn, output_flag, t, template):
#     post = []
#     max_theta, min_theta, max_theta_dot, min_theta_dot = get_theta_bounds(output_flag, template, x)
#     for action in range(2):
#         sin_cos_table = get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action=action)
#         for angle_interval, theta_dot, thetaacc_interval, xacc_interval in sin_cos_table:
#             gurobi_model = grb.Model()
#             gurobi_model.setParam('OutputFlag', output_flag)
#             input = generate_input_region(gurobi_model, template, x)
#             feasible_angle = generate_angle_guard(gurobi_model, input, angle_interval, theta_dot)
#             if feasible_angle:
#                 # sintheta = gurobi_model.addMVar(shape=(1,), lb=sin_interval[0].inf, ub=sin_interval[0].sup, name="sin_theta")
#                 # costheta = gurobi_model.addMVar(shape=(1,), lb=cos_interval[0].inf, ub=cos_interval[0].sup, name="cos_theta")
#                 thetaacc = gurobi_model.addMVar(shape=(1,), lb=thetaacc_interval[0].inf, ub=thetaacc_interval[0].sup, name="thetaacc")
#                 xacc = gurobi_model.addMVar(shape=(1,), lb=xacc_interval[0].inf, ub=xacc_interval[0].sup, name="xacc")
#                 feasible_action = generate_nn_guard(gurobi_model, input, nn, action_ego=action)
#                 if feasible_action:
#                     # apply dynamic
#                     x_prime = apply_dynamic(input, gurobi_model, action, thetaacc=thetaacc, xacc=xacc)
#                     found_successor, x_prime_results = h_repr_to_plot(gurobi_model, template, x_prime)
#                     if found_successor:
#                         post.append(tuple(x_prime_results))
#     return post


def post_milp(x, nn, output_flag, t, template):
    """milp method"""
    post = []

    # positive case action 0
    # chosen_action = expected_action if case == 0 else (expected_action + 1) % 2  # invert the action in the case where the angle is negative
    chosen_action = 0
    case = 0  # positive

    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', output_flag)
    gurobi_model.setParam('Threads', 2)
    input = generate_input_region(gurobi_model, template, x)
    gurobi_model.addConstr(input[3] <= 2)
    gurobi_model.addConstr(input[3] >= -2)
    gurobi_model.addConstr(input[2] >= -math.pi / 2)
    gurobi_model.addConstr(input[2] <= math.pi / 2)
    max_theta, min_theta, max_theta_dot, min_theta_dot = get_theta_bounds(gurobi_model, input)
    sin_cos_table = get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action=chosen_action)
    # gurobi_model.addConstr(input[2] >= 0)  # positive case
    feasible_action = generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
    if feasible_action:
        thetaacc, xacc = generate_angle_milp(gurobi_model, input, sin_cos_table)
        # apply dynamic
        x_prime = apply_dynamic(input, gurobi_model, chosen_action, thetaacc=thetaacc, xacc=xacc)
        # gurobi_model.addConstr(x_prime[3] <= 0.625)
        gurobi_model.addConstr(x_prime[3] <= 2)
        gurobi_model.addConstr(x_prime[3] >= -2)
        gurobi_model.addConstr(x_prime[2] >= -math.pi / 2)
        gurobi_model.addConstr(x_prime[2] <= math.pi / 2)
        gurobi_model.update()
        gurobi_model.optimize()
        found_successor, x_prime_results = h_repr_to_plot(gurobi_model, template, x_prime)
        if found_successor:
            post.append(tuple(x_prime_results))

    # positive case action 1
    chosen_action = 1
    case = 0  # positive
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', output_flag)
    gurobi_model.setParam('Threads', 2)
    input = generate_input_region(gurobi_model, template, x)
    gurobi_model.addConstr(input[3] <= 2)
    gurobi_model.addConstr(input[3] >= -2)
    gurobi_model.addConstr(input[2] >= -math.pi / 2)
    gurobi_model.addConstr(input[2] <= math.pi / 2)
    max_theta, min_theta, max_theta_dot, min_theta_dot = get_theta_bounds(gurobi_model, input)
    sin_cos_table = get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action=chosen_action)
    # gurobi_model.addConstr(input[2] >= 0)  # positive case
    feasible_action = generate_nn_guard(gurobi_model, input, nn, action_ego=chosen_action)
    if feasible_action:
        thetaacc, xacc = generate_angle_milp(gurobi_model, input, sin_cos_table)
        # apply dynamic
        x_prime = apply_dynamic(input, gurobi_model, chosen_action, thetaacc=thetaacc, xacc=xacc)
        gurobi_model.addConstr(x_prime[3] <= 2)
        gurobi_model.addConstr(x_prime[3] >= -2)
        gurobi_model.addConstr(x_prime[2] >= -math.pi / 2)
        gurobi_model.addConstr(x_prime[2] <= math.pi / 2)
        gurobi_model.update()
        gurobi_model.optimize()
        found_successor, x_prime_results = h_repr_to_plot(gurobi_model, template, x_prime)
        if found_successor:
            post.append(tuple(x_prime_results))

    # # negative case action 0
    # chosen_action = 0
    # executed_action = 1
    # sin_cos_table = get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action=chosen_action)
    # gurobi_model = grb.Model()
    # gurobi_model.setParam('OutputFlag', output_flag)
    # gurobi_model.setParam('Threads', 2)
    # input = generate_input_region(gurobi_model, template, x)
    # gurobi_model.addConstr(input[2] <= 0)  # negative case
    # reverse_input = make_reverse_var(gurobi_model, input)
    # feasible_action = generate_nn_guard(gurobi_model, reverse_input, nn, action_ego=chosen_action)
    # if feasible_action:
    #     thetaacc, xacc = generate_angle_milp(gurobi_model, input, sin_cos_table)
    #     # apply dynamic
    #     x_prime = apply_dynamic(input, gurobi_model, executed_action, thetaacc=thetaacc, xacc=xacc)
    #     found_successor, x_prime_results = h_repr_to_plot(gurobi_model, template, x_prime)
    #     if found_successor:
    #         post.append(tuple(x_prime_results))
    #
    # # negative case action 1
    # chosen_action = 1
    # executed_action = 0
    # sin_cos_table = get_sin_cos_table(max_theta, min_theta, max_theta_dot, min_theta_dot, action=chosen_action)
    # gurobi_model = grb.Model()
    # gurobi_model.setParam('OutputFlag', output_flag)
    # gurobi_model.setParam('Threads', 2)
    # input = generate_input_region(gurobi_model, template, x)
    # gurobi_model.addConstr(input[2] <= 0)  # negative case
    # reverse_input = make_reverse_var(gurobi_model, input)
    # feasible_action = generate_nn_guard(gurobi_model, reverse_input, nn, action_ego=chosen_action)
    # if feasible_action:
    #     thetaacc, xacc = generate_angle_milp(gurobi_model, input, sin_cos_table)
    #     # apply dynamic
    #     x_prime = apply_dynamic(input, gurobi_model, executed_action, thetaacc=thetaacc, xacc=xacc)
    #     found_successor, x_prime_results = h_repr_to_plot(gurobi_model, template, x_prime)
    #     if found_successor:
    #         post.append(tuple(x_prime_results))

    return post


def make_reverse_var(gurobi_model, input):
    z = gurobi_model.addMVar(shape=env_input_size, lb=float("-inf"), name="input_positive")
    gurobi_model.addConstr(z[0] == input[0])
    gurobi_model.addConstr(z[1] == input[1])
    gurobi_model.addConstr(z[2] == -input[2])
    gurobi_model.addConstr(z[3] == input[3])
    gurobi_model.update()
    return z


def make_positive_case(gurobi_model, input, case):
    if case == 0:
        gurobi_model.addConstr(input[2] >= 0)
        return True, input
    else:
        gurobi_model.addConstr(input[2] <= 0)  # mirror input in case of negative angle
        z = gurobi_model.addMVar(shape=env_input_size, lb=float("-inf"), name="input_positive")
        gurobi_model.addConstr(z[0] == -input[0])
        gurobi_model.addConstr(z[1] == -input[1])
        gurobi_model.addConstr(z[2] == -input[2])
        gurobi_model.addConstr(z[3] == -input[3])

    gurobi_model.update()
    gurobi_model.optimize()
    return gurobi_model.status == 2, z


def get_theta_bounds(gurobi_model, input):
    gurobi_model.setObjective(input[2].sum(), grb.GRB.MAXIMIZE)
    gurobi_model.optimize()
    max_theta = gurobi_model.getVars()[2].X

    gurobi_model.setObjective(input[2].sum(), grb.GRB.MINIMIZE)
    gurobi_model.optimize()
    min_theta = gurobi_model.getVars()[2].X

    gurobi_model.setObjective(input[3].sum(), grb.GRB.MAXIMIZE)
    gurobi_model.optimize()
    max_theta_dot = gurobi_model.getVars()[3].X

    gurobi_model.setObjective(input[3].sum(), grb.GRB.MINIMIZE)
    gurobi_model.optimize()
    min_theta_dot = gurobi_model.getVars()[3].X
    return max_theta, min_theta, max_theta_dot, min_theta_dot


def get_template(mode=0):
    x = e(env_input_size, 0)
    x_dot = e(env_input_size, 1)
    theta = e(env_input_size, 2)
    theta_dot = e(env_input_size, 3)
    if mode == 0:  # box directions with intervals
        # input_boundaries = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        input_boundaries = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        # input_boundaries = [0.04373426, -0.04373426, -0.04980056, 0.04980056, 0.045, -0.045, -0.51, 0.51]
        # optimise in a direction
        template = []
        for dimension in range(env_input_size):
            template.append(e(env_input_size, dimension))
            template.append(-e(env_input_size, dimension))
        template = np.array(template)  # the 6 dimensions in 2 variables
        return input_boundaries, template
    if mode == 1:  # directions to easily find fixed point
        input_boundaries = [20]
        template = np.array([theta, -theta, theta_dot, -theta_dot, theta + theta_dot, -(theta + theta_dot), (theta - theta_dot), -(theta - theta_dot)])  # x_dot, -x_dot,theta_dot - theta
        return input_boundaries, template
    if mode == 2:
        input_boundaries = None
        template = np.array([theta, -theta, theta_dot, -theta_dot])
        return input_boundaries, template
    if mode == 3:
        input_boundaries = None
        template = np.array([theta, theta_dot, -theta_dot])
        return input_boundaries, template
    if mode == 4:
        input_boundaries = [0.09375, 0.625, 0.625, 0.0625, 0.1875]
        # input_boundaries = [0.09375, 0.5, 0.5, 0.0625, 0.09375]
        template = np.array([theta, theta_dot, -theta_dot, theta + theta_dot, (theta - theta_dot)])
        return input_boundaries, template
    if mode == 5:
        input_boundaries = [0.125, 0.0625, 0.1875]
        template = np.array([theta, theta + theta_dot, (theta - theta_dot)])
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

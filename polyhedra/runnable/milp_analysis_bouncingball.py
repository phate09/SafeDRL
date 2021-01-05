from collections import defaultdict

import gurobi as grb
import numpy as np
import pypoman
import ray
import torch

from agents.ppo.train_PPO_bouncingball import get_PPO_trainer
from agents.ray_utils import convert_ray_policy_to_sequential
from polyhedra.plot_utils import show_polygon_list2

env_input_size = 2


def generate_input_region(gurobi_model, templates, boundaries):
    input = gurobi_model.addMVar(shape=env_input_size, lb=float("-inf"), name="input")
    for j, template in enumerate(templates):
        gurobi_model.update()
        multiplication = 0
        for i in range(env_input_size):
            multiplication += template[i] * input[i]
        gurobi_model.addConstr(multiplication <= boundaries[j], name=f"input_constr_{j}")
    return input


def generate_guard(gurobi_model: grb.Model, input, case=0):
    eps = 1e-6
    if case == 0:  # v <= 0 && p <= 0
        gurobi_model.addConstr(input[1] <= 0, name=f"cond1")
        gurobi_model.addConstr(input[0] <= 0, name=f"cond2")
    if case == 1:  # v_prime <= 0 and p_prime > 4
        gurobi_model.addConstr(input[1] <= 0, name=f"cond1")
        gurobi_model.addConstr(input[0] >= 4, name=f"cond2")
    if case == 2:  # v_prime > 0 and p_prime > 4
        gurobi_model.addConstr(input[1] >= 0, name=f"cond1")
        gurobi_model.addConstr(input[0] >= 4, name=f"cond2")
    if case == 3:  # ball out of reach and not bounce
        gurobi_model.addConstr(input[0] <= 4 - eps, name=f"cond1")
        gurobi_model.addConstr(input[0] >= 0 + eps, name=f"cond2")
    gurobi_model.update()
    gurobi_model.optimize()
    # assert gurobi_model.status == 2, "LP wasn't optimally solved"
    return gurobi_model.status == 2


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
    last_layer = gurobi_vars[-1]
    if action_ego == 0:
        gurobi_model.addConstr(last_layer[0] >= last_layer[1], name="last_layer")
    else:
        gurobi_model.addConstr(last_layer[1] >= last_layer[0], name="last_layer")
    gurobi_model.update()
    gurobi_model.optimize()
    # assert gurobi_model.status == 2, "LP wasn't optimally solved"
    return gurobi_model.status == 2


def apply_dynamic(input, gurobi_model: grb.Model):
    '''

    :param input:
    :param gurobi_model:
    :param t:
    :return:
    '''

    p = input[0]
    v = input[1]
    dt = 0.1
    z = gurobi_model.addMVar(shape=(env_input_size,), lb=float("-inf"), name=f"x_prime")
    # pos_max = gurobi_model.addMVar(shape=(1,), lb=float("-inf"), name=f"pos_max")
    v_second = v - 9.81 * dt
    p_second = p + dt * v_second
    gurobi_model.addConstr(z[1] == v_second, name=f"dyna_constr_1")
    # gurobi_model.addConstr(pos_max == grb.max_([p_second, 0]), name=f"dyna_constr_2")
    # gurobi_model.addConstr(z[1] == p_second, name=f"dyna_constr_2")
    max_switch = gurobi_model.addMVar(lb=0, ub=1, shape=p_second.shape, vtype=grb.GRB.INTEGER, name=f"max_switch")
    M = 10e6
    # gurobi_model.addConstr(v == grb.max_(0, gurobi_vars[-1]))
    gurobi_model.addConstr(z[0] >= p_second)
    gurobi_model.addConstr(z[0] <= p_second + M * max_switch)
    gurobi_model.addConstr(z[0] >= 0)
    gurobi_model.addConstr(z[0] <= M - M * max_switch)

    return z


def apply_dynamic2(input_prime, gurobi_model: grb.Model, case):
    p_prime = input_prime[0]
    v_prime = input_prime[1]
    z = gurobi_model.addMVar(shape=(env_input_size,), lb=float("-inf"), name=f"x_prime")
    v_second = v_prime
    p_second = p_prime
    if case == 0:  # v <= 0 && p <= 0
        v_second = -(0.90) * v_prime
        p_second = 0
    if case == 1:  # v <= 0 && p >= 4 && action = 1
        v_second = v_prime - 4
        p_second = 4
    if case == 2:  # v >=0 && p >= 4 && action = 1
        v_second = -(0.9) * v_prime - 4
        p_second = 4
    if case == 3:  # p>=0
        v_second = v_prime
        p_second = p_prime

    gurobi_model.addConstr(z[1] == v_second, name=f"dyna_constr2_1")
    gurobi_model.addConstr(z[0] == p_second, name=f"dyna_constr2_2")
    return z


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
    return window_template, window_boundaries


def main():
    output_flag = False
    ray.init(local_mode=True)
    config, trainer = get_PPO_trainer(use_gpu=0)
    trainer.restore("/home/edoardo/ray_results/PPO_BouncingBall_2021-01-04_18-58-32smp2ln1g/checkpoint_272/checkpoint-272")
    policy = trainer.get_policy()
    sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
    layers = []
    for l in sequential_nn:
        layers.append(l)
    nn = torch.nn.Sequential(*layers)
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', output_flag)
    input_boundaries, template = get_template(0)

    start1 = np.array([9, -8, 0, 0.1])
    # start2 = np.array([0, 0, -7, 10])
    template_2d = np.array([[0, 1], [1, 0]])
    vertices_list = defaultdict(list)
    seen = []
    frontier = []
    frontier.append((0, tuple(start1)))
    # frontier.append((0, tuple(start2)))
    max_t = 0
    while len(frontier) != 0:
        t, x = frontier.pop()
        max_t = max(max_t, t)
        if t > 500:
            break
        if any([contained(x, s) for s in seen]):
            continue
        vertices = windowed_projection(template, np.array(x), template_2d)
        vertices_list[t].append(vertices)
        seen.append(x)
        x_primes = post(x, nn, output_flag, t, template)
        # for v in vertices_container:
        #     vertices_time.append((t + 1, v[0, 0]))
        for x_prime in x_primes:
            x_prime = tuple(np.array(x_prime).round(4))  # todo should we round to prevent numerical errors?
            frontier = [(u, y) for u, y in frontier if not contained(y, x_prime)]
            if not any([contained(x_prime, y) for u, y in frontier]):
                frontier.append(((t + 1), x_prime))

    print(f"T={max_t}")
    show_polygon_list2(vertices_list, "p", "v")  # show_polygon_list2(vertices_list)
    # import plotly.graph_objects as go
    # fig = go.Figure()
    # trace1 = go.Scatter(x=[x[0] for x in vertices_time], y=[x[1] for x in vertices_time], mode='markers', )
    # fig.add_trace(trace1)
    # fig.show()


def contained(x: tuple, y: tuple):
    # y contains x
    assert len(x) == len(y)
    for i in range(len(x)):
        if x[i] > y[i]:
            return False
    return True


def post(x, nn, output_flag, t, template):
    post = []

    def standard_op():
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        input = generate_input_region(gurobi_model, template, x)
        z = apply_dynamic(input, gurobi_model)
        return gurobi_model, z, input

    # case 0
    gurobi_model, z, input = standard_op()
    feasible0 = generate_guard(gurobi_model, z, case=0)  # bounce
    if feasible0:  # action is irrelevant in this case
        # apply dynamic
        x_prime = apply_dynamic2(z, gurobi_model, case=0)
        found_successor, x_prime_results = h_repr_to_plot(gurobi_model, template, x_prime)
        if found_successor:
            post.append(tuple(x_prime_results))

    # case 1 : ball going down and hit
    gurobi_model, z, input = standard_op()
    feasible11 = generate_guard(gurobi_model, z, case=1)
    if feasible11:
        feasible12 = generate_nn_guard(gurobi_model, input, nn, action_ego=1)  # check for action =1 over input (not z!)
        if feasible12:
            # apply dynamic
            x_prime = apply_dynamic2(z, gurobi_model, case=1)
            found_successor, x_prime_results = h_repr_to_plot(gurobi_model, template, x_prime)
            if found_successor:
                post.append(tuple(x_prime_results))
    # case 2 : ball going up and hit
    gurobi_model, z, input = standard_op()
    feasible21 = generate_guard(gurobi_model, z, case=2)
    if feasible21:
        feasible22 = generate_nn_guard(gurobi_model, input, nn, action_ego=1)  # check for action =1 over input (not z!)
        if feasible22:
            # apply dynamic
            x_prime = apply_dynamic2(z, gurobi_model, case=2)
            found_successor, x_prime_results = h_repr_to_plot(gurobi_model, template, x_prime)
            if found_successor:
                post.append(tuple(x_prime_results))
    # case 1 alt : ball going down and NO hit
    gurobi_model, z, input = standard_op()
    feasible11_alt = generate_guard(gurobi_model, z, case=1)
    if feasible11_alt:
        feasible12_alt = generate_nn_guard(gurobi_model, input, nn, action_ego=0)  # check for action = 0 over input (not z!)
        if feasible12_alt:
            # apply dynamic
            x_prime = apply_dynamic2(z, gurobi_model, case=3)  # normal dynamic
            found_successor, x_prime_results = h_repr_to_plot(gurobi_model, template, x_prime)
            if found_successor:
                post.append(tuple(x_prime_results))
    # case 2 alt : ball going up and NO hit
    gurobi_model, z, input = standard_op()
    feasible21_alt = generate_guard(gurobi_model, z, case=2)
    if feasible21_alt:
        feasible22_alt = generate_nn_guard(gurobi_model, input, nn, action_ego=0)  # check for action = 0 over input (not z!)
        if feasible22_alt:
            # apply dynamic
            x_prime = apply_dynamic2(z, gurobi_model, case=3)  # normal dynamic
            found_successor, x_prime_results = h_repr_to_plot(gurobi_model, template, x_prime)
            if found_successor:
                post.append(tuple(x_prime_results))
    # case 3 : ball out of reach and not bounce
    gurobi_model, z, input = standard_op()
    feasible3 = generate_guard(gurobi_model, z, case=3)  # out of reach
    if feasible3:  # action is irrelevant in this case
        # apply dynamic
        x_prime = apply_dynamic2(z, gurobi_model, case=3)  # normal dynamic
        found_successor, x_prime_results = h_repr_to_plot(gurobi_model, template, x_prime)
        if found_successor:
            post.append(tuple(x_prime_results))

    return post


def windowed_projection(template, x_results: np.ndarray, template_2d):
    ub_lb_window_boundaries = np.array([1000, 1000, -100, -100])
    window_A, window_b = create_window_boundary(template, x_results, template_2d, ub_lb_window_boundaries)
    vertices, rays = pypoman.projection.project_polyhedron((template_2d, np.array([0, 0])), (window_A, window_b), canonicalize=False)
    vertices = np.vstack(vertices)
    return vertices


def get_template(mode=0):
    p = e(env_input_size, 0)
    v = e(env_input_size, 1)
    if mode == 0:  # box directions with intervals
        input_boundaries = [0, 0, 10, 10]
        # optimise in a direction
        template = []
        for dimension in range(env_input_size):
            template.append(e(env_input_size, dimension))
            template.append(-e(env_input_size, dimension))
        template = np.array(template)  # the 6 dimensions in 2 variables
        return input_boundaries, template
    if mode == 1:  # directions to easily find fixed point
        input_boundaries = [20]
        template = np.array([v, -v, -p])
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

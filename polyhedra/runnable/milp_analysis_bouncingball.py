import gurobi as grb
import numpy as np
import pypoman
import ray
import torch

from agents.ppo.train_PPO_bouncingball import get_PPO_trainer
from agents.ray_utils import convert_ray_policy_to_sequential
from polyhedra.graph_explorer import GraphExplorer
from polyhedra.net_methods import generate_nn_torch
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


def generate_guard(gurobi_model: grb.Model, input, action_ego=0, case=0):
    if case == 0:  # v <= 0 && p <= 0
        gurobi_model.addConstr(input[0] <= 0, name=f"cond1")
        gurobi_model.addConstr(input[1] <= 0, name=f"cond2")
    if case == 1:
        gurobi_model.addConstr(input[0] <= 0, name=f"cond1")
        gurobi_model.addConstr(input[1] >= 4, name=f"cond2")
    if case == 2:
        gurobi_model.addConstr(input[0] >= 0, name=f"cond1")
        gurobi_model.addConstr(input[1] >= 4, name=f"cond2")
    if case == 3:
        gurobi_model.addConstr(input[1] >= 0, name=f"cond2")
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


def apply_dynamic(input, gurobi_model: grb.Model, action_ego=0, case=0):
    '''

    :param input:
    :param gurobi_model:
    :param action_ego:
    :param t:
    :return:
    '''

    v = input[0]
    p = input[1]
    dt = 0.1
    z = gurobi_model.addMVar(shape=(env_input_size,), lb=float("-inf"), name=f"x_prime")
    if case == 0:  # v <= 0 && p <= 0
        v_prime = -(0.85) * v
        p_prime = 0
    if case == 1:  # v <= 0 && p >= 4 && action = 1
        v_prime = -4
        p_prime = p
    if case == 2:  # v >=0 && p >= 4 && action = 1
        v_prime = -(0.9 + 0.05) * v - 4
        p_prime = p
    if case == 3:  # p>=0
        v_prime = v
        p_prime = p
    v_second = v_prime - 9.81 * dt
    p_second = p_prime + dt * v_prime
    gurobi_model.addConstr(z[0] == v_second, name=f"dyna_constr_1")
    gurobi_model.addConstr(z[1] == p_second, name=f"dyna_constr_2")
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
    # windowed_projection_max = np.maximum(window_boundaries, projection)
    # windowed_projection_min = np.minimum(window_boundaries, projection)
    # windowed_projection = np.concatenate((windowed_projection_max.squeeze(), windowed_projection_min.squeeze()), axis=0)
    return window_template, window_boundaries


def main():
    output_flag = False
    mode = 2  # 0 hardcoded guard, 1 nn guard, 2 trained nn
    if mode == 2:
        ray.init(local_mode=True)
        config, trainer = get_PPO_trainer(use_gpu=0)
        # trainer.restore("/home/edoardo/ray_results/PPO_BouncingBall_2021-01-03_11-52-42e5nav0w_/checkpoint_9/checkpoint-9")
        policy = trainer.get_policy()
        sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
        layers = []
        for l in sequential_nn:
            layers.append(l)

        nn = torch.nn.Sequential(*layers)
    elif mode == 1:
        nn = generate_nn_torch(six_dim=True, min_distance=20, max_distance=22)  # min_speed=30,max_speed=36
    else:
        nn = None
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', output_flag)
    graph = GraphExplorer(None)

    input_boundaries, template = get_template(0)

    input = generate_input_region(gurobi_model, template, input_boundaries)
    # _, template = get_template(1)
    x_results = optimise(template, gurobi_model, input)
    # input_boundaries, template = get_template(1)
    if x_results is None:
        print("Model unsatisfiable")
        return
    root = tuple(x_results)
    graph.root = root
    graph.store_in_fringe(root)
    template_2d = np.array([[1, 0], [0, 1]])
    vertices = windowed_projection(template, x_results, template_2d)
    vertices_time = []
    vertices_list = []
    vertices_list.append([vertices])
    vertices_time = [[0, vertices[0, 1]]]
    # vertices = windowed_projection(template, x_results, template2d_speed)
    # vertices_list.append([vertices])

    # show_polygon_list2(vertices_list)
    seen = []
    frontier = [(0, root)]
    while len(frontier) != 0:
        t, x = frontier.pop()
        if t > 200:
            break
        if any([contained(x, s) for s in seen]):
            continue
        seen.append(x)
        # vertices = windowed_projection(template, x_results)
        # vertices_list.append([vertices])
        vertices_container = []
        x_primes = post(x, graph, mode, nn, output_flag, t, template, vertices_container)
        vertices_list.append(vertices_container)
        for v in vertices_container:
            vertices_time.append((t, v[0, 1]))
        for x_prime in x_primes:

            frontier = [(u, y) for u, y in frontier if not contained(y, x_prime)]
            if not any([contained(x_prime, y) for u, y in frontier]):
                frontier.append(((t + 1), x_prime))

    print(f"T={t}")
    show_polygon_list2(vertices_list, "p", "v")  # show_polygon_list2(vertices_list)
    import plotly.graph_objects as go
    fig = go.Figure()
    trace1 = go.Scatter(x=[x[0] for x in vertices_time], y=[x[1] for x in vertices_time], mode='markers', )
    fig.add_trace(trace1)
    fig.show()


def contained(x: tuple, y: tuple):
    # y contains x
    assert len(x) == len(y)
    for i in range(len(x)):
        if x[i] > y[i]:
            return False
    return True


def post(x, graph, mode, nn, output_flag, t, template, timestep_container):
    post = []
    if mode == 1 or mode == 2:
        # case 0
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        input = generate_input_region(gurobi_model, template, x)
        feasible = generate_guard(gurobi_model, input, action_ego=0, case=0)
        # feasible = generate_nn_guard(gurobi_model, input, nn, action_ego=0)
        if feasible:
            # apply dynamic
            x_prime = apply_dynamic(input, gurobi_model, 0, case=0)
            found_successor, x_prime_results = h_repr_to_plot(graph, gurobi_model, template, timestep_container, x_prime, t)
            if found_successor:
                post.append(tuple(x_prime_results))
        # case 3
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        input = generate_input_region(gurobi_model, template, x)
        feasible = generate_guard(gurobi_model, input, action_ego=0, case=3)
        # feasible = generate_nn_guard(gurobi_model, input, nn, action_ego=0)
        if feasible:
            # apply dynamic
            x_prime = apply_dynamic(input, gurobi_model, 0, case=3)
            found_successor, x_prime_results = h_repr_to_plot(graph, gurobi_model, template, timestep_container, x_prime, t)
            if found_successor:
                post.append(tuple(x_prime_results))
        # case 1
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        input = generate_input_region(gurobi_model, template, x)
        feasible = generate_guard(gurobi_model, input, action_ego=0, case=1)
        feasible = generate_nn_guard(gurobi_model, input, nn, action_ego=1)
        if feasible:
            # apply dynamic
            x_prime = apply_dynamic(input, gurobi_model, 1, case=1)
            found_successor, x_prime_results = h_repr_to_plot(graph, gurobi_model, template, timestep_container, x_prime, t)
            if found_successor:
                post.append(tuple(x_prime_results))
        # case 2
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        input = generate_input_region(gurobi_model, template, x)
        feasible = generate_guard(gurobi_model, input, action_ego=0, case=2)
        feasible = generate_nn_guard(gurobi_model, input, nn, action_ego=1)
        if feasible:
            # apply dynamic
            x_prime = apply_dynamic(input, gurobi_model, 1, case=2)
            found_successor, x_prime_results = h_repr_to_plot(graph, gurobi_model, template, timestep_container, x_prime, t)
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
    v = e(env_input_size, 0)
    p = e(env_input_size, 1)
    if mode == 0:  # box directions with intervals
        input_boundaries = [0, -0, 10, -10]
        # optimise in a direction
        template = []
        for dimension in range(env_input_size):
            template.append(e(env_input_size, dimension))
            template.append(-e(env_input_size, dimension))
        template = np.array(template)  # the 6 dimensions in 2 variables
        return input_boundaries, template
    if mode == 1:  # directions to easily find fixed point
        input_boundaries = [20]
        template = np.array([v, p])
        return input_boundaries, template


def e(n, i):
    result = [0] * n
    result[i] = 1
    return np.array(result)


def h_repr_to_plot(graph, gurobi_model, template, vertices_list, x_prime, t):
    x_prime_results = optimise(template, gurobi_model, x_prime)  # h representation
    added = False
    if x_prime_results is not None:
        x_prime_tuple = tuple(x_prime_results)
        added = graph.store_in_fringe(x_prime_tuple)
        if added:
            template_2d = np.array([[1, 0], [0, 1]])
            vertices = windowed_projection(template, x_prime_results, template_2d)
            vertices_list.append(vertices)
            # template2d_speed = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
            # vertices = windowed_projection(template, x_prime_results, template2d_speed)
            # vertices_list.append(vertices)
    return added, x_prime_results


if __name__ == '__main__':
    main()

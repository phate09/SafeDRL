import numpy as np
import gurobi as grb
import pypoman
import torch
from polyhedra.graph_explorer import GraphExplorer
from polyhedra.net_methods import generate_nn_torch
import functools
from polyhedra.plot_utils import show_polygon_list, show_polygon_list2


def generate_input_region(gurobi_model, templates, boundaries):
    input = gurobi_model.addMVar(shape=6, lb=float("-inf"), name="input")
    for j, template in enumerate(templates):
        gurobi_model.update()
        multiplication = 0
        for i in range(6):
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
    # gurobi_model.setObjective(v[action_ego].sum(), grb.GRB.MAXIMIZE)  # maximise the output
    last_layer = gurobi_vars[-1]
    if action_ego == 0:
        gurobi_model.addConstr(last_layer[0] >= last_layer[1])
    else:
        gurobi_model.addConstr(last_layer[1] >= last_layer[0])
    gurobi_model.update()
    gurobi_model.optimize()
    # assert gurobi_model.status == 2, "LP wasn't optimally solved"
    return gurobi_model.status == 2


def apply_dynamic(input, gurobi_model: grb.Model, action_ego=0, t=0):
    '''

    :param input:
    :param gurobi_model:
    :param action_ego:
    :param t:
    :return:

    lead 100km/h 28m/s
    ego 130km/h  36.1 m/s


    '''

    x_lead = input[0]
    x_ego = input[1]
    v_lead = input[2]
    v_ego = input[3]
    a_lead = input[4]
    a_ego = input[5]
    z = gurobi_model.addMVar(shape=(6,), lb=float("-inf"), name=f"x_prime_{t}")
    const_acc = 3
    dt = .1  # seconds
    if action_ego == 0:
        acceleration = -const_acc
    elif action_ego == 1:
        acceleration = const_acc
    else:
        acceleration = 0
    a_ego_prime = acceleration
    v_ego_prime = v_ego + a_ego * dt
    v_lead_prime = v_lead + a_lead * dt
    x_lead_prime = x_lead + v_lead_prime * dt
    x_ego_prime = x_ego + v_ego_prime * dt
    # delta_x_prime = (x_lead + (v_lead + (a_lead + 0) * dt) * dt) - (x_ego + (v_ego + (a_ego + acceleration) * dt) * dt)
    # delta_v_prime = (v_lead + (a_lead + 0) * dt) - (v_ego + (a_ego + acceleration) * dt)
    gurobi_model.addConstr(z[0] == x_lead_prime, name=f"dyna_constr_1_{t}")
    gurobi_model.addConstr(z[1] == x_ego_prime, name=f"dyna_constr_2_{t}")
    gurobi_model.addConstr(z[2] == v_lead_prime, name=f"dyna_constr_3_{t}")
    gurobi_model.addConstr(z[3] == v_ego_prime, name=f"dyna_constr_4_{t}")
    gurobi_model.addConstr(z[4] == a_lead, name=f"dyna_constr_5_{t}")  # no change in a_lead
    gurobi_model.addConstr(z[5] == a_ego_prime, name=f"dyna_constr_6_{t}")
    return z


def optimise(templates, gurobi_model: grb.Model, x_prime):
    results = []
    for template in templates:
        gurobi_model.update()
        gurobi_model.setObjective(sum((template[i] * x_prime[i]) for i in range(6)), grb.GRB.MAXIMIZE)
        gurobi_model.optimize()
        # print_model(gurobi_model)
        if gurobi_model.status != 2:
            return None
        result = gurobi_model.ObjVal
        results.append(result)
    print(results)
    return np.array(results)


def print_model(gurobi_model):
    print("----------------------------")
    print(f'x_lead:{gurobi_model.getVarByName("input[0]").X}')
    print(f'x_ego:{gurobi_model.getVarByName("input[1]").X}')
    print(f'v_lead:{gurobi_model.getVarByName("input[2]").X}')
    print(f'v_ego:{gurobi_model.getVarByName("input[3]").X}')
    print(f'y_lead:{gurobi_model.getVarByName("input[4]").X}')
    print(f'y_ego:{gurobi_model.getVarByName("input[5]").X}')
    print(f'x_lead_prime:{gurobi_model.getVarByName("x_prime[0]").X}')
    print(f'x_ego_prime:{gurobi_model.getVarByName("x_prime[1]").X}')
    print(f'v_lead_prime:{gurobi_model.getVarByName("x_prime[2]").X}')
    print(f'v_ego_prime:{gurobi_model.getVarByName("x_prime[3]").X}')
    print(f'y_lead_prime:{gurobi_model.getVarByName("x_prime[4]").X}')
    print(f'y_ego_prime:{gurobi_model.getVarByName("x_prime[5]").X}')
    print("----------------------------")


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
    mode = 0  # 0 hardcoded guard, 1 nn guard
    output_flag = False
    nn = generate_nn_torch(six_dim=True, min_distance=20, max_distance=22)  # min_speed=30,max_speed=36
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', output_flag)
    graph = GraphExplorer(None)

    input_boundaries, template = get_template(3)

    input = generate_input_region(gurobi_model, template, input_boundaries)
    _, template = get_template(1)
    x_results = optimise(template, gurobi_model, input)
    # input_boundaries, template = get_template(3)
    if x_results is None:
        print("Model unsatisfiable")
        return
    root = tuple(x_results)
    graph.root = root
    graph.store_in_fringe(root)
    template_2d = np.array([[1, 0, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0]])
    template2d_speed = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    vertices = windowed_projection(template, x_results, template_2d)
    vertices_list = []
    vertices_list.append([vertices])
    # vertices = windowed_projection(template, x_results, template2d_speed)
    # vertices_list.append([vertices])

    # show_polygon_list2(vertices_list)
    seen = []
    frontier = [(0, root)]
    while len(frontier) != 0:
        t, x = frontier.pop()
        if t > 100:
            break
        if any([contained(x, s) for s in seen]):
            continue
        seen.append(x)
        # vertices = windowed_projection(template, x_results)
        # vertices_list.append([vertices])
        vertices_container = []
        x_primes = post(x, graph, mode, nn, output_flag, 0, template, vertices_container)
        vertices_list.append(vertices_container)
        for x_prime in x_primes:

            frontier = [(u, y) for u, y in frontier if not contained(y, x_prime)]
            if not any([contained(x_prime, y) for u, y in frontier]):
                frontier.append(((t + 1), x_prime))

    print(f"T={t}")
    show_polygon_list2(vertices_list, "x_lead-x_ego", "x_ego")  # show_polygon_list2(vertices_list)


def contained(x: tuple, y: tuple):
    # y contains x
    assert len(x) == len(y)
    for i in range(len(x)):
        if x[i] > y[i]:
            return False
    return True


def post(x, graph, mode, nn, output_flag, t, template, timestep_container):
    post = []
    if mode == 0:
        # deceleration 1
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        input = generate_input_region(gurobi_model, template, x)
        generate_mock_guard(gurobi_model, input, 0, t=0)
        # apply dynamic
        x_prime = apply_dynamic(input, gurobi_model, 0, t=t)  # 0
        found_successor, x_prime_results = h_repr_to_plot(graph, gurobi_model, template, timestep_container, x_prime)
        if found_successor:
            post.append(tuple(x_prime_results))
        # # deceleration 2
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        input = generate_input_region(gurobi_model, template, x)
        generate_mock_guard(gurobi_model, input, 0, t=1)  # # apply dynamic
        x_prime = apply_dynamic(input, gurobi_model, 0, t=t)  # 0
        found_successor, x_prime_results = h_repr_to_plot(graph, gurobi_model, template, timestep_container, x_prime)
        if found_successor:
            post.append(tuple(x_prime_results))
        # # acceleration
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        input = generate_input_region(gurobi_model, template, x)
        generate_mock_guard(gurobi_model, input, 1)  # # apply dynamic  #
        x_prime = apply_dynamic(input, gurobi_model, 1, t=t)  # 1
        found_successor, x_prime_results = h_repr_to_plot(graph, gurobi_model, template, timestep_container, x_prime)
        if found_successor:
            post.append(tuple(x_prime_results))
    if mode == 1:
        # deceleration
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        input = generate_input_region(gurobi_model, template, x)
        feasible = generate_nn_guard(gurobi_model, input, nn, action_ego=0)
        if feasible:
            # apply dynamic
            x_prime = apply_dynamic(input, gurobi_model, 0, t=t)
            found_successor, x_prime_results = h_repr_to_plot(graph, gurobi_model, template, timestep_container, x_prime)
            if found_successor:
                post.append(tuple(x_prime_results))
        # # acceleration
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', output_flag)
        input = generate_input_region(gurobi_model, template, x)
        feasible = generate_nn_guard(gurobi_model, input, nn, action_ego=1)
        if feasible:
            # apply dynamic
            x_prime = apply_dynamic(input, gurobi_model, 1, t=t)
            found_successor, x_prime_results = h_repr_to_plot(graph, gurobi_model, template, timestep_container, x_prime)
            if found_successor:
                post.append(tuple(x_prime_results))
    return post


def windowed_projection(template, x_results, template_2d):
    x_axis = template_2d[0]
    y_axis = template_2d[1]
    ub_lb_window_boundaries = np.array([1000, 1000, -100, -100])
    window_A, window_b = create_window_boundary(template, x_results, template_2d, ub_lb_window_boundaries)
    vertices, rays = pypoman.projection.project_polyhedron((template_2d, np.array([0, 0])), (window_A, window_b), canonicalize=False)
    vertices = np.vstack(vertices)
    return vertices


def get_template(mode=0):
    x_lead = e(6, 0)
    x_ego = e(6, 1)
    v_lead = e(6, 2)
    v_ego = e(6, 3)
    a_lead = e(6, 4)
    a_ego = e(6, 5)
    if mode == 0:  # box directions with intervals
        input_boundaries = [52, -40, 31, -30, 30, -27, 30.5, -30, 0, -0, 0, -0]
        # optimise in a direction
        template = []
        for dimension in range(6):
            template.append(e(6, dimension))
            template.append(-e(6, dimension))
        template = np.array(template)  # the 6 dimensions in 2 variables

        # t1 = [0] * 6
        # t1[0] = -1
        # t1[1] = 1
        # template = np.vstack([template, t1])
        return input_boundaries, template
    if mode == 1:  # directions to easily find fixed point

        input_boundaries = [20]

        template = np.array([a_lead, -a_lead, a_ego, -a_ego, v_lead, -v_lead, v_ego, -(x_lead - x_ego), (x_lead - x_ego)])
        return input_boundaries, template
    if mode == 2:
        input_boundaries = [0, -100, 30, -31, 20, -30, 0, -35, 0, -0, -10, -10, 20]
        # optimise in a direction
        template = []
        for dimension in range(6):
            t1 = [0] * 6
            t1[dimension] = 1
            t2 = [0] * 6
            t2[dimension] = -1
            template.append(t1)
            template.append(t2)
        # template = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])  # the 8 dimensions in 2 variables
        template = np.array(template)  # the 6 dimensions in 2 variables

        t1 = [0] * 6
        t1[0] = 1
        t1[1] = -1
        template = np.vstack([template, t1])
        return input_boundaries, template
    if mode == 3:  # single point box directions +diagonal
        input_boundaries = [30, -30, 0, -0, 28, -28, 36, -36, 0, -0, 0, -0, 0]
        # optimise in a direction
        template = []
        for dimension in range(6):
            t1 = [0] * 6
            t1[dimension] = 1
            t2 = [0] * 6
            t2[dimension] = -1
            template.append(t1)
            template.append(t2)
        # template = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])  # the 8 dimensions in 2 variables
        template = np.array(template)  # the 6 dimensions in 2 variables

        t1 = [0] * 6
        t1[0] = -1
        t1[1] = 1
        template = np.vstack([template, t1])
        return input_boundaries, template
    if mode == 4:  # octagon, every pair of variables
        input_boundaries = [20]
        template = []
        for dimension in range(6):
            t1 = [0] * 6
            t1[dimension] = 1
            t2 = [0] * 6
            t2[dimension] = -1
            template.append(t1)
            template.append(t2)
            for other_dimension in range(dimension + 1, 6):
                t1 = [0] * 6
                t1[dimension] = 1
                t1[other_dimension] = -1
                t2 = [0] * 6
                t2[dimension] = -1
                t2[other_dimension] = 1
                t3 = [0] * 6
                t3[dimension] = 1
                t3[other_dimension] = 1
                t4 = [0] * 6
                t4[dimension] = -1
                t4[other_dimension] = -1
                template.append(t1)
                template.append(t2)
                template.append(t3)
                template.append(t4)
        return input_boundaries, np.array(template)


def e(n, i):
    result = [0] * n
    result[i] = 1
    return np.array(result)


def h_repr_to_plot(graph, gurobi_model, template, vertices_list, x_prime):
    x_prime_results = optimise(template, gurobi_model, x_prime)  # h representation
    added = False
    if x_prime_results is not None:
        x_prime_tuple = tuple(x_prime_results)
        added = graph.store_in_fringe(x_prime_tuple)
        if added:
            template_2d = np.array([[1, 0, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0]])
            vertices = windowed_projection(template, x_prime_results, template_2d)
            vertices_list.append(vertices)
            # template2d_speed = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
            # vertices = windowed_projection(template, x_prime_results, template2d_speed)
            # vertices_list.append(vertices)
    return added, x_prime_results


if __name__ == '__main__':
    main()

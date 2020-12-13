import numpy as np
import gurobi as grb
import pypoman

from polyhedra.graph_explorer import GraphExplorer
from polyhedra.net_methods import generate_nn_torch

from polyhedra.plot_utils import show_polygon_list, show_polygon_list2


def generate_input_region(gurobi_model, templates, boundaries):
    input = gurobi_model.addVars(6, lb=float("-inf"), name="input")
    for j, template in enumerate(templates):
        gurobi_model.update()
        multiplication = 0
        for i in range(6):
            multiplication += template[i] * input[i]
        gurobi_model.addConstr(multiplication >= boundaries[j], name=f"input_constr_{j}")
    return input


def generate_mock_guard(gurobi_model: grb.Model, input, action_ego=0, t=0):
    if action_ego == 0:
        if t == 0:
            gurobi_model.addConstr(input[3] >= 30, name=f"cond2_{t}")
        else:
            gurobi_model.addConstr((input[0] - input[1]) <= 50, name=f"cond1_{t}")
    else:
        epsilon = 1e-8
        gurobi_model.addConstr((input[0] - input[1]) >= 50 + epsilon, name=f"cond1_{t}")
        gurobi_model.addConstr(input[3] <= 30 + epsilon, name=f"cond2_{t}")


def apply_dynamic(input, gurobi_model: grb.Model, action_ego=0, t=0):
    x_lead = input[0]
    x_ego = input[1]
    v_lead = input[2]
    v_ego = input[3]
    y_lead = input[4]
    y_ego = input[5]
    z = gurobi_model.addVars(6, lb=float("-inf"), name=f"x_prime_{t}")
    a_ego = 1
    dt = .1
    if action_ego == 0:
        acceleration = -a_ego
    else:
        acceleration = a_ego
    y_ego_prime = y_ego + acceleration
    v_ego_prime = v_ego + (y_ego + acceleration) * dt
    v_lead_prime = v_lead + y_lead * dt
    x_lead_prime = x_lead + (v_lead + (y_lead + 0) * dt) * dt
    x_ego_prime = x_ego + (v_ego + (y_ego + acceleration) * dt) * dt
    # delta_x_prime = (x_lead + (v_lead + (y_lead + 0) * dt) * dt) - (x_ego + (v_ego + (y_ego + acceleration) * dt) * dt)
    # delta_v_prime = (v_lead + (y_lead + 0) * dt) - (v_ego + (y_ego + acceleration) * dt)
    gurobi_model.addConstr(z[0] == x_lead_prime, name=f"dyna_constr_1_{t}")
    gurobi_model.addConstr(z[1] == x_ego_prime, name=f"dyna_constr_2_{t}")
    gurobi_model.addConstr(z[2] == v_lead_prime, name=f"dyna_constr_3_{t}")
    gurobi_model.addConstr(z[3] == v_ego_prime, name=f"dyna_constr_4_{t}")
    gurobi_model.addConstr(z[4] == y_lead, name=f"dyna_constr_5_{t}")  # no change in y_lead
    gurobi_model.addConstr(z[5] == y_ego_prime, name=f"dyna_constr_6_{t}")
    return z


def optimise(templates, gurobi_model: grb.Model, x_prime):
    results = []
    for template in templates:
        gurobi_model.update()
        gurobi_model.setObjective(sum((template[i] * x_prime[i]) for i in range(6)), grb.GRB.MINIMIZE)
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


def main():
    nn = generate_nn_torch()
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', False)
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
    graph = GraphExplorer(template)
    input_boundaries = [90, -92, 30, -31, 20, -30, 30, -30.5, 0, -0, 0, -0]
    root = tuple(input_boundaries)
    graph.root = root
    graph.store_boundary(root)
    input = generate_input_region(gurobi_model, template, input_boundaries)
    x_results = optimise(template, gurobi_model, input)
    if x_results is None:
        print("Model unsatisfiable")
        return
    x_vertices = pypoman.duality.compute_polytope_vertices(-template, -x_results)
    vertices_list = []
    vertices_list.append([x_vertices])
    for t in range(15):
        fringe = graph.get_next_in_fringe()
        timestep_container = []
        for fringe_element in fringe:
            found_successor = False
            # deceleration 1
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', False)
            input = generate_input_region(gurobi_model, template, fringe_element)
            generate_mock_guard(gurobi_model, input, 0, t=0)
            # apply dynamic
            x_prime = apply_dynamic(input, gurobi_model, 0, t=t)
            found_successor = h_repr_to_plot(found_successor, fringe_element, graph, gurobi_model, template, timestep_container, x_prime)
            # deceleration 2
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', False)
            input = generate_input_region(gurobi_model, template, fringe_element)
            generate_mock_guard(gurobi_model, input, 0, t=1)
            # apply dynamic
            x_prime = apply_dynamic(input, gurobi_model, 0, t=t)
            found_successor = h_repr_to_plot(found_successor, fringe_element, graph, gurobi_model, template, timestep_container, x_prime)
            # # acceleration
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', False)
            input = generate_input_region(gurobi_model, template, fringe_element)
            generate_mock_guard(gurobi_model, input, 1)
            # apply dynamic
            x_prime = apply_dynamic(input, gurobi_model, 1, t=t)
            found_successor = h_repr_to_plot(found_successor, fringe_element, graph, gurobi_model, template, timestep_container, x_prime)
            if not found_successor:
                graph.graph.nodes[fringe_element]["ignore"] = True
        vertices_list.append(timestep_container)
    show_polygon_list(vertices_list)
    show_polygon_list2(vertices_list)


def h_repr_to_plot(found_successor, fringe, graph, gurobi_model, template, vertices_list, x_prime):
    x_prime_results = optimise(template, gurobi_model, x_prime)  # h representation
    if x_prime_results is not None:
        x_prime_tuple = tuple(x_prime_results)
        found_successor = True
        graph.store_boundary(x_prime_tuple)
        graph.graph.add_edge(fringe, x_prime_tuple)
        x_prime_vertices = pypoman.duality.compute_polytope_vertices(-template, -x_prime_results)
        vertices_list.append(x_prime_vertices)
    return found_successor


if __name__ == '__main__':
    main()

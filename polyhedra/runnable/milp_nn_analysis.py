import torch
import torch.nn
import numpy as np
import gurobi as grb
import pypoman
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mosaic.utils import compute_trace_polygon, PolygonSort
from polyhedra.net_methods import generate_nn_torch, generate_mock_input
import plotly.graph_objects as go
import itertools


def generate_input_region(gurobi_model):
    input = gurobi_model.addVars(6, lb=float("-inf"), name="input")
    # x_lead
    gurobi_model.addConstr(input[0] >= 90, name="input_constr_1")
    gurobi_model.addConstr(input[0] <= 92, name="input_constr_2")
    # x_ego
    gurobi_model.addConstr(input[1] >= 30, name="input_constr_3")
    gurobi_model.addConstr(input[1] <= 31, name="input_constr_4")
    # v_lead
    gurobi_model.addConstr(input[2] >= 20, name="input_constr_5")
    gurobi_model.addConstr(input[2] <= 30, name="input_constr_6")
    # v_ego
    gurobi_model.addConstr(input[3] >= 30, name="input_constr_7")
    gurobi_model.addConstr(input[3] <= 30.5, name="input_constr_8")
    # y_lead
    gurobi_model.addConstr(input[4] >= 0, name="input_constr_9")
    gurobi_model.addConstr(input[4] <= 0, name="input_constr_10")
    # y_ego
    gurobi_model.addConstr(input[5] >= 0, name="input_constr_11")
    gurobi_model.addConstr(input[5] <= 0, name="input_constr_12")
    return input


# def generate_mock_guard(gurobi_model: grb.Model, input, mode=0, t=0):
#     if mode == 0:  # should accelerate
#         gurobi_model.addConstr((input[0] - input[1]) >= 50, name=f"cond1_{t}")
#         gurobi_model.addConstr(input[3] <= 25, name=f"cond2_{t}")
#     elif mode == 1:  # should decelerate
#         gurobi_model.addConstr((input[0] - input[1]) >= 50, name=f"cond1_{t}")
#         gurobi_model.addConstr(input[3] >= 25, name=f"cond2_{t}")
#     elif mode == 2:  # should decelerate
#         gurobi_model.addConstr((input[0] - input[1]) <= 50, name=f"cond1_{t}")
#         gurobi_model.addConstr(input[3] >= 25, name=f"cond2_{t}")
#     else:  # should decelerate
#         gurobi_model.addConstr((input[0] - input[1]) <= 50, name=f"cond1_{t}")
#         gurobi_model.addConstr(input[3] <= 25, name=f"cond2_{t}")

def generate_mock_guard(gurobi_model: grb.Model, input, action_ego=0, t=0):
    if action_ego == 0:
        y = gurobi_model.addVars(2, vtype=grb.GRB.BINARY, name=f"or_indicator_{t}")
        gurobi_model.addGenConstrIndicator(y[0], True, ((input[0] - input[1]) <= 50), name=f"cond1_{t}")
        gurobi_model.addGenConstrIndicator(y[1], True, (input[3] >= 25), name=f"cond2_{t}")
        gurobi_model.addConstr(y[0] + y[1] == 1, name=f"cond3_{t}")
    else:
        constr3 = gurobi_model.getConstrByName(f"cond3_{t}")
        if constr3 is not None:
            gurobi_model.remove(constr3)
        gurobi_model.addConstr((input[0] - input[1]) >= 50, name=f"cond1_{t}")
        gurobi_model.addConstr(input[3] <= 25, name=f"cond2_{t}")


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
    input = generate_input_region(gurobi_model)
    x_results = optimise(template, gurobi_model, input)
    if x_results is None:
        print("Model unsatisfiable")
        return
    x_vertices = pypoman.duality.compute_polytope_vertices(template, x_results)
    vertices_list = []
    vertices_list.append(x_vertices)
    for t in range(20):
        generate_mock_guard(gurobi_model, input, 0, t=t)
        # apply dynamic
        x_prime = apply_dynamic(input, gurobi_model, 0, t=t)
        # enclosing halfspaces  # max <template,y>
        x_prime_results = optimise(template, gurobi_model, x_prime)  # h representation
        if x_prime_results is None:
            break
        x_prime_vertices = pypoman.duality.compute_polytope_vertices(template, x_prime_results)
        vertices_list.append(x_prime_vertices)
        input = x_prime
    show_polygon_list(vertices_list)


def transform_vertices(polygon_vertices_list):
    result = []
    for vertices in polygon_vertices_list:
        transformed_vertices = []
        for vertex in vertices:
            transformed_vertex = np.zeros(shape=(2,))
            transformed_vertex[0] = vertex[0] - vertex[1]
            transformed_vertex[1] = vertex[3]
            transformed_vertices.append(transformed_vertex)
        result.append(transformed_vertices)
    return result


def show_polygon_list(polygon_vertices_list):  # x_prime_vertices, x_vertices

    # scaler = StandardScaler()
    # scaler.fit(polygon_vertices_list[0])  # list(itertools.chain.from_iterable(polygon_vertices_list)))
    # scaled_list = []
    # for x in polygon_vertices_list:
    #     scaled_list.append(scaler.transform(x))
    # pca = PCA(n_components=2)
    # pca.fit(list(itertools.chain.from_iterable(scaled_list)))  #
    # principal_components_list = []
    # for x_scaled in scaled_list:
    #     principal_components_list.append(pca.transform(x_scaled))
    # todo apply transformation of points
    principal_components_list = transform_vertices(polygon_vertices_list)
    traces = []
    for principal_component in principal_components_list:
        traces.append(compute_polygon_trace(principal_component))
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    fig.show()


def compute_polygon_trace(principalComponents):
    polygon1 = PolygonSort(principalComponents)
    trace1 = compute_trace_polygon(polygon1)
    return trace1


if __name__ == '__main__':
    main()

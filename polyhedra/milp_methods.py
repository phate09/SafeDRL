import numpy as np
import gurobipy as grb

'''Collection of methods to handle abstract states in MILP model'''

def generate_input_region(gurobi_model, templates, boundaries, env_input_size):
    input = gurobi_model.addMVar(shape=env_input_size, lb=float("-inf"), ub=float("inf"), name="input")
    generate_region_constraints(gurobi_model, templates, input, boundaries, env_input_size)
    return input


def generate_region_constraints(gurobi_model, templates, input, boundaries, env_input_size, invert=False, eps=0.0):
    for j, template in enumerate(templates):
        gurobi_model.update()
        multiplication = 0
        for i in range(env_input_size):
            multiplication += template[i] * input[i]
        if not invert:
            gurobi_model.addConstr(multiplication <= boundaries[j] - eps, name=f"input_constr_{j}")
        else:
            gurobi_model.addConstr(multiplication >= boundaries[j] + eps, name=f"input_constr_{j}")

def optimise(templates: np.ndarray, gurobi_model: grb.Model, x_prime: tuple):
    results = []
    for template in templates:
        gurobi_model.update()
        gurobi_model.setObjective(sum((template[i] * x_prime[i]) for i in range(len(template))), grb.GRB.MAXIMIZE)
        gurobi_model.optimize()
        # print_model(gurobi_model)
        if gurobi_model.status == 5:
            result = float("inf")
            results.append(result)
            continue
        if gurobi_model.status == 4 or gurobi_model.status == 3:
            return None
        assert gurobi_model.status == 2, f"gurobi_model.status=={gurobi_model.status}"
        # if gurobi_model.status != 2:
        #     return None
        result = gurobi_model.ObjVal
        results.append(result)
    return np.array(results)

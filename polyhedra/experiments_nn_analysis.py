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

from polyhedra.plot_utils import show_polygon_list3


class Experiment():
    def __init__(self, env_input_size: int):
        self.get_nn_fn = None
        self.plot_fn = None
        self.post_fn_remote = None
        self.template_2d: np.ndarray = None
        self.input_template: np.ndarray = None
        self.input_boundaries: List = None
        self.analysis_template: np.ndarray = None
        self.unsafe_zone: List[Tuple] = None
        self.output_flag = False
        self.env_input_size: int = env_input_size
        self.n_workers = 8
        self.plotting_time_interval = 60 * 2
        self.time_horizon = 100
        self.use_bfs = True  # use Depth-first-search or Depth-first-search
        self.local_mode = False  # disable multi processing

    def run_experiment(self, local_mode=False):
        assert self.get_nn_fn is not None
        assert self.plot_fn is not None
        assert self.post_fn_remote is not None
        assert self.template_2d is not None
        assert self.input_template is not None
        assert self.input_boundaries is not None
        assert self.analysis_template is not None
        assert self.unsafe_zone is not None
        self.local_mode = local_mode
        if local_mode:
            print("Running the experiment in LOCAL MODE")
        nn: torch.nn.Sequential = self.get_nn_fn()
        root = self.generate_root_polytope()
        max_t, num_already_visited, vertices_list, unsafe = self.main_loop(nn, self.analysis_template, [root], self.template_2d)
        print(f"T={max_t}")

        print(f"The algorithm skipped {num_already_visited} already visited states")
        if unsafe:
            print("The agent is unsafe")
        elif max_t < self.time_horizon:
            print("The agent is safe")
        else:
            print(f"It could not be determined if the agent is safe or not within {self.time_horizon} steps. Increase 'time_horizon' to increase the number of steps to analyse")

    def main_loop(self, nn, template, root_list: List[Tuple], template_2d):
        vertices_list = defaultdict(list)
        ray.init(local_mode=self.local_mode, log_to_driver=False)
        seen = []
        frontier = [(0, x) for x in root_list]
        max_t = 0
        num_already_visited = 0
        widgets = [progressbar.Variable('n_workers'), ', ', progressbar.Variable('frontier'), ', ', progressbar.Variable('seen'), ', ', progressbar.Variable('num_already_visited'), ", ",
                   progressbar.Variable('max_t'), ", ", progressbar.Variable('last_visited_state')]
        proc_ids = []
        last_time_plot = None
        with progressbar.ProgressBar(widgets=widgets) as bar:
            while len(frontier) != 0 or len(proc_ids) != 0:
                while len(proc_ids) < self.n_workers and len(frontier) != 0:
                    t, x = frontier.pop(0) if self.use_bfs else frontier.pop()
                    if max_t > self.time_horizon:
                        break
                    if any([contained(x, s) for s in seen]):
                        num_already_visited += 1
                        continue
                    max_t = max(max_t, t)
                    vertices_list[t].append(np.array(x))
                    if self.check_unsafe(template, x):
                        print(f"Unsafe state found at timestep t={t}")
                        print(x)
                        return max_t, num_already_visited, vertices_list, True
                    seen.append(x)
                    proc_ids.append(self.post_fn_remote.remote(self,x, nn, self.output_flag, t, template))
                if last_time_plot is None or time.time() - last_time_plot >= self.plotting_time_interval:
                    self.plot_fn(vertices_list, template, template_2d)
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
        self.plot_fn(vertices_list, template, template_2d)
        return max_t, num_already_visited, vertices_list, False

    def generate_root_polytope(self):
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', self.output_flag)
        input = Experiment.generate_input_region(gurobi_model, self.input_template, self.input_boundaries, self.env_input_size)
        x_results = self.optimise(self.analysis_template, gurobi_model, input)
        if x_results is None:
            print("Model unsatisfiable")
            return None
        root = tuple(x_results)
        return root

    @staticmethod
    def generate_input_region(gurobi_model, templates, boundaries, env_input_size):
        input = gurobi_model.addMVar(shape=env_input_size, lb=float("-inf"), ub=float("inf"), name="input")
        Experiment.generate_region_constraints(gurobi_model, templates, input, boundaries, env_input_size)
        return input

    @staticmethod
    def generate_region_constraints(gurobi_model, templates, input, boundaries, env_input_size):
        for j, template in enumerate(templates):
            gurobi_model.update()
            multiplication = 0
            for i in range(env_input_size):
                multiplication += template[i] * input[i]
            gurobi_model.addConstr(multiplication <= boundaries[j], name=f"input_constr_{j}")

    def optimise(self, templates: np.ndarray, gurobi_model: grb.Model, x_prime: tuple):
        results = []
        for template in templates:
            gurobi_model.update()
            gurobi_model.setObjective(sum((template[i] * x_prime[i]) for i in range(self.env_input_size)), grb.GRB.MAXIMIZE)
            gurobi_model.optimize()
            # print_model(gurobi_model)
            if gurobi_model.status != 2:
                return None
            result = gurobi_model.ObjVal
            results.append(result)
        return np.array(results)

    def check_unsafe(self, template, bnds):
        for A, b in self.unsafe_zone:
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', False)
            input = gurobi_model.addMVar(shape=(self.env_input_size,), lb=float("-inf"), name="input")
            Experiment.generate_region_constraints(gurobi_model, template, input, bnds, self.env_input_size)
            Experiment.generate_region_constraints(gurobi_model, A, input, b, self.env_input_size)
            gurobi_model.update()
            gurobi_model.optimize()
            if gurobi_model.status == 2:
                return True
        return False

    @staticmethod
    def e(n, i):
        result = [0] * n
        result[i] = 1
        return np.array(result)

    def h_repr_to_plot(self, gurobi_model, template, x_prime):
        x_prime_results = self.optimise(template, gurobi_model, x_prime)  # h representation
        return x_prime_results is not None, x_prime_results

    @staticmethod
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

    @staticmethod
    def generic_plot(title_x, title_y, vertices_list, template, template_2d):
        show_polygon_list3(vertices_list, title_x, title_y, template, template_2d)


def contained(x: tuple, y: tuple):
    # y contains x
    assert len(x) == len(y)
    for i in range(len(x)):
        if x[i] > y[i]:
            return False
    return True

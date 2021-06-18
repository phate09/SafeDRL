import csv
import datetime
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Tuple, List
import pyomo.environ as pyo
import pyomo.gdp as gdp
import gurobipy as grb
import networkx
import numpy as np
import progressbar
import ray
import torch
from interval import interval, imath
import plotly.graph_objects as go
from pyomo.core import TransformationFactory
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints

from polyhedra.plot_utils import show_polygon_list3, show_polygon_list31d


class Experiment():
    def __init__(self, env_input_size: int):
        self.before_start_fn = None
        self.update_progress_fn = None
        self.rounding_value = 1024
        self.get_nn_fn = None
        self.plot_fn = None
        self.post_fn_remote = None
        self.assign_lbl_fn = None
        self.additional_seen_fn = None  # adds additional elements to seen list
        self.template_2d: np.ndarray = None
        self.input_template: np.ndarray = None
        self.input_boundaries: List = None
        self.analysis_template: np.ndarray = None
        self.unsafe_zone: List[Tuple] = None
        self.output_flag = False
        self.env_input_size: int = env_input_size
        self.n_workers = 8
        self.plotting_time_interval = 60 * 5
        self.time_horizon = 100
        self.use_bfs = True  # use Breadth-first-search or Depth-first-search
        self.local_mode = False  # disable multi processing
        self.use_rounding = True
        self.show_progressbar = True
        self.show_progress_plot = True
        self.save_dir = None
        self.keep_model = False  # whether to keep the gurobi model for later timesteps
        self.graph = networkx.Graph()  # set None to disable use of graph

    def run_experiment(self):
        assert self.get_nn_fn is not None
        assert self.plot_fn is not None
        assert self.post_fn_remote is not None
        assert self.template_2d is not None
        assert self.input_template is not None
        assert self.input_boundaries is not None
        assert self.analysis_template is not None
        assert self.unsafe_zone is not None
        assert self.assign_lbl_fn is not None
        experiment_start_time = time.time()
        nn: torch.nn.Sequential = self.get_nn_fn()

        max_t, num_already_visited, vertices_list, unsafe = self.main_loop(nn, self.analysis_template, self.template_2d)
        print(f"T={max_t}")

        print(f"The algorithm skipped {num_already_visited} already visited states")
        safe = None
        if unsafe:
            print("The agent is unsafe")
            safe = False
        elif max_t < self.time_horizon:
            print("The agent is safe")
            safe = True
        else:
            print(f"It could not be determined if the agent is safe or not within {self.time_horizon} steps. Increase 'time_horizon' to increase the number of steps to analyse")
            safe = None
        experiment_end_time = time.time()
        elapsed_seconds = round((experiment_end_time - experiment_start_time))
        print(f"Total verification time {str(datetime.timedelta(seconds=elapsed_seconds))}")
        return elapsed_seconds, safe, max_t

    def main_loop(self, nn, template, template_2d):
        root = self.generate_root_polytope()
        root_pair = (root, 0)  # label for root is always 0
        root_list = [root_pair]
        vertices_list = defaultdict(list)
        seen = []
        if self.additional_seen_fn is not None:
            for extra in self.additional_seen_fn():
                seen.append(extra)
        frontier = [(0, x) for x in root_list]
        if self.graph is not None:
            self.graph.add_node(root_pair)
        max_t = 0
        num_already_visited = 0
        widgets = [progressbar.Variable('n_workers'), ', ', progressbar.Variable('frontier'), ', ', progressbar.Variable('seen'), ', ', progressbar.Variable('num_already_visited'), ", ",
                   progressbar.Variable('max_t'), ", ", progressbar.Variable('last_visited_state')]
        proc_ids = []
        last_time_plot = None
        if self.before_start_fn is not None:
            self.before_start_fn(nn)
        with progressbar.ProgressBar(widgets=widgets) if self.show_progressbar else nullcontext() as bar:
            while len(frontier) != 0 or len(proc_ids) != 0:
                while len(proc_ids) < self.n_workers and len(frontier) != 0:
                    t, (x, x_label) = frontier.pop(0) if self.use_bfs else frontier.pop()
                    if max_t > self.time_horizon:
                        print(f"Reached horizon t={t}")
                        self.plot_fn(vertices_list, template, template_2d)
                        return max_t, num_already_visited, vertices_list, False
                    contained_flag = False
                    to_remove = []
                    for (s, s_label) in seen:
                        if s_label == x_label:
                            if contained(x, s):
                                contained_flag = True
                                break
                            if contained(s, x):
                                to_remove.append((s, s_label))
                    for rem in to_remove:
                        num_already_visited += 1
                        seen.remove(rem)
                    if contained_flag:
                        num_already_visited += 1
                        continue
                    max_t = max(max_t, t)
                    vertices_list[t].append(np.array(x))
                    if self.check_unsafe(template, x, x_label):
                        print(f"Unsafe state found at timestep t={t}")
                        print((x, x_label))
                        self.plot_fn(vertices_list, template, template_2d)
                        return max_t, num_already_visited, vertices_list, True
                    seen.append((x, x_label))
                    proc_ids.append(self.post_fn_remote.remote(self, x, x_label, nn, self.output_flag, t, template))
                if last_time_plot is None or time.time() - last_time_plot >= self.plotting_time_interval:
                    if last_time_plot is not None:
                        self.plot_fn(vertices_list, template, template_2d)
                    last_time_plot = time.time()
                if self.update_progress_fn is not None:
                    self.update_progress_fn(n_workers=len(proc_ids), seen=len(seen), frontier=len(frontier), num_already_visited=num_already_visited, max_t=max_t)
                if self.show_progressbar:
                    bar.update(value=bar.value + 1, n_workers=len(proc_ids), seen=len(seen), frontier=len(frontier), num_already_visited=num_already_visited, last_visited_state=str(x), max_t=max_t)
                ready_ids, proc_ids = ray.wait(proc_ids, num_returns=len(proc_ids), timeout=0.5)
                if len(ready_ids) != 0:
                    x_primes_list = ray.get(ready_ids)
                    assert len(x_primes_list) != 0, "something is wrong with the calculation of the successor"
                    for x_primes in x_primes_list:
                        for x_prime, (parent, parent_lbl) in x_primes:
                            x_prime_label = self.assign_lbl_fn(x_prime, parent, parent_lbl)
                            if self.use_rounding:
                                # x_prime_rounded = tuple(np.trunc(np.array(x_prime) * self.rounding_value) / self.rounding_value)  # todo should we round to prevent numerical errors?
                                x_prime_rounded = self.round_tuple(x_prime, self.rounding_value)
                                # x_prime_rounded should always be bigger than x_prime
                                assert contained(x_prime, x_prime_rounded), f"{x_prime} not contained in {x_prime_rounded}"
                                x_prime = x_prime_rounded
                            frontier = [(u, (y, y_label)) for u, (y, y_label) in frontier if not (y_label == x_prime_label and contained(y, x_prime))]
                            if not any([(y_label == x_prime_label and contained(x_prime, y)) for u, (y, y_label) in frontier]):

                                frontier.append(((t + 1), (x_prime, x_prime_label)))
                                if self.graph is not None:
                                    self.graph.add_edge((parent, parent_lbl), (x_prime, x_prime_label))
                                # print(x_prime)
                            else:
                                num_already_visited += 1
        self.plot_fn(vertices_list, template, template_2d)
        return max_t, num_already_visited, vertices_list, False

    @staticmethod
    def round_tuple(x, rounding_value):
        '''To be used only for template values, this is not a good rounding in other cases'''
        rounded_x = []
        for val in x:
            rounded_value = Experiment.round_single(val, rounding_value)
            rounded_x.append(rounded_value)
        return tuple(rounded_x)

    @staticmethod
    def round_single(val, rounding_value):
        '''To be used only for template values, this is not a good rounding in other cases'''
        if val < 0:
            rounded_value = -1 * math.floor(abs(val) * rounding_value) / rounding_value
        else:
            rounded_value = math.ceil(abs(val) * rounding_value) / rounding_value
        return rounded_value

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
    def generate_input_region_pyo(model: pyo.ConcreteModel, templates, boundaries, env_input_size, name="input"):
        input = pyo.Var(range(env_input_size), domain=pyo.Reals, name=name)
        model.add_component(name, input)
        Experiment.generate_region_constraints_pyo(model, templates, model.input, boundaries, env_input_size, name=f"{name}_constraints")
        return model.input

    @staticmethod
    def generate_region_constraints(gurobi_model, templates, input, boundaries, env_input_size, invert=False):
        for j, template in enumerate(templates):
            gurobi_model.update()
            multiplication = 0
            for i in range(env_input_size):
                multiplication += template[i] * input[i]
            if not invert:
                gurobi_model.addConstr(multiplication <= boundaries[j], name=f"input_constr_{j}")
            else:
                gurobi_model.addConstr(multiplication >= boundaries[j], name=f"input_constr_{j}")

    @staticmethod
    def generate_region_constraints_pyo(model: pyo.ConcreteModel, templates, input, boundaries, env_input_size, invert=False, name="region_constraints"):
        region_constraints = pyo.ConstraintList()
        model.add_component(name, region_constraints)
        for j, template in enumerate(templates):
            multiplication = 0
            for i in range(env_input_size):
                multiplication += template[i] * input[i]
            if not invert:
                # gurobi_model.addConstr(multiplication <= boundaries[j], name=f"input_constr_{j}")
                region_constraints.add(multiplication <= boundaries[j])
            else:
                # gurobi_model.addConstr(multiplication >= boundaries[j], name=f"input_constr_{j}")
                region_constraints.add(multiplication >= boundaries[j])

    def optimise(self, templates: np.ndarray, gurobi_model: grb.Model, x_prime: tuple):
        results = []
        for template in templates:
            gurobi_model.update()
            gurobi_model.setObjective(sum((template[i] * x_prime[i]) for i in range(self.env_input_size)), grb.GRB.MAXIMIZE)
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

    def optimise_pyo(self, templates: np.ndarray, model: pyo.ConcreteModel, x_prime):
        results = []
        for template in templates:
            model.del_component(model.obj)
            model.obj = pyo.Objective(expr=sum((template[i] * x_prime[i]) for i in range(self.env_input_size)), sense=pyo.maximize)
            result = pyo.SolverFactory('glpk').solve(model)
            # assert (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal), f"LP wasn't optimally solved {x}"
            # print_model(gurobi_model)
            if (result.solver.status == SolverStatus.ok):
                if (result.solver.termination_condition == TerminationCondition.optimal):
                    result = pyo.value(model.obj)
                    results.append(result)
                elif (result.solver.termination_condition == TerminationCondition.unbounded):
                    result = float("inf")
                    results.append(result)
                    continue
                else:
                    return None
            else:
                return None
        return np.array(results)

    def check_unsafe(self, template, bnds, x_label):
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
        result[min(i, n)] = 1
        return np.array(result)

    @staticmethod
    def octagon(n):
        template = []
        for i in range(n):
            x = Experiment.e(n, i)
            template.append(x)
            template.append(-x)
            for j in range(0, i):
                y = Experiment.e(n, j)
                template.append(x + y)
                template.append(x - y)
                template.append(y - x)
                template.append(-y - x)
        return np.stack(template)

    @staticmethod
    def combinations(items: List[np.ndarray]):
        template = []
        for i, item in enumerate(items):
            template.append(item)
            template.append(-item)
            for j in range(0, i):
                other = items[j]
                template.append(item + other)
                template.append(item - other)
                template.append(other - item)
                template.append(-other - item)
        return np.stack(template)

    @staticmethod
    def box(n):
        template = []
        for i in range(n):
            x = Experiment.e(n, i)
            template.append(x)
            template.append(-x)
        return np.stack(template)

    def h_repr_to_plot(self, gurobi_model, template, x_prime):
        x_prime_results = self.optimise(template, gurobi_model, x_prime)  # h representation
        return x_prime_results is not None, x_prime_results

    @staticmethod
    def generate_nn_guard(gurobi_model: grb.Model, input, nn: torch.nn.Sequential, action_ego=0, M=1e2):
        gurobi_model.setParam("DualReductions", 0)
        gurobi_vars = []
        gurobi_vars.append(input)
        for i, layer in enumerate(nn):

            # print(layer)
            if type(layer) is torch.nn.Linear:
                v = gurobi_model.addMVar(lb=float("-inf"), shape=(int(layer.out_features)), name=f"layer_{i}")
                lin_expr = layer.weight.data.numpy() @ gurobi_vars[-1]
                if layer.bias is not None:
                    lin_expr = lin_expr + layer.bias.data.numpy()
                gurobi_model.addConstr(v == lin_expr, name=f"linear_constr_{i}")
                gurobi_vars.append(v)
                gurobi_model.update()
                gurobi_model.optimize()
                assert gurobi_model.status == 2, "LP wasn't optimally solved"
            elif type(layer) is torch.nn.ReLU:
                v = gurobi_model.addMVar(lb=float("-inf"), shape=gurobi_vars[-1].shape, name=f"layer_{i}")  # same shape as previous

                z = gurobi_model.addMVar(shape=gurobi_vars[-1].shape, vtype=grb.GRB.BINARY, name=f"relu_{i}")  # lb=0, ub=1,
                # gurobi_model.addConstr(v == grb.max_(0, gurobi_vars[-1]))
                gurobi_model.addConstr(v >= gurobi_vars[-1], name=f"relu_constr_1_{i}")
                gurobi_model.addConstr(v <= gurobi_vars[-1] + M * z, name=f"relu_constr_2_{i}")
                gurobi_model.addConstr(v >= 0, name=f"relu_constr_3_{i}")
                gurobi_model.addConstr(v <= M - M * z, name=f"relu_constr_4_{i}")
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
        # gurobi_model.update()
        # gurobi_model.optimize()
        # assert gurobi_model.status == 2, "LP wasn't optimally solved"
        # gurobi_model.setObjective(v[action_ego].sum(), grb.GRB.MAXIMIZE)  # maximise the output
        last_layer = gurobi_vars[-1]
        for i in range(last_layer.shape[0]):
            if i == action_ego:
                continue
            gurobi_model.addConstr(last_layer[action_ego] >= last_layer[i], name="last_layer")
        # if action_ego == 0:
        #     gurobi_model.addConstr(last_layer[0] >= last_layer[1], name="last_layer")
        # else:
        #     gurobi_model.addConstr(last_layer[1] >= last_layer[0], name="last_layer")
        gurobi_model.update()
        gurobi_model.optimize()
        # assert gurobi_model.status == 2, "LP wasn't optimally solved"
        return gurobi_model.status == 2 or gurobi_model.status == 5

    @staticmethod
    def generate_nn_guard_positive(gurobi_model: grb.Model, input, nn: torch.nn.Sequential,positive, M=1e2):
        gurobi_model.setParam("DualReductions", 0)
        gurobi_vars = []
        gurobi_vars.append(input)
        for i, layer in enumerate(nn):

            # print(layer)
            if type(layer) is torch.nn.Linear:
                v = gurobi_model.addMVar(lb=float("-inf"), shape=(int(layer.out_features)), name=f"layer_{i}")
                lin_expr = layer.weight.data.numpy() @ gurobi_vars[-1]
                if layer.bias is not None:
                    lin_expr = lin_expr + layer.bias.data.numpy()
                gurobi_model.addConstr(v == lin_expr, name=f"linear_constr_{i}")
                gurobi_vars.append(v)
                gurobi_model.update()
                gurobi_model.optimize()
                assert gurobi_model.status == 2, "LP wasn't optimally solved"
            elif type(layer) is torch.nn.ReLU:
                v = gurobi_model.addMVar(lb=float("-inf"), shape=gurobi_vars[-1].shape, name=f"layer_{i}")  # same shape as previous

                z = gurobi_model.addMVar(shape=gurobi_vars[-1].shape, vtype=grb.GRB.BINARY, name=f"relu_{i}")  # lb=0, ub=1,
                # gurobi_model.addConstr(v == grb.max_(0, gurobi_vars[-1]))
                gurobi_model.addConstr(v >= gurobi_vars[-1], name=f"relu_constr_1_{i}")
                gurobi_model.addConstr(v <= gurobi_vars[-1] + M * z, name=f"relu_constr_2_{i}")
                gurobi_model.addConstr(v >= 0, name=f"relu_constr_3_{i}")
                gurobi_model.addConstr(v <= M - M * z, name=f"relu_constr_4_{i}")
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
        # gurobi_model.update()
        # gurobi_model.optimize()
        # assert gurobi_model.status == 2, "LP wasn't optimally solved"
        # gurobi_model.setObjective(v[action_ego].sum(), grb.GRB.MAXIMIZE)  # maximise the output
        last_layer = gurobi_vars[-1]
        if positive:
            gurobi_model.addConstr(last_layer[0] >= 0, name="last_layer")
        else:
            gurobi_model.addConstr(last_layer[0] <= 0, name="last_layer")
        gurobi_model.update()
        gurobi_model.optimize()
        # assert gurobi_model.status == 2, "LP wasn't optimally solved"
        return gurobi_model.status == 2 or gurobi_model.status == 5
    @staticmethod
    def generate_nn_guard_pyo(model: pyo.ConcreteModel, input, nn: torch.nn.Sequential, action_ego=0, M=1e2):
        model.nn_contraints = pyo.ConstraintList()
        gurobi_vars = []
        gurobi_vars.append(input)
        for i, layer in enumerate(nn):
            if type(layer) is torch.nn.Linear:
                layer_size = int(layer.out_features)
                v = pyo.Var(range(layer_size), name=f"layer_{i}", within=pyo.Reals)
                model.add_component(name=f"layer_{i}", val=v)
                lin_expr = np.zeros(layer_size)
                weights = layer.weight.data.numpy()
                bias = 0
                if layer.bias is not None:
                    bias = layer.bias.data.numpy()
                else:
                    bias = np.zeros(layer_size)
                for j in range(layer_size):
                    res = sum(gurobi_vars[-1][k] * weights[j, k] for k in range(weights.shape[1])) + bias[j]

                for j in range(layer_size):
                    model.nn_contraints.add(v[j] == sum(gurobi_vars[-1][k] * weights[j, k] for k in range(weights.shape[1])) + bias[j])
                gurobi_vars.append(v)
            elif type(layer) is torch.nn.ReLU:
                layer_size = int(nn[i - 1].out_features)
                v = pyo.Var(range(layer_size), name=f"layer_{i}", within=pyo.PositiveReals)
                model.add_component(name=f"layer_{i}", val=v)

                z = pyo.Var(range(layer_size), name=f"relu_{i}", within=pyo.Binary)
                model.add_component(name=f"relu_{i}", val=z)
                # for j in range(layer_size):
                #     model.nn_contraints.add(expr=v[j] >= gurobi_vars[-1][j])
                #     model.nn_contraints.add(expr=v[j] <= gurobi_vars[-1][j] + M * z[j])
                #     model.nn_contraints.add(expr=v[j] >= 0)
                #     model.nn_contraints.add(expr=v[j] <= M - M * z[j])

                for j in range(layer_size):
                    # model.nn_contraints.add(expr=v[j] <= gurobi_vars[-1][j])
                    dis = gdp.Disjunction(expr=[[v[j] >= gurobi_vars[-1][j], v[j] <= gurobi_vars[-1][j], gurobi_vars[-1][j] >= 0], [v[j] == 0, gurobi_vars[-1][j] <= 0]])
                    model.add_component(f"relu_{i}_{j}", dis)
                gurobi_vars.append(v)
                """
                y = Relu(x)
                0 <= z <= 1, z is integer
                y >= x
                y <= x + Mz
                y >= 0
                y <= M - Mz"""
        for i in range(len(gurobi_vars[-1])):
            if i == action_ego:
                continue
            model.nn_contraints.add(gurobi_vars[-1][action_ego] >= gurobi_vars[-1][i])
        model.obj = pyo.Objective(expr=gurobi_vars[-1][action_ego], sense=pyo.minimize)
        TransformationFactory('gdp.bigm').apply_to(model, bigM=M)
        result = pyo.SolverFactory('glpk').solve(model)
        if (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal):
            return True
        elif (result.solver.termination_condition == TerminationCondition.infeasible):
            # log_infeasible_constraints(model)
            return False
        else:
            print(f"Solver status: {result.solver.status}")

    @staticmethod
    def generate_nn_guard_continuous(gurobi_model: grb.Model, input, nn: torch.nn.Sequential):
        gurobi_vars = []
        gurobi_vars.append(input)
        for i, layer in enumerate(nn):

            # print(layer)
            if type(layer) is torch.nn.Linear:
                v = gurobi_model.addMVar(lb=float("-inf"), shape=(int(layer.out_features)), name=f"layer_{i}")
                lin_expr = layer.weight.data.numpy() @ gurobi_vars[-1]
                if layer.bias is not None:
                    lin_expr = lin_expr + layer.bias.data.numpy()
                gurobi_model.addConstr(v == lin_expr, name=f"linear_constr_{i}")
                gurobi_vars.append(v)
            elif type(layer) is torch.nn.ReLU:
                v = gurobi_model.addMVar(lb=float("-inf"), shape=gurobi_vars[-1].shape, name=f"layer_{i}")  # same shape as previous
                z = gurobi_model.addMVar(lb=0, ub=1, shape=gurobi_vars[-1].shape, vtype=grb.GRB.INTEGER, name=f"relu_{i}")
                M = 1e3
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
            elif type(layer) is torch.nn.Hardtanh:
                layerTanh: torch.nn.Hardtanh = layer
                min_val = layerTanh.min_val
                max_val = layerTanh.max_val
                M = 10e3
                v1 = gurobi_model.addMVar(lb=float("-inf"), shape=gurobi_vars[-1].shape, name=f"layer_{i}")  # same shape as previous
                z1 = gurobi_model.addMVar(lb=0, ub=1, shape=gurobi_vars[-1].shape, vtype=grb.GRB.INTEGER, name=f"hardtanh1_{i}")
                z2 = gurobi_model.addMVar(lb=0, ub=1, shape=gurobi_vars[-1].shape, vtype=grb.GRB.INTEGER, name=f"hardtanh2_{i}")
                gurobi_model.addConstr(v1 >= gurobi_vars[-1], name=f"hardtanh1_constr_1_{i}")
                gurobi_model.addConstr(v1 <= gurobi_vars[-1] + M * z1, name=f"hardtanh1_constr_2_{i}")
                gurobi_model.addConstr(v1 >= min_val, name=f"hardtanh1_constr_3_{i}")
                gurobi_model.addConstr(v1 <= min_val + M - M * z1, name=f"hardtanh1_constr_4_{i}")
                gurobi_vars.append(v1)
                v2 = gurobi_model.addMVar(lb=float("-inf"), shape=gurobi_vars[-1].shape, name=f"layer_{i}")  # same shape as previous
                gurobi_model.addConstr(v2 <= gurobi_vars[-1], name=f"hardtanh2_constr_1_{i}")
                gurobi_model.addConstr(v2 >= gurobi_vars[-1] - M * z2, name=f"hardtanh2_constr_2_{i}")
                gurobi_model.addConstr(v2 <= max_val, name=f"hardtanh2_constr_3_{i}")
                gurobi_model.addConstr(v2 >= max_val - M + M * z2, name=f"hardtanh2_constr_4_{i}")
                gurobi_vars.append(v2)
            else:
                raise Exception("Unrecognised layer")
        gurobi_model.update()
        gurobi_model.optimize()
        assert gurobi_model.status == 2, "LP wasn't optimally solved"
        last_layer = gurobi_vars[-1]
        gurobi_model.setObjective(last_layer[0].sum(), grb.GRB.MAXIMIZE)  # maximise the output
        gurobi_model.update()
        gurobi_model.optimize()
        assert gurobi_model.status == 2, "LP wasn't optimally solved"
        max_val = gurobi_model.ObjVal
        gurobi_model.setObjective(last_layer[0].sum(), grb.GRB.MINIMIZE)  # maximise the output
        gurobi_model.update()
        gurobi_model.optimize()
        assert gurobi_model.status == 2, "LP wasn't optimally solved"
        min_val = gurobi_model.ObjVal
        return last_layer, max_val, min_val

    def generic_plot(self, title_x, title_y, vertices_list, template, template_2d):
        fig, simple_vertices = show_polygon_list3(vertices_list, title_x, title_y, template, template_2d)
        if self.show_progress_plot:
            fig.show()
        if self.save_dir is not None:
            width = 2560
            height = 1440
            scale = 1
            fig.write_image(os.path.join(self.save_dir, "plot.svg"), width=width, height=height, scale=scale)
            fig.write_image(os.path.join(self.save_dir, "plot.png"), width=width, height=height, scale=scale)
            fig.write_image(os.path.join(self.save_dir, "plot.jpeg"), width=width, height=height, scale=scale)
            fig.write_image(os.path.join(self.save_dir, "plot.pdf"), width=width, height=height, scale=scale)
            fig.write_html(os.path.join(self.save_dir, "plot.html"), include_plotlyjs="cdn")
            fig.write_json(os.path.join(self.save_dir, "plot.json"))
            with open(os.path.join(self.save_dir, "plot.csv"), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
                for timestep in simple_vertices:
                    for item in timestep:
                        # assert len(item) == 4
                        for vertex in item:
                            wr.writerow(vertex)
                        wr.writerow(item[0])  # write back the first item
                    wr.writerow("")

    def generic_plot1d(self, title_x, title_y, vertices_list, template, template_2d):
        fig, simple_vertices = show_polygon_list31d(vertices_list, title_x, title_y, template, template_2d)
        if self.show_progress_plot:
            fig.show()
        if self.save_dir is not None:
            width = 2560
            height = 1440
            scale = 1
            fig.write_image(os.path.join(self.save_dir, "plot.svg"), width=width, height=height, scale=scale)
            fig.write_image(os.path.join(self.save_dir, "plot.png"), width=width, height=height, scale=scale)
            fig.write_image(os.path.join(self.save_dir, "plot.jpeg"), width=width, height=height, scale=scale)
            fig.write_image(os.path.join(self.save_dir, "plot.pdf"), width=width, height=height, scale=scale)
            fig.write_html(os.path.join(self.save_dir, "plot.html"), include_plotlyjs="cdn")
            fig.write_json(os.path.join(self.save_dir, "plot.json"))
            with open(os.path.join(self.save_dir, "plot.csv"), 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
                for timestep in simple_vertices:
                    for item in timestep:
                        # assert len(item) == 4
                        for vertex in item:
                            wr.writerow(vertex)
                        wr.writerow(item[0])  # write back the first item
                    wr.writerow("")


def contained(x: tuple, y: tuple, eps=1e-9):
    # y contains x
    assert len(x) == len(y)
    for i in range(len(x)):
        if x[i] > y[i] + eps:
            return False
    return True

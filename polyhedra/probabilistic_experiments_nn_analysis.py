import csv
import datetime
import math
import os
import sys
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
from py4j.java_gateway import JavaGateway
from pyomo.core import TransformationFactory
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints

from polyhedra.experiments_nn_analysis import Experiment
from polyhedra.partitioning import sample_and_split, pick_longest_dimension, split_polyhedron, create_range_bounds_model, acceptable_range
from polyhedra.plot_utils import show_polygon_list3, show_polygon_list31d
from polyhedra.prism_methods import calculate_target_probabilities, recreate_prism_PPO
from utility.standard_progressbar import StandardProgressBar


class ProbabilisticExperiment(Experiment):
    def __init__(self, env_input_size: int):
        super().__init__(env_input_size)
        self.use_entropy_split = True

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

    class LoopStats():
        def __init__(self):
            self.seen = []
            self.frontier = []  # contains the elements to explore
            self.vertices_list = defaultdict(list)
            self.max_t = 0
            self.num_already_visited = 0
            self.proc_ids = []
            self.last_time_plot = None
            self.exit_flag = False
            self.is_agent_unsafe = False
            self.root = None

    def main_loop(self, nn, template, template_2d):
        root = self.generate_root_polytope()
        root_pair = (root, 0)  # label for root is always 0
        root_list = [root_pair]
        stats = ProbabilisticExperiment.LoopStats()
        stats.root = root
        if self.additional_seen_fn is not None:
            for extra in self.additional_seen_fn():
                stats.seen.append(extra)
        stats.frontier = [(0, x) for x in root_list]
        if self.graph is not None:
            self.graph.add_node(root_pair)
        widgets = [progressbar.Variable('n_workers'), ', ', progressbar.Variable('frontier'), ', ', progressbar.Variable('seen'), ', ', progressbar.Variable('num_already_visited'), ", ",
                   progressbar.Variable('max_t'), ", ", progressbar.Variable('last_visited_state')]
        if self.before_start_fn is not None:
            self.before_start_fn(nn)
        with progressbar.ProgressBar(widgets=widgets) if self.show_progressbar else nullcontext() as bar_main:
            while len(stats.frontier) != 0 or len(stats.proc_ids) != 0:
                self.inner_loop_step(stats, template_2d, template, nn, bar_main)
        self.plot_fn(stats.vertices_list, template, template_2d)
        return stats.max_t, stats.num_already_visited, stats.vertices_list, stats.is_agent_unsafe

    @ray.remote
    def post_milp(self, x, x_label, nn, output_flag, t, template):
        raise NotImplementedError()  # need to implement this method, remember to put @ray.remote as an attribute

    def inner_loop_step(self, stats: LoopStats, template_2d, template, nn, bar_main):
        # fills up the worker threads
        while len(stats.proc_ids) < self.n_workers and len(stats.frontier) != 0:
            t, (x, x_label) = stats.frontier.pop(0) if self.use_bfs else stats.frontier.pop()
            if stats.max_t > self.time_horizon:
                print(f"Reached horizon t={t}")
                stats.exit_flag = True
                break
            contained_flag = False
            to_remove = []
            for (s, s_label) in stats.seen:
                if s_label == x_label:
                    if contained(x, s):
                        contained_flag = True
                        break
                    if contained(s, x):
                        to_remove.append((s, s_label))
            for rem in to_remove:
                stats.num_already_visited += 1
                stats.seen.remove(rem)
            if contained_flag:
                stats.num_already_visited += 1
                continue
            stats.max_t = max(stats.max_t, t)
            stats.vertices_list[t].append(np.array(x))
            stats.seen.append((x, x_label))

            split_proc_id = self.check_split(t, x, nn, bar_main, stats, template, template_2d)
            if split_proc_id is not None:  # splitting
                stats.proc_ids.append(split_proc_id)
            else:  # calculate successor
                stats.proc_ids.append(self.post_fn_remote.remote(self, x, x_label, nn, self.output_flag, t, template))
            if self.show_progressbar:
                bar_main.update(value=bar_main.value + 1, n_workers=len(stats.proc_ids), seen=len(stats.seen), frontier=len(stats.frontier), num_already_visited=stats.num_already_visited,
                                last_visited_state=str(x),
                                max_t=stats.max_t)

        if stats.last_time_plot is None or time.time() - stats.last_time_plot >= self.plotting_time_interval:
            if stats.last_time_plot is not None:
                self.plot_fn(stats.vertices_list, template, template_2d)
            stats.last_time_plot = time.time()
        if self.update_progress_fn is not None:
            self.update_progress_fn(n_workers=len(stats.proc_ids), seen=len(stats.seen), frontier=len(stats.frontier), num_already_visited=stats.num_already_visited, max_t=stats.max_t)

        # process the results (if any)
        new_frontier = self.collect_results(stats, template)
        # update prism
        self.update_prism_step(stats.frontier, new_frontier, stats.root)
        stats.new_frontier = []  # resets the new_frontier

    def collect_results(self, stats, template):
        """collects the results from the various processes, creates a list with the newly added states"""
        new_frontier = []
        ready_ids, stats.proc_ids = ray.wait(stats.proc_ids, num_returns=len(stats.proc_ids), timeout=0.5)
        if len(ready_ids) != 0:
            results_list = ray.get(ready_ids)
            assert len(results_list) != 0, "something is wrong with the calculation of the successor"
            for results in results_list:
                for x_prime, (parent, parent_lbl), t_prime in results:  # these are the individual elements in the list which is returned by post_milp
                    x_prime_label = self.assign_lbl_fn(x_prime, parent, parent_lbl)
                    successor = (x_prime, x_prime_label)
                    if self.use_rounding:
                        x_prime_rounded = self.round_tuple(x_prime, self.rounding_value)

                        assert contained(x_prime, x_prime_rounded), f"{x_prime} not contained in {x_prime_rounded}"  # x_prime_rounded should always be bigger than x_prime
                        x_prime = x_prime_rounded
                    stats.frontier = [(u, (y, y_label)) for u, (y, y_label) in stats.frontier if not (y_label == x_prime_label and contained(y, x_prime))]
                    # ---check contained
                    if not any([(y_label == x_prime_label and contained(x_prime, y)) for u, (y, y_label) in stats.frontier]):  # todo in the case in which the state is contained, put it in the graph
                        unsafe = self.check_unsafe(template, x_prime, self.unsafe_zone)
                        if not unsafe:
                            new_frontier.append((t_prime, successor))  # put the item back into the queue
                        if self.graph is not None:
                            self.graph.add_edge((parent, parent_lbl), successor)  # todo add action and probability ranges
                        if unsafe:
                            self.graph.nodes[successor]["safe"] = False
                            stats.seen.append(successor)
                        # print(x_prime)
                    else:
                        stats.num_already_visited += 1
        return new_frontier

    def update_prism_step(self, frontier, new_frontier, root):
        gateway, mc, mdp, mapping = recreate_prism_PPO(self.graph, root)
        with StandardProgressBar(prefix="Updating probabilities ", max_value=len(new_frontier) + 1).start() as bar:
            for t, x in new_frontier:
                try:
                    minmin, minmax, maxmin, maxmax = calculate_target_probabilities(gateway, mc, mdp, mapping, targets=[x])
                    bar.update(bar.value + 1)
                except Exception as e:
                    print("warning")
                if maxmax > 1e-6:  # check if state is irrelevant
                    frontier.append((t, x))  # next computation only considers relevant states
                else:
                    self.graph.nodes[x]["irrelevant"] = True



    def check_split(self, t, x, nn, bar_main, stats, template, template_2d) -> List:
        # -------splitting
        ranges_probs = create_range_bounds_model(template, x, self.env_input_size, nn)
        split_flag = acceptable_range(ranges_probs)
        new_frontier = []
        if split_flag:
            bar_main.update(value=bar_main.value + 1, current_t=t, last_action="split", last_polytope=str(x))
            to_split = []
            to_split.append(x)
            bar_main.update(force=True)
            bar_main.fd.flush()
            print("", file=sys.stderr)  # new line
            # new_frontier = pickle.load(open("new_frontier.p","rb"))
            widgets = [progressbar.Variable('splitting_queue'), ", ", progressbar.Variable('frontier_size'), ", ", progressbar.widgets.Timer()]
            with progressbar.ProgressBar(prefix=f"Splitting states: ", widgets=widgets, is_terminal=True, term_width=200, redirect_stdout=True).start() as bar_main:
                while len(to_split) != 0:
                    # bar_main.update(value=bar_main.value + 1, splitting_queue=len(to_split), frontier_size=len(new_frontier))
                    to_analyse = to_split.pop()
                    if self.use_entropy_split:
                        split1, split2 = sample_and_split(nn, template, np.array(to_analyse), self.env_input_size)
                    else:
                        dimension = pick_longest_dimension(template, x)
                        split1, split2 = split_polyhedron(template, x, dimension)
                    ranges_probs1 = create_range_bounds_model(template, split1, self.env_input_size, nn)
                    split_flag1 = acceptable_range(ranges_probs1)
                    if split_flag1:
                        to_split.append(split1)
                    else:
                        new_frontier.insert(0, split1)
                        # plot_frontier(new_frontier)
                        self.graph.add_edge(x, split1, action="split")
                    ranges_probs2 = create_range_bounds_model(template, split2, self.env_input_size, nn)
                    split_flag2 = acceptable_range(ranges_probs2)
                    if split_flag2:
                        to_split.append(split2)
                    else:
                        new_frontier.insert(0, split2)
                        # plot_frontier(new_frontier)
                        self.graph.add_edge(x, split2, action="split")
            # print("finished splitting")
            # ----for plotting
            # colours = []
            # for x in new_frontier:
            #     ranges_probs1 = create_range_bounds_model(template, x, self.env_input_size, nn)
            #     colours.append(np.mean(ranges_probs1[0]))
            # print("", file=sys.stderr)  # new line
            # fig = show_polygons(template, [x[1] for x in new_frontier], template_2d, colours)
            # fig.write_html("new_frontier.html")
            # print("", file=sys.stderr)  # new line
        return []

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
    def generate_nn_guard_positive(gurobi_model: grb.Model, input, nn: torch.nn.Sequential, positive, M=1e2):
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

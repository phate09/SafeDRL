import csv
import datetime
import heapq
import math
import os
import pickle
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Tuple, List

import gurobipy as grb
import networkx
import numpy as np
import progressbar
import ray
import torch

from environment.stopping_car import StoppingCar
from polyhedra.milp_methods import generate_input_region, optimise, generate_region_constraints
from polyhedra.partitioning import split_polyhedron_milp, find_inverted_dimension
from polyhedra.plot_utils import show_polygon_list3, show_polygon_list31d
from polyhedra.runnable.templates import polytope
from symbolic import unroll_methods


class Experiment():
    def __init__(self, env_input_size: int):
        self.before_start_fn = None
        self.update_progress_fn = None
        self.rounding_value = 1024
        self.get_nn_fn = None
        self.plot_fn = None
        self.post_fn_remote = None
        self.assign_lbl_fn = self.assign_lbl_dummy
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
        self.save_graph = True
        self.use_split = True  # enable/disable splitting
        self.load_graph = False
        self.use_abstract_mapping = False  # decide whether to use the precomputed abstract mapping or compute it on the fly
        self.use_contained = True  # enable/disable containment check
        self.use_split_with_seen = True  # enable/disable splitting polytopes when they are partially contained within previously visited abstract states
        self.keep_model = False  # whether to keep the gurobi model for later timesteps
        self.graph = networkx.DiGraph()  # set None to disable use of graph

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
        assert self.save_dir is not None
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
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

    def assign_lbl_dummy(self, x_prime, parent, parent_lbl):
        """assigns the same label to every state"""
        return 0

    def create_range_bounds_model(self, template, x, env_input_size, nn, round=-1):
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', False)
        input = generate_input_region(gurobi_model, template, x, env_input_size)
        gurobi_model.update()
        gurobi_model.optimize()
        assert gurobi_model.status == 2, "LP wasn't optimally solved"
        observation = self.get_observation_variable(input, gurobi_model)  # get the observation from the input
        ranges = Experiment.get_range_bounds(observation, nn, gurobi_model)
        ranges_probs = unroll_methods.softmax_interval(ranges)
        if round >= 0:
            pass
            # todo round the probabilities
        return ranges_probs

    @staticmethod
    def get_range_bounds(input, nn, gurobi_model, M=1e6):
        gurobi_vars = []
        gurobi_vars.append(input)
        Experiment.build_nn_model_core(gurobi_model, gurobi_vars, nn, M)
        last_layer = gurobi_vars[-1]
        ranges = []
        for i in range(last_layer.shape[0]):
            gurobi_model.setObjective(last_layer[i].sum(), grb.GRB.MAXIMIZE)  # maximise the output
            gurobi_model.update()
            gurobi_model.optimize()
            ub = last_layer[i].X[0]
            gurobi_model.setObjective(last_layer[i].sum(), grb.GRB.MINIMIZE)  # maximise the output
            gurobi_model.update()
            gurobi_model.optimize()
            lb = last_layer[i].X[0]
            ranges.append((lb, ub))
        return ranges

    def get_observation_variable(self, input, gurobi_model):
        """to override if the obsevation variable is different from the input (e.g. adversarial examples)"""
        return input

    def main_loop(self, nn, template, template_2d):
        root = self.generate_root_polytope()
        root_pair = (root, 0)  # label for root is always 0
        root_list = [root_pair]
        if not self.load_graph:
            stats = Experiment.LoopStats()
            stats.root = root_pair
            stats.start_time = datetime.datetime.now()
            if self.additional_seen_fn is not None:
                for extra in self.additional_seen_fn():
                    stats.seen.append(extra)
            stats.frontier = [(0, x) for x in root_list]
            if self.graph is not None:
                self.graph.add_node(root_pair)
            widgets = [progressbar.Variable('n_workers'), ', ', progressbar.Variable('frontier'), ', ', progressbar.Variable('seen'), ', ', progressbar.Variable('num_already_visited'), ", ",
                       progressbar.Variable('max_t'), ", ", progressbar.widgets.Timer()]
            if self.before_start_fn is not None:
                self.before_start_fn(nn)
            with progressbar.ProgressBar(widgets=widgets) if self.show_progressbar else nullcontext() as bar_main:
                while len(stats.frontier) != 0 or len(stats.proc_ids) != 0:
                    self.inner_loop_step(stats, template_2d, template, nn, bar_main)
            self.plot_fn(stats.vertices_list, template, template_2d)
            # gateway, mc, mdp, mapping = recreate_prism_PPO(self.graph, root_pair)
            # inv_map = {v: k for k, v in mapping.items()}
            stats.end_time = datetime.datetime.now()
            networkx.write_gpickle(self.graph, os.path.join(self.save_dir, "graph.p"))
            pickle.dump(stats, open(os.path.join(self.save_dir, "stats.p"), "wb"))
        else:
            self.graph = networkx.read_gpickle(os.path.join(self.save_dir, "graph.p"))
            stats: Experiment.LoopStats = pickle.load(open(os.path.join(self.save_dir, "stats.p"), "rb"))
            # gateway, mc, mdp, mapping = recreate_prism_PPO(self.graph, root_pair)
        return stats.max_t, stats.num_already_visited, stats.vertices_list, stats.is_agent_unsafe

    class LoopStats():
        """Class that contains some information to share inbetween loops"""

        def __init__(self):
            self.seen = []
            self.frontier = []  # contains the elements to explore
            self.to_replace_split = []  # elements to replace/split because they are too big
            self.vertices_list = defaultdict(list)  # contains point at every timesteps, this is for plotting purposes only
            self.max_t = 0
            self.num_already_visited = 0
            self.num_irrelevant = 0
            self.proc_ids = []
            self.last_time_plot = None
            self.exit_flag = False
            self.is_agent_unsafe = False
            self.discarded = []  # list of elements after the maximum horizon, used to restart the computation (if we wanted to expand the horizon)
            self.root = None
            self.elapsed_time = 0
            self.start_time = None
            self.last_time = None
            self.end_time = None

    def inner_loop_step(self, stats: LoopStats, template_2d, template, nn, bar_main):
        # fills up the worker threads
        while len(stats.proc_ids) < self.n_workers and len(stats.frontier) != 0:
            t, (x, x_label) = heapq.heappop(stats.frontier) if self.use_bfs else stats.frontier.pop()
            if t >= self.time_horizon:
                print(f"Discard timestep t={t}")
                stats.discarded.append((x, x_label))
                continue
            stats.max_t = max(stats.max_t, t)

            if self.use_contained:
                contained_flag = False
                to_remove = []
                for (s, s_label) in stats.seen:
                    if s_label == x_label:
                        if contained(x, s):
                            if not self.graph.has_predecessor((x, x_label), (s, x_label)):  # ensures that if there was a split it doesn't count as contained
                                self.graph.add_edge((x, x_label), (s, x_label), action="contained", lb=1.0, ub=1.0)
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
            stats.seen.append((x, x_label))
            if self.show_progressbar:
                bar_main.update(value=bar_main.value + 1, n_workers=len(stats.proc_ids), seen=len(stats.seen), frontier=len(stats.frontier), num_already_visited=stats.num_already_visited,
                                # elapsed_time=(datetime.datetime.now()-stats.start_time).total_seconds()/60.0,
                                max_t=stats.max_t)
            # if self.use_split:
            #     if self.use_abstract_mapping:
            #         pass
            #         # splitted_elements = self.split_item_abstract_mapping(x, [m[0] for m in self.abstract_mapping])
            #         # n_fragments = len(splitted_elements)
            #         # if n_fragments > 1:
            #         #     new_fragments = []
            #         #     stats.seen.remove((x, x_label))  # remove the parent node from seen if it has been split to prevent unnecessary loops when we check for containment
            #         #     for i, splitted_polytope in enumerate(splitted_elements):
            #         #         successor_info = Experiment.SuccessorInfo()
            #         #         successor_info.successor = tuple(splitted_polytope)
            #         #         successor_info.parent = x
            #         #         successor_info.parent_lbl = x_label
            #         #         successor_info.t = t
            #         #         successor_info.action = f"split{i}"
            #         #         new_fragments.append(successor_info)
            #         #         # todo probably include also the probabilities associated with each fragment
            #         #     stats.proc_ids.append(ray.put(new_fragments))
            #         #     continue
            #     else:  # split on the go
            #         if len(list(self.graph.in_edges((x, x_label)))) == 0 or not "split" in self.graph.edges[list(self.graph.in_edges((x, x_label)))[0]].get("action"):
            #             if self.can_be_splitted(template, x):
            #                 splitted_elements = self.check_split(t, x, x_label, nn, bar_main, stats, template, template_2d)
            #                 n_fragments = len(splitted_elements)
            #                 if n_fragments > 1:
            #                     new_fragments = []
            #                     stats.seen.remove((x, x_label))  # remove the parent node from seen if it has been split to prevent unnecessary loops when we check for containment
            #                     for i, (splitted_polytope, probs_range) in enumerate(splitted_elements):
            #                         successor_info = Experiment.SuccessorInfo()
            #                         successor_info.successor = tuple(splitted_polytope)
            #                         successor_info.parent = x
            #                         successor_info.parent_lbl = x_label
            #                         successor_info.t = t
            #                         successor_info.action = f"split{i}"
            #                         new_fragments.append(successor_info)
            #                     stats.proc_ids.append(ray.put(new_fragments))
            #                     continue

            if self.use_split_with_seen:  # split according to the seen elements
                splitted_elements2 = self.split_item_abstract_mapping(x, [m[0] for m in stats.seen])  # splits according to the seen list
                n_fragments = len(splitted_elements2)
                if self.use_split_with_seen and n_fragments > 1:
                    new_fragments = []
                    stats.seen.remove((x, x_label))  # remove the parent node from seen if it has been split to prevent unnecessary loops when we check for containment
                    for i, splitted_polytope in enumerate(splitted_elements2):
                        successor_info = Experiment.SuccessorInfo()
                        successor_info.successor = tuple(splitted_polytope)
                        successor_info.parent = x
                        successor_info.parent_lbl = x_label
                        successor_info.t = t
                        successor_info.action = f"split{i}"
                        new_fragments.append(successor_info)
                    stats.proc_ids.append(ray.put(new_fragments))
                    continue
            # if nothing else applies, compute the successor
            stats.proc_ids.append(self.post_fn_remote.remote(self, x, x_label, nn, self.output_flag, t, template))  # compute successors

        if stats.last_time_plot is None or time.time() - stats.last_time_plot >= self.plotting_time_interval:
            if stats.last_time_plot is not None:
                self.plot_fn(stats.vertices_list, template, template_2d)
            stats.last_time_plot = time.time()
        if self.update_progress_fn is not None:
            self.update_progress_fn(n_workers=len(stats.proc_ids), seen=len(stats.seen), frontier=len(stats.frontier), num_already_visited=stats.num_already_visited, max_t=stats.max_t)

        # process the results (if any)
        new_frontier = self.collect_results(stats, template)

        # if self.avoid_irrelevants:
        #     # update prism
        #     self.update_prism_step(stats.frontier, new_frontier, stats.root, stats)
        # else:
        if self.use_bfs:
            for element in new_frontier:
                heapq.heappush(stats.frontier, element)
        else:
            stats.frontier.extend(new_frontier)
        stats.new_frontier = []  # resets the new_frontier
        if self.save_graph:
            stats.last_time_save = datetime.datetime.now()
            networkx.write_gpickle(self.graph, os.path.join(self.save_dir, "graph.p"))
            pickle.dump(stats, open(os.path.join(self.save_dir, "stats.p"), "wb"))

    def collect_results(self, stats, template):
        """collects the results from the various processes, creates a list with the newly added states"""
        new_frontier = []
        ready_ids, stats.proc_ids = ray.wait(stats.proc_ids, num_returns=len(stats.proc_ids), timeout=0.5)
        if len(ready_ids) != 0:
            results_list = ray.get(ready_ids)
            assert len(results_list) != 0, "something is wrong with the calculation of the successor"
            for results in results_list:
                successor_info: Experiment.SuccessorInfo
                for successor_info in results:  # these are the individual elements in the list which is returned by post_milp
                    successor_info.successor_lbl = self.assign_lbl_fn(successor_info.successor, successor_info.parent, successor_info.parent_lbl)
                    if self.use_rounding:
                        x_prime_rounded = self.round_tuple(successor_info.successor, self.rounding_value)

                        assert contained(successor_info.successor,
                                         x_prime_rounded), f"{successor_info.successor} not contained in {x_prime_rounded}"  # x_prime_rounded should always be bigger than x_prime
                        successor_info.successor = x_prime_rounded
                    # stats.frontier = [(u, (y, y_label)) for u, (y, y_label) in stats.frontier if
                    #                   not (y_label == successor_info.successor_lbl and contained(y, successor_info.successor))]  # remove from frontier if the successor is bigger
                    is_splitted = "split" in successor_info.action
                    # ---check splitted
                    if is_splitted:
                        if successor_info.get_parent_node() in stats.seen:
                            stats.seen.remove(successor_info.get_parent_node())  # remove the parent node from seen if it has been split to prevent unnecessary loops when we check for containment
                    else:
                        stats.vertices_list[successor_info.t].append(successor_info.get_successor_node())
                    unsafe = self.check_unsafe(template, successor_info.successor, self.unsafe_zone)
                    self.graph.add_edge(successor_info.get_parent_node(), successor_info.get_successor_node(), action=successor_info.action, lb=successor_info.lb, ub=successor_info.ub)
                    if not unsafe:
                        new_frontier.append((successor_info.t, successor_info.get_successor_node()))  # put the item back into the queue
                    else:  # unsafe
                        self.graph.nodes[successor_info.get_successor_node()]["safe"] = False
                        stats.seen.append(successor_info.get_successor_node())  # todo are we sure we want to keep track of unsafe elements?
                        # print(x_prime)
        return new_frontier

    def split_item_abstract_mapping(self, current_polytope, abstract_mapping):
        to_split = [current_polytope]
        processed = []
        relevant_polytopes = self.find_relevant_polytopes(current_polytope, abstract_mapping)

        widgets = [progressbar.Variable('splitting_queue'), ", ", progressbar.Variable('frontier_size'), ", ", progressbar.widgets.Timer()]
        # with progressbar.ProgressBar(prefix=f"Splitting states: ", widgets=widgets, is_terminal=True, term_width=200, redirect_stdout=True).start() as bar_split:
        while len(to_split) > 0:
            current_polytope = to_split.pop()
            split_happened = False
            for x in relevant_polytopes:
                choose = self.check_intersection(x, current_polytope)
                if choose:
                    dimensions_volume = self.find_important_dimensions(current_polytope, x)
                    dimensions_volume = [x for x in dimensions_volume if find_inverted_dimension(-self.analysis_template[x[0]], self.analysis_template) != -1]
                    if len(dimensions_volume) > 0:
                        assert len(dimensions_volume) > 0
                        min_idx = dimensions_volume[np.argmin(np.array(dimensions_volume)[:, 1])][0]
                        splits = split_polyhedron_milp(self.analysis_template, current_polytope, min_idx, x[min_idx])
                        to_split.append(tuple(splits[0]))
                        to_split.append(tuple(splits[1]))
                        split_happened = True
                        break
                # bar_split.update(splitting_queue=len(to_split), frontier_size=len(processed))
            if not split_happened:
                processed.append(current_polytope)

        # colours = []
        # for x, ranges_probs in frontier:
        #     colours.append(np.mean(ranges_probs[0]))
        # print("", file=sys.stderr)  # new line
        # fig = show_polygons(template, [x[0] for x in frontier] + to_split + processed, self.template_2d, colours + [0.5] * len(to_split) + [0.5] * len(processed))
        return processed

    def find_important_dimensions(self, poly1, poly2):
        '''assuming check_contained(poly1,poly2) returns true, we are interested in the halfspaces that matter in terms of splitting of poly1
        poly1 = root, poly2 = candidate
        '''
        # #Binary Space Partitioning
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', self.output_flag)
        input1 = generate_input_region(gurobi_model, self.analysis_template, poly1, self.env_input_size)
        relevant_directions = []
        for j, dimension in enumerate(self.analysis_template):
            # inverted_dimension = find_inverted_dimension(-dimension, self.analysis_template)
            # if inverted_dimension == -1:
            #     continue
            multiplication = 0
            for i in range(self.env_input_size):
                multiplication += dimension[i] * input1[i]
            previous_constraint = gurobi_model.getConstrByName("check_contained_constraint")
            if previous_constraint is not None:
                gurobi_model.remove(previous_constraint)
                gurobi_model.update()
            gurobi_model.addConstr(multiplication <= poly2[j], name=f"check_contained_constraint")
            gurobi_model.update()
            x_results = optimise(self.analysis_template, gurobi_model, input1)
            if np.allclose(np.array(poly1), x_results) is False:
                samples = polytope.sample(1000, self.analysis_template, x_results)
                from scipy.spatial import ConvexHull
                hull = ConvexHull(samples)
                volume = hull.volume  # estimated volume from points helps prioritise halfspaces
                relevant_directions.append((j, volume))
        return relevant_directions

    def find_relevant_polytopes(self, current_polytope, abstract_mapping):
        '''finds which elements of abstract_mapping are relevant to current_polytope'''
        new_frontier = []
        for x in abstract_mapping:
            if x == current_polytope:  # ignore itself
                continue
            choose = self.check_intersection(x, current_polytope)
            if choose:
                new_frontier.append(x)
        frontier = new_frontier  # shrink the selection of polytopes to only the ones which are relevant
        return frontier

    def check_intersection(self, poly1, poly2, eps=1e-5):
        '''checks if a polytope is inside another (even partially)'''
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', self.output_flag)
        input1 = generate_input_region(gurobi_model, self.analysis_template, poly1, self.env_input_size)
        generate_region_constraints(gurobi_model, self.analysis_template, input1, poly2, self.env_input_size, eps=eps)  # use epsilon to prevent single points
        x_results = optimise(self.analysis_template, gurobi_model, input1)
        if x_results is None:
            # not contained
            return False
        else:
            return True

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
        input = generate_input_region(gurobi_model, self.input_template, self.input_boundaries, self.env_input_size)
        x_results = optimise(self.analysis_template, gurobi_model, input)
        if x_results is None:
            print("Model unsatisfiable")
            return None
        root = tuple(x_results)
        return root

    def check_unsafe(self, template, bnds, x_label):
        for A, b in self.unsafe_zone:
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', False)
            input = gurobi_model.addMVar(shape=(self.env_input_size,), lb=float("-inf"), name="input")
            generate_region_constraints(gurobi_model, template, input, bnds, self.env_input_size)
            generate_region_constraints(gurobi_model, A, input, b, self.env_input_size)
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
        x_prime_results = optimise(template, gurobi_model, x_prime)  # h representation
        return x_prime_results is not None, x_prime_results

    @staticmethod
    def generate_nn_guard(gurobi_model: grb.Model, input, nn: torch.nn.Sequential, action_ego=0, M=1e2):
        gurobi_model.setParam("DualReductions", 0)
        gurobi_vars = []
        gurobi_vars.append(input)
        Experiment.build_nn_model_core(gurobi_model, gurobi_vars, nn, M)
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
    def build_nn_model_core(gurobi_model, gurobi_vars, nn, M):
        """Construct the MILP model for the body of the neural network (no last layer)"""
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
            elif type(layer) is torch.nn.Hardtanh:
                layerTanh: torch.nn.Hardtanh = layer
                min_val = layerTanh.min_val
                max_val = layerTanh.max_val
                # M = 10e3
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
        gurobi_model.update()
        gurobi_model.optimize()
        assert gurobi_model.status == 2, "LP wasn't optimally solved"

    @staticmethod
    def generate_nn_guard_positive(gurobi_model: grb.Model, input, nn: torch.nn.Sequential, positive, M=1e2, eps=0):
        gurobi_model.setParam("DualReductions", 0)
        gurobi_vars = []
        gurobi_vars.append(input)
        Experiment.build_nn_model_core(gurobi_model, gurobi_vars, nn, M)
        last_layer = gurobi_vars[-1]
        if positive:
            constraint = gurobi_model.addConstr(last_layer[0] + eps >= 0, name="last_layer")
        else:
            constraint = gurobi_model.addConstr(last_layer[0] + eps <= 0, name="last_layer")
        gurobi_model.update()
        gurobi_model.optimize()
        feasible = gurobi_model.status == 2 or gurobi_model.status == 5
        gurobi_model.remove(constraint)
        gurobi_model.update()
        gurobi_model.optimize()
        assert gurobi_model.status == 2, "LP wasn't optimally solved"
        return feasible

    @staticmethod
    def generate_nn_guard_continuous(gurobi_model: grb.Model, input, nn: torch.nn.Sequential, M=1e3):
        gurobi_vars = []
        gurobi_vars.append(input)
        Experiment.build_nn_model_core(gurobi_model, gurobi_vars, nn, M)
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



    class SuccessorInfo:
        def __init__(self):
            self.action = None
            self.lb = 1.0
            self.ub = 1.0
            self.parent = None
            self.parent_lbl = None
            self.successor = None
            self.successor_lbl = None
            self.t = None

        def get_successor_node(self):
            return self.successor, self.successor_lbl

        def get_parent_node(self):
            return self.parent, self.parent_lbl


def contained(x: tuple, y: tuple, eps=1e-9):
    # y contains x
    assert len(x) == len(y)
    for i in range(len(x)):
        if x[i] > y[i] + eps:
            return False
    return True

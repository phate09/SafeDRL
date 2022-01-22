import csv
import datetime
import heapq
import math
import os
import pickle
import sys
import time
from contextlib import nullcontext
from typing import Tuple, List, Any

import gurobipy as grb
import networkx
import numpy as np
import progressbar
import ray
import torch

from polyhedra.experiments_nn_analysis import Experiment
from polyhedra.milp_methods import generate_input_region, optimise, generate_region_constraints
from polyhedra.partitioning import sample_and_split, is_split_range, split_polyhedron_milp, find_inverted_dimension
from polyhedra.plot_utils import show_polygon_list3, show_polygon_list31d, show_polygons
from polyhedra.prism_methods import calculate_target_probabilities, recreate_prism_PPO, extract_probabilities
from runnables.runnable.templates import polytope
from utility.standard_progressbar import StandardProgressBar


class ProbabilisticExperiment(Experiment):
    def __init__(self, env_input_size: int):
        super().__init__(env_input_size)
        self.max_t_split = -1  # limits the splitting to a given timestep threshold
        self.minimum_length = 0.1
        self.use_entropy_split = True

        self.max_probability_split = 0.33
        self.save_graph = True
        self.rgb_plot = False  # enable/disable plotting probabilities as rgb (for 3 action agent)
        self.avoid_irrelevants = False  # enable/disable pruning of irrelevant (p<1e-6)
        self.abstract_mapping_path = "/home/edoardo/Development/SafeDRL/runnables/verification_runs/new_frontier3.p"
        self.abstract_mapping = []

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

    def load_abstract_mapping(self):
        self.abstract_mapping = pickle.load(open(self.abstract_mapping_path, "rb"))

    def main_loop(self, nn, template, template_2d):
        self.load_abstract_mapping()  # loads the mapping
        root = self.generate_root_polytope()
        root_pair = (root, 0)  # label for root is always 0
        root_list = [root_pair]
        if not self.load_graph:
            stats = ProbabilisticExperiment.LoopStats()
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
            gateway, mc, mdp, mapping = recreate_prism_PPO(self.graph, root_pair)
            inv_map = {v: k for k, v in mapping.items()}
            stats.end_time = datetime.datetime.now()
            networkx.write_gpickle(self.graph, os.path.join(self.save_dir, "graph.p"))
            pickle.dump(stats, open(os.path.join(self.save_dir, "stats.p"), "wb"))
        else:
            self.graph = networkx.read_gpickle(os.path.join(self.save_dir, "graph.p"))
            stats: ProbabilisticExperiment.LoopStats = pickle.load(open(os.path.join(self.save_dir, "stats.p"), "rb"))
            gateway, mc, mdp, mapping = recreate_prism_PPO(self.graph, root_pair)

        # ----for plotting
        colours = []
        to_plot = list(self.graph.successors(root_pair))
        # to_plot = stats.vertices_list[1]
        # to_plot = [root_pair]
        bad_nodes = []
        for x in self.graph.adj:
            safe = self.graph.nodes[x].get("safe")
            irrelevant = self.graph.nodes[x].get("irrelevant")
            if safe is not None:
                if safe is False:
                    bad_nodes.append(x)
            if irrelevant is not None:
                if irrelevant is True:
                    bad_nodes.append(x)
        terminal_states_ids = [mapping[x] for x in bad_nodes]
        for node in to_plot:
            starting_id = mapping[node]
            sum = 0
            # for terminal_state_id in terminal_states_ids:
            #     minmin, minmax, maxmin, maxmax = extract_probabilities([terminal_state_id], gateway, mc, mdp, root_id=starting_id)
            #     sum+=maxmax
            minmin, minmax, maxmin, maxmax = extract_probabilities(terminal_states_ids, gateway, mc, mdp, root_id=starting_id)
            # colours.append(np.mean([maxmin, maxmax]))
            colours.append(maxmax)
        # for x,x_label in to_plot:
        #     ranges_probs1 = self.create_range_bounds_model(template, x, self.env_input_size, nn)
        #     colours.append(np.mean(ranges_probs1[0]))
        print("", file=sys.stderr)  # new line
        fig = show_polygons(template, [x[0] for x in to_plot], self.template_2d, colours)
        fig.write_html("initial_state_unsafe_prob.html")
        fig.show()
        print("", file=sys.stderr)  # new line
        return stats.max_t, stats.num_already_visited, stats.vertices_list, stats.is_agent_unsafe

    @ray.remote
    def post_milp(self, x, x_label, nn, output_flag, t, template):
        raise NotImplementedError()  # need to implement this method, remember to put @ray.remote as an attribute

    def inner_loop_step(self, stats: Experiment.LoopStats, template_2d, template, nn, bar_main):
        # fills up the worker threads
        while len(stats.proc_ids) < self.n_workers and len(stats.frontier) != 0:
            t, (x, x_label) = heapq.heappop(stats.frontier) if self.use_bfs else stats.frontier.pop()
            if t >= self.time_horizon or (datetime.datetime.now() - stats.start_time).seconds / 60 > self.max_elapsed_time:
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
            if self.use_split:
                if self.max_t_split < 0 or t < self.max_t_split:  # limits the splitting to a given timestep
                    if self.use_abstract_mapping:
                        splitted_elements = self.split_item_abstract_mapping(x, [m[0] for m in self.abstract_mapping])
                        n_fragments = len(splitted_elements)
                        if n_fragments > 1:
                            new_fragments = []
                            stats.seen.remove((x, x_label))  # remove the parent node from seen if it has been split to prevent unnecessary loops when we check for containment
                            for i, splitted_polytope in enumerate(splitted_elements):
                                successor_info = Experiment.SuccessorInfo()
                                successor_info.successor = tuple(splitted_polytope)
                                successor_info.parent = x
                                successor_info.parent_lbl = x_label
                                successor_info.t = t
                                successor_info.action = f"split{i}"
                                new_fragments.append(successor_info)
                            stats.proc_ids.append(ray.put(new_fragments))
                            continue
                    else:  # split on the go
                        if len(list(self.graph.in_edges((x, x_label)))) == 0 or not "split" in self.graph.edges[list(self.graph.in_edges((x, x_label)))[0]].get("action"):
                            if self.can_be_splitted(template, x):
                                splitted_elements = self.check_split(t, x, x_label, nn, bar_main, stats, template, template_2d)
                                n_fragments = len(splitted_elements)
                                if n_fragments > 1:
                                    new_fragments = []
                                    stats.seen.remove((x, x_label))  # remove the parent node from seen if it has been split to prevent unnecessary loops when we check for containment
                                    for i, (splitted_polytope, probs_range) in enumerate(splitted_elements):
                                        successor_info = Experiment.SuccessorInfo()
                                        successor_info.successor = tuple(splitted_polytope)
                                        successor_info.parent = x
                                        successor_info.parent_lbl = x_label
                                        successor_info.t = t
                                        successor_info.action = f"split{i}"
                                        new_fragments.append(successor_info)
                                    stats.proc_ids.append(ray.put(new_fragments))
                                    continue

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

        if self.avoid_irrelevants:
            # update prism
            self.update_prism_step(stats.frontier, new_frontier, stats.root, stats)
        else:
            if self.use_bfs:
                for element in new_frontier:
                    heapq.heappush(stats.frontier, element)
            else:
                stats.frontier.extend(new_frontier)
        # todo go through the tree and decide if we want to split already visited nodes based on the max and min probability of encountering a terminal state
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

    def update_prism_step(self, frontier, new_frontier, root, stats):
        gateway, mc, mdp, mapping = recreate_prism_PPO(self.graph, root)
        inv_map = {v: k for k, v in mapping.items()}
        with StandardProgressBar(prefix="Updating probabilities ", max_value=len(new_frontier) + 1).start() as bar:
            for t, x in new_frontier:
                try:
                    minmin, minmax, maxmin, maxmax = calculate_target_probabilities(gateway, mc, mdp, mapping, targets=[x])
                    assert maxmax != 0
                    bar.update(bar.value + 1)
                except Exception as e:
                    print("warning")
                    print(e)
                if maxmax > 1e-6:  # check if state is irrelevant
                    frontier.append((t, x))  # next computation only considers relevant states
                else:
                    self.graph.nodes[x]["irrelevant"] = True
                    stats.num_irrelevant += 1

        # bad_nodes = []
        # for x in self.graph.adj:
        #     safe = self.graph.nodes[x].get("safe")
        #     irrelevant = self.graph.nodes[x].get("irrelevant")
        #     if safe is not None:
        #         if safe is False:
        #             bad_nodes.append(x)
        #     if irrelevant is not None:
        #         if irrelevant is True:
        #             bad_nodes.append(x)
        # terminal_states_ids = [mapping[x] for x in bad_nodes]
        # if len(terminal_states_ids) != 0:
        #     # for node in self.graph.nodes:
        #     #     if node in bad_nodes:
        #     #         continue
        #     starting_id = mapping[stats.root]
        #     sum = 0
        #     # for terminal_state_id in terminal_states_ids:
        #     #     minmin, minmax, maxmin, maxmax = extract_probabilities([terminal_state_id], gateway, mc, mdp, root_id=starting_id)
        #     #     sum+=maxmax
        #     minmin, minmax, maxmin, maxmax = extract_probabilities(terminal_states_ids, gateway, mc, mdp, root_id=starting_id)
        #     if maxmin > 0.2:
        #         print(maxmin)

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

    def find_important_dimensions(self, poly1, poly2):
        '''assuming check_contained(poly1,poly2) returns true, we are interested in the halfspaces that matter in terms of splitting of poly1
        poly1 = root, poly2 = candidate
        '''
        # #Binary Space Partitioning
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', self.output_flag)
        input1 = generate_input_region(gurobi_model, self.analysis_template, poly1, self.env_input_size)
        relevant_directions = []
        for j, template in enumerate(self.analysis_template):
            multiplication = 0
            for i in range(self.env_input_size):
                multiplication += template[i] * input1[i]
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

    def check_split(self, t, x, x_label, nn, bar_main, stats, template, template_2d) -> List[Tuple[Any, Any]]:
        # -------splitting
        pre_nn = self.get_pre_nn()
        # self.find_direction_split(template,x,nn,pre_nn)
        ranges_probs = self.sample_probabilities(template, x, nn, pre_nn)  # sampled version
        if not is_split_range(ranges_probs, self.max_probability_split):  # refine only if the range is small
            ranges_probs = self.create_range_bounds_model(template, x, self.env_input_size, nn)
        new_frontier = []
        to_split = []
        n_splits = 0
        to_split.append((x, ranges_probs))
        print("", file=sys.stderr)  # new line
        widgets = [progressbar.Variable('splitting_queue'), ", ", progressbar.Variable('frontier_size'), ", ", progressbar.widgets.Timer()]
        with progressbar.ProgressBar(prefix=f"Splitting states: ", widgets=widgets, is_terminal=True, term_width=200, redirect_stdout=True).start() as bar_split:
            while len(to_split) != 0:
                bar_split.update(value=bar_split.value + 1, splitting_queue=len(to_split), frontier_size=len(new_frontier))
                to_analyse, ranges_probs = to_split.pop()
                action_to_split = np.nanargmax([x[1] - x[0] for x in ranges_probs])
                split_flag = is_split_range(ranges_probs, self.max_probability_split)
                can_be_split = self.can_be_splitted(template, to_analyse)
                if split_flag and can_be_split:
                    split1, split2 = sample_and_split(self.get_pre_nn(), nn, template, np.array(to_analyse), self.env_input_size, template_2d, minimum_length=self.minimum_length,
                                                      action=action_to_split)
                    n_splits += 1
                    if split1 is None or split2 is None:
                        split1, split2 = sample_and_split(self.get_pre_nn(), nn, template, np.array(to_analyse), self.env_input_size, template_2d)
                    ranges_probs1 = self.sample_probabilities(template, split1, nn, pre_nn)  # sampled version
                    if not is_split_range(ranges_probs1, self.max_probability_split):  # refine only if the range is small
                        ranges_probs1 = self.create_range_bounds_model(template, split1, self.env_input_size, nn)
                    ranges_probs2 = self.sample_probabilities(template, split2, nn, pre_nn)  # sampled version
                    if not is_split_range(ranges_probs2, self.max_probability_split):  # refine only if the range is small
                        ranges_probs2 = self.create_range_bounds_model(template, split2, self.env_input_size, nn)
                    to_split.append((tuple(split1), ranges_probs1))
                    to_split.append((tuple(split2), ranges_probs2))

                else:
                    new_frontier.append((to_analyse, ranges_probs))
                    # plot_frontier(new_frontier)

        # colours = []
        # for x, ranges_probs in new_frontier + to_split:
        #     colours.append(np.mean(ranges_probs[0]))
        # print("", file=sys.stderr)  # new line
        # fig = show_polygons(template, [x[0] for x in new_frontier + to_split], template_2d, colours)
        # fig.write_html("new_frontier.html")
        print("", file=sys.stderr)  # new line
        return new_frontier

    def can_be_splitted(self, template, x):
        at_least_one_valid_dimension = False
        dimension_lengths = []
        for i, dimension in enumerate(template):
            inverted_dimension = find_inverted_dimension(-dimension, template)
            if inverted_dimension == -1:
                continue
            dimension_length = x[i] + x[inverted_dimension]
            dimension_lengths.append(dimension_length)
            if dimension_length > self.minimum_length:
                at_least_one_valid_dimension = True
        return at_least_one_valid_dimension

    def sample_probabilities(self, template, x, nn, pre_nn):
        samples = polytope.sample(10000, template, np.array(x))
        preprocessed = pre_nn(torch.tensor(samples).float())
        samples_ontput = torch.softmax(nn(preprocessed), 1)
        predicted_label = samples_ontput.detach().numpy()[:, 0]
        min_prob = np.min(samples_ontput.detach().numpy(), 0)
        max_prob = np.max(samples_ontput.detach().numpy(), 0)
        result = [(min_prob[0], max_prob[0]), (min_prob[1], max_prob[1])]
        return result

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

    @staticmethod
    def generate_input_region(gurobi_model, templates, boundaries, env_input_size):
        input = gurobi_model.addMVar(shape=env_input_size, lb=float("-inf"), ub=float("inf"), name="input")
        generate_region_constraints(gurobi_model, templates, input, boundaries, env_input_size)
        return input

    def get_pre_nn(self):
        """Returns the transformation operation to transform from input to observation"""
        return torch.nn.Identity()

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


def find_contained(x_info: Experiment.SuccessorInfo, seen):
    for i, i_label in seen:
        if contained(x_info.successor, i) and x_info.successor_lbl == i_label:
            return True, (i, i_label)
    return False, -1

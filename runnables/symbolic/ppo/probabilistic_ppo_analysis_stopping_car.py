import os
import sys
from collections import defaultdict
from typing import List, Tuple

import gurobi as grb
import networkx
import numpy as np
import plotly.graph_objects as go
import progressbar
import pypoman
import ray
import torch

from mosaic.utils import PolygonSort, compute_trace_polygons
from polyhedra.experiments_nn_analysis import Experiment, contained
from polyhedra.partitioning import sample_and_split, pick_longest_dimension, split_polyhedron, is_split_range, create_range_bounds_model
from polyhedra.plot_utils import show_polygon_list3, compute_polygon_trace, windowed_projection
from polyhedra.prism_methods import calculate_target_probabilities, recreate_prism_PPO
from runnables.runnable.experiment.run_experiment_stopping_car import StoppingCarExperiment2
from training.ppo.train_PPO_car import get_PPO_trainer
from training.ray_utils import convert_ray_policy_to_sequential
from utility.standard_progressbar import StandardProgressBar


def show_polygons(template, boundaries, template_2d, colours=None):
    fig = go.Figure()
    for i, boundary in enumerate(boundaries):
        vertices = windowed_projection(template, boundary, template_2d)
        # vertices, rays = pypoman.projection.project_polyhedron((template_2d, np.array([0, 0])), (template, np.array(boundaries)), canonicalize=False)
        assert vertices is not None
        sorted_vertices = PolygonSort(vertices)
        trace = compute_trace_polygons([sorted_vertices], colours=[colours[i]])
        fig.add_trace(trace)
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()
    return fig


def get_nn():
    config, trainer = get_PPO_trainer(use_gpu=0)
    trainer.restore("/home/edoardo/ray_results/PPO_StoppingCar_2020-12-30_17-06-3265yz3d63/checkpoint_65/checkpoint-65")
    policy = trainer.get_policy()
    sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
    l0 = torch.nn.Linear(4, 2, bias=False)
    l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, -1], [1, -1, 0, 0]], dtype=torch.float32))
    layers = [l0]
    for l in sequential_nn:
        layers.append(l)
    nn = torch.nn.Sequential(*layers)
    return nn
    # return sequential_nn


def optimise(templates: np.ndarray, gurobi_model: grb.Model, x_prime: tuple, env_input_size: int):
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
    return np.array(results)


def check_unsafe(template, bnds, unsafe_zone, env_input_size):
    for A, b in unsafe_zone:
        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', False)
        input = gurobi_model.addMVar(shape=(env_input_size,), lb=float("-inf"), name="input")
        Experiment.generate_region_constraints(gurobi_model, template, input, bnds, env_input_size)
        Experiment.generate_region_constraints(gurobi_model, A, input, b, env_input_size)
        gurobi_model.update()
        gurobi_model.optimize()
        if gurobi_model.status == 2:
            return True
    return False


def find_contained(x, seen):
    for i in seen:
        if contained(x, i):
            return True, i
    return False, -1


def merge_nodes(to_merge: list, merged_object: tuple, graph):
    """Relabels the nodes so that the to_merge nodes point to the merged_object node"""
    mapping = dict()
    for node in graph.nodes:
        if node in to_merge:
            mapping[node] = merged_object
        else:
            mapping[node] = node
    networkx.relabel_nodes(graph, mapping, False)  # merge inplace, without making a copy of the graph


def evolutionary_merge(graph: networkx.DiGraph, template: np.ndarray):
    nodes = list(graph.nodes)
    n_nodes = graph.number_of_nodes()


def plot_frontier(new_frontier):
    temp_frontier = [x for t, x in new_frontier]
    vertices_list = [pypoman.duality.compute_polytope_vertices(template, np.array(x)) for x in temp_frontier]
    plot_points = [[tuple(x) for x in vertices_item] for vertices_item in vertices_list]
    fig = go.Figure()
    fig.add_trace(compute_polygon_trace(plot_points))
    # fig.update_layout(xaxis_title=x_axis_title, yaxis_title=y_axis_title)
    fig.show()


if __name__ == '__main__':
    ray.init(local_mode=False)
    nn = get_nn()
    save_dir = "/home/edoardo/Development/SafeDRL/"
    load = False
    explore = True
    save = False
    output_flag = False
    show_graph = True
    use_entropy_split = True
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', output_flag)
    env_input_size = 4
    rounding_value = 1024
    horizon = 13
    # input_boundaries = tuple([50, -40, 10, 0, 36, -28, 36, -28])
    input_boundaries = tuple([50, -40, 0, 0, 28, -28, 36, -28])
    distance = [Experiment.e(env_input_size, 0) - Experiment.e(env_input_size, 1)]
    collision_distance = 0
    unsafe_zone: List[Tuple] = [(distance, np.array([collision_distance]))]
    input_template = Experiment.box(env_input_size)
    x_lead = Experiment.e(env_input_size, 0)
    x_ego = Experiment.e(env_input_size, 1)
    v_lead = Experiment.e(env_input_size, 2)
    v_ego = Experiment.e(env_input_size, 3)
    template_2d: np.ndarray = np.array([v_lead - v_ego, x_lead - x_ego])
    # _, template = StoppingCarExperiment.get_template(1)
    # template = np.array([Experiment.e(env_input_size, 0) - Experiment.e(env_input_size, 1), -(Experiment.e(env_input_size, 0) - Experiment.e(env_input_size, 1)),
    #                      Experiment.e(env_input_size, 2) - Experiment.e(env_input_size, 3), -(Experiment.e(env_input_size, 2) - Experiment.e(env_input_size, 3))])
    template = Experiment.combinations([x_lead - x_ego, v_lead - v_ego])
    input = Experiment.generate_input_region(gurobi_model, input_template, input_boundaries, env_input_size)
    root = tuple(optimise(template, gurobi_model, input, env_input_size))
    new_frontier = []
    if load:
        graph = networkx.read_gpickle(os.path.join(save_dir, "graph.g"))
        seen = list(graph.nodes)
        frontier = []
        terminal = []
        for node in graph.nodes:
            attr = graph.nodes[node]
            if not attr.get("safe", True):
                terminal.append(node)
            if not attr.get("irrelevant", False):
                if len(list(graph.neighbors(node))) == 0:
                    t = networkx.shortest_path_length(graph, source=root, target=node)
                    frontier.append((t, node))
        # todo recreate vertices_list?
    else:
        graph = networkx.DiGraph()
        frontier = [(0, root)]
        terminal = []
        seen = [root]
    if explore:
        if show_graph:
            vertices_list = defaultdict(list)
            vertices_list[0].append(root)
        else:
            vertices_list = None

        exit_flag = False
        widgets = [progressbar.Variable('current_t'), ", ", progressbar.Variable('last_action'), ", ", progressbar.Variable('last_polytope'), ", ", progressbar.widgets.Timer()]
        with progressbar.ProgressBar(prefix=f"Main loop: ", widgets=widgets, is_terminal=True, term_width=200, redirect_stdout=True).start() as bar_main:
            while len(new_frontier) != 0 or len(frontier) != 0:
                while len(frontier) != 0:  # performs one exploration iteration
                    t, x = frontier.pop(0)
                    if t == horizon:
                        exit_flag = True
                        break
                    ranges_probs = create_range_bounds_model(template, x, env_input_size, nn)
                    split_flag = is_split_range(ranges_probs)
                    # split_flag = False
                    if split_flag:
                        bar_main.update(value=bar_main.value + 1, current_t=t, last_action="split", last_polytope=str(x))
                        to_split = []
                        to_split.append(x)
                        bar_main.update(force=True)
                        bar_main.fd.flush()
                        print("", file=sys.stderr)  # new line
                        # new_frontier = pickle.load(open("new_frontier.p","rb"))
                        widgets = [progressbar.Variable('splitting_queue'), ", ", progressbar.Variable('frontier_size'), ", ", progressbar.widgets.Timer()]
                        with progressbar.ProgressBar(prefix=f"Splitting states: ", widgets=widgets, is_terminal=True, term_width=200, redirect_stdout=True).start() as bar:
                            while len(to_split) != 0:
                                bar.update(value=bar.value + 1, splitting_queue=len(to_split), frontier_size=len(new_frontier))
                                to_analyse = to_split.pop()
                                if use_entropy_split:
                                    split1, split2 = sample_and_split(nn, template, np.array(to_analyse), env_input_size)
                                else:
                                    dimension = pick_longest_dimension(template, x)
                                    split1, split2 = split_polyhedron(template, x, dimension)
                                ranges_probs1 = create_range_bounds_model(template, split1, env_input_size, nn)
                                split_flag1 = is_split_range(ranges_probs1)
                                if split_flag1:
                                    to_split.append(split1)
                                else:
                                    new_frontier.insert(0, (t, split1))
                                    # plot_frontier(new_frontier)
                                    graph.add_edge(x, split1, action="split")
                                ranges_probs2 = create_range_bounds_model(template, split2, env_input_size, nn)
                                split_flag2 = is_split_range(ranges_probs2)
                                if split_flag2:
                                    to_split.append(split2)
                                else:
                                    new_frontier.insert(0, (t, split2))
                                    # plot_frontier(new_frontier)
                                    graph.add_edge(x, split2, action="split")
                        # print("finished splitting")

                        colours = []
                        for _, x in new_frontier:
                            ranges_probs1 = create_range_bounds_model(template, x, env_input_size, nn)
                            colours.append(np.mean(ranges_probs1[0]))
                        print("", file=sys.stderr)  # new line
                        fig = show_polygons(template, [x[1] for x in new_frontier], template_2d, colours)
                        fig.write_html("new_frontier.html")
                        print("", file=sys.stderr)  # new line
                    else:
                        for chosen_action in range(2):
                            gurobi_model = grb.Model()
                            gurobi_model.setParam('OutputFlag', output_flag)
                            input = Experiment.generate_input_region(gurobi_model, template, x, env_input_size)
                            x_prime = StoppingCarExperiment2.apply_dynamic(input, gurobi_model, action=chosen_action, env_input_size=env_input_size)
                            if ranges_probs[chosen_action][1] <= 1e-6:  # ignore very small probabilities of happening
                                # skip action
                                continue
                            x_prime_results = optimise(template, gurobi_model, x_prime, env_input_size)
                            successor = tuple(x_prime_results)
                            successor = Experiment.round_tuple(successor, rounding_value)
                            # print(successor)
                            is_contained, contained_item = find_contained(successor, seen)
                            if not is_contained:
                                unsafe = check_unsafe(template, x_prime_results, unsafe_zone, env_input_size)
                                if not unsafe:
                                    new_frontier.append((t + 1, successor))
                                    if show_graph:
                                        vertices_list[t].append(successor)
                                else:
                                    bar_main.update(value=bar_main.value + 1, current_t=t, last_action="unsafe", last_polytope=str(successor))
                                    # print("unsafe")
                                    # print(successor)
                                graph.add_edge(x, successor, action=chosen_action, lb=ranges_probs[chosen_action][0], ub=ranges_probs[chosen_action][1])
                                seen.append(successor)
                                if unsafe:
                                    graph.nodes[successor]["safe"] = not unsafe
                                    terminal.append(successor)
                            else:
                                bar_main.update(value=bar_main.value + 1, current_t=t, last_action="skipped", last_polytope=str(successor))
                                # print("skipped")
                                graph.add_edge(x, contained_item, action=chosen_action, lb=ranges_probs[chosen_action][0], ub=ranges_probs[chosen_action][1])
                    if exit_flag:
                        break
                # update prism
                gateway, mc, mdp, mapping = recreate_prism_PPO(graph, root)
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
                            graph.nodes[x]["irrelevant"] = True
                new_frontier = []
                if exit_flag:
                    break
        if show_graph:
            fig, simple_vertices = show_polygon_list3(vertices_list, "delta_v", "delta_x", template, template_2d)
            fig.show()

            width = 2560
            height = 1440
            scale = 1
            fig.write_html(os.path.join(save_dir, "plot.html"))
            fig.write_image(os.path.join(save_dir, "plot.svg"), width=width, height=height, scale=scale)
        if save:
            networkx.write_gpickle(graph, os.path.join(save_dir, "graph.g"))
            print("Graph saved")

    gateway, mc, mdp, mapping = recreate_prism_PPO(graph, root)
    targets = []
    for x, attr in graph.nodes.data():
        if not attr.get("safe", True) or attr.get("irrelevant", False):
            targets.append(x)
    calculate_target_probabilities(gateway, mc, mdp, mapping, targets=targets)

import os
from collections import defaultdict
from typing import List, Tuple

import pypoman
import ray
import torch
import plotly.graph_objects as go
from py4j.java_collections import ListConverter
from py4j.java_gateway import JavaGateway

from agents.ppo.train_PPO_car import get_PPO_trainer
from agents.ray_utils import convert_ray_policy_to_sequential
import gurobi as grb
import networkx

from polyhedra.plot_utils import show_polygon_list3, compute_polygon_trace
from polyhedra.runnable.experiment.run_experiment_stopping_car import StoppingCarExperiment, StoppingCarExperiment2
from polyhedra.runnable.templates.dikin_walk_simplified import sample_polyhedron, find_dimension_split, find_dimension_split2
from polyhedra.runnable.templates.find_polyhedron_split import pick_longest_dimension, split_polyhedron
from symbolic import unroll_methods
from polyhedra.experiments_nn_analysis import Experiment, contained
import numpy as np

from utility.standard_progressbar import StandardProgressBar


def sample_and_split(nn, template, boundaries):
    samples = sample_polyhedron(template, boundaries, 5000)
    samples_ontput = torch.softmax(nn(torch.tensor(samples).float()), 1)
    predicted_label = samples_ontput.detach().numpy()[:, 0]
    chosen_dimension, decision_point = find_dimension_split2(samples, predicted_label, template)
    split1, split2 = split_polyhedron(template, boundaries, chosen_dimension, decision_point)
    return split1, split2


def get_nn():
    config, trainer = get_PPO_trainer(use_gpu=0)
    trainer.restore("/home/edoardo/ray_results/PPO_StoppingCar_2020-12-30_17-06-3265yz3d63/checkpoint_65/checkpoint-65")
    policy = trainer.get_policy()
    sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
    # l0 = torch.nn.Linear(6, 2, bias=False)
    # l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0]], dtype=torch.float32))
    # layers = [l0]
    # for l in sequential_nn:
    #     layers.append(l)
    # nn = torch.nn.Sequential(*layers)
    # return nn
    return sequential_nn


def get_range_bounds(input, nn, gurobi_model):
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


def recreate_prism_PPO(graph, root, max_t: int = None):
    gateway = JavaGateway(auto_field=True)
    mc = gateway.jvm.explicit.IMDPModelChecker(None)
    mdp = gateway.entry_point.reset_mdp()
    eval = gateway.jvm.prism.Evaluator.createForDoubleIntervals()
    mdp.setEvaluator(eval)
    filter_by_action = False
    gateway.entry_point.add_states(graph.number_of_nodes())
    path_length = networkx.shortest_path_length(graph, source=root)
    descendants_dict = defaultdict(bool)
    descendants_true = []
    for descendant in path_length.keys():
        if max_t is None or path_length[descendant] <= max_t:  # limit descendants to depth max_t
            descendants_dict[descendant] = True
            descendants_true.append(descendant)
    mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))  # mapping between each node and an integer
    with StandardProgressBar(prefix="Updating Prism ", max_value=len(descendants_dict) + 1).start() as bar:
        for parent_id, edges in graph.adjacency():  # generate the edges
            if descendants_dict[parent_id]:  # if it is within the filter
                if len(edges.items()) != 0:  # filter out states with no successors (we want to add just edges)
                    if filter_by_action:
                        actions = set([edges[key].get("action") for key in edges])  # get unique actions
                        for action in actions:
                            distribution = gateway.jvm.explicit.Distribution(eval)
                            filtered_edges = [key for key in edges if edges[key].get("action") == action]  # get only the edges matching the current action
                            for successor in filtered_edges:
                                edgeattr = edges[successor]
                                ub = edgeattr.get("ub")
                                lb = edgeattr.get("lb")
                                assert ub is not None
                                assert lb is not None
                                distribution.add(int(mapping[successor]), gateway.jvm.common.Interval(lb, ub))
                            mdp.addActionLabelledChoice(int(mapping[parent_id]), distribution, action)
                    else:  # markov chain following the policy
                        actions = set([edges[key].get("action") for key in edges])  # get unique actions
                        if "split" in actions:
                            # nondeterministic choice
                            for successor in edges:
                                distribution = gateway.jvm.explicit.Distribution(eval)
                                distribution.add(int(mapping[successor]), gateway.jvm.common.Interval(1.0, 1.0))
                                mdp.addActionLabelledChoice(int(mapping[parent_id]), distribution, "split")
                        else:
                            distribution = gateway.jvm.explicit.Distribution(eval)
                            for successor in edges:
                                edgeattr = edges[successor]
                                ub = edgeattr.get("ub")
                                lb = edgeattr.get("lb")
                                assert ub is not None
                                assert lb is not None
                                distribution.add(int(mapping[successor]), gateway.jvm.common.Interval(lb, ub))
                            mdp.addActionLabelledChoice(int(mapping[parent_id]), distribution, "policy")

                else:
                    # zero successors
                    pass
                bar.update(bar.value + 1)  # else:  # print(f"Non descending item found")  # to_remove.append(parent_id)  # pass

    print("prism updated")

    mdp.exportToDotFile("imdppy.dot")
    #
    # terminal_states_java = ListConverter().convert(terminal_states, gateway._gateway_client)

    # half_terminal_states = [mapping[x] for x in self.get_terminal_states_ids(half=True, dict_filter=descendants_dict)]
    # half_terminal_states_java = ListConverter().convert(half_terminal_states, gateway._gateway_client)
    # # get probabilities from prism to encounter a terminal state
    # solution_min = list(gateway.entry_point.check_state_list(terminal_states_java, True))
    # solution_max = list(gateway.entry_point.check_state_list(half_terminal_states_java, False))
    # # update the probabilities in the graph
    # with StandardProgressBar(prefix="Updating probabilities in the graph ", max_value=len(descendants_true)) as bar:
    #     for descendant in descendants_true:
    #         graph.nodes[descendant]['ub'] = solution_max[mapping[descendant]]
    #         graph.nodes[descendant]['lb'] = solution_min[mapping[descendant]]
    #         bar.update(bar.value + 1)
    # print("Prism updated with new data")
    prism_needs_update = False
    return gateway, mc, mdp, mapping


def calculate_target_probabilities(gateway, model_checker, mdp, mapping, targets: List[Tuple]):
    terminal_states = [mapping[x] for x in targets]
    minmin, minmax, maxmin, maxmax = extract_probabilities(terminal_states, gateway, model_checker, mdp)
    # print(f"minmin: {minmin}")
    # print(f"minmax: {minmax}")
    # print(f"maxmin: {maxmin}")
    # print(f"maxmax: {maxmax}")
    return minmin, minmax, maxmin, maxmax


def extract_probabilities(targets: list, gateway, mc, mdp):
    mdp.findDeadlocks(True)
    target = gateway.jvm.java.util.BitSet()
    for id in targets:
        target.set(id)
    res = mc.computeReachProbs(mdp, target, gateway.jvm.explicit.MinMax.min().setMinUnc(True))
    minmin = res.soln[0]
    res = mc.computeReachProbs(mdp, target, gateway.jvm.explicit.MinMax.min().setMinUnc(False))
    minmax = res.soln[0]
    res = mc.computeReachProbs(mdp, target, gateway.jvm.explicit.MinMax.max().setMinUnc(True))
    maxmin = res.soln[0]
    res = mc.computeReachProbs(mdp, target, gateway.jvm.explicit.MinMax.max().setMinUnc(False))
    maxmax = res.soln[0]
    return minmin, minmax, maxmin, maxmax


def evolutionary_merge(graph: networkx.DiGraph, template: np.ndarray):
    nodes = list(graph.nodes)
    n_nodes = graph.number_of_nodes()


def create_range_bounds_model(template, x, env_input_size, nn, round=-1):
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', output_flag)
    input = Experiment.generate_input_region(gurobi_model, template, x, env_input_size)
    # observation = gurobi_model.addMVar(shape=(2,), lb=float("-inf"), ub=float("inf"), name="input")
    # epsilon = 0
    # gurobi_model.addConstr(observation[1] <= input[0] - input[1] + epsilon / 2, name=f"obs_constr21")
    # gurobi_model.addConstr(observation[1] >= input[0] - input[1] - epsilon / 2, name=f"obs_constr22")
    # gurobi_model.addConstr(observation[0] <= input[2] - input[3] + epsilon / 2, name=f"obs_constr11")
    # gurobi_model.addConstr(observation[0] >= input[2] - input[3] - epsilon / 2, name=f"obs_constr12")
    ranges = get_range_bounds(input, nn, gurobi_model)
    ranges_probs = unroll_methods.softmax_interval(ranges)
    if round >= 0:
        pass
        # todo round the probabilities
    return ranges_probs


def acceptable_range(ranges_probs):
    split_flag = False
    for chosen_action in range(2):
        prob_diff = ranges_probs[chosen_action][1] - ranges_probs[chosen_action][0]
        if prob_diff > 0.2:
            # should split the input
            split_flag = True
            break
    return split_flag


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
    show_graph = False
    use_entropy_split = True
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', output_flag)
    env_input_size = 2
    rounding_value = 1024
    horizon = 13
    # input_boundaries = tuple([50, -40, 10, 0, 36, -28, 36, -28])
    input_boundaries = tuple([50, 0, 10, 10])
    template_2d: np.ndarray = np.array([Experiment.e(env_input_size, 0), Experiment.e(env_input_size, 1)])
    distance = [Experiment.e(env_input_size, 0) - Experiment.e(env_input_size, 1)]
    collision_distance = 0
    unsafe_zone: List[Tuple] = [(distance, np.array([collision_distance]))]
    input_template = Experiment.box(env_input_size)
    # _, template = StoppingCarExperiment.get_template(1)
    # template = np.array([Experiment.e(env_input_size, 0) - Experiment.e(env_input_size, 1), -(Experiment.e(env_input_size, 0) - Experiment.e(env_input_size, 1)),
    #                      Experiment.e(env_input_size, 2) - Experiment.e(env_input_size, 3), -(Experiment.e(env_input_size, 2) - Experiment.e(env_input_size, 3))])
    template = input_template
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
        while len(new_frontier) != 0 or len(frontier) != 0:
            while len(frontier) != 0:  # performs one exploration iteration
                t, x = frontier.pop(0)
                if t == horizon:
                    exit_flag = True
                    break
                ranges_probs = create_range_bounds_model(template, x, env_input_size, nn)
                split_flag = acceptable_range(ranges_probs)
                if split_flag:
                    to_split = []
                    to_split.append(x)
                    while len(to_split) != 0:
                        to_analyse = to_split.pop()
                        ranges_probs = create_range_bounds_model(template, to_analyse, env_input_size, nn)
                        split_flag = acceptable_range(ranges_probs)

                        # check prob range not too wide
                        if split_flag:
                            if use_entropy_split:
                                split1, split2 = sample_and_split(nn, template, to_analyse)
                            else:
                                dimension = pick_longest_dimension(template, x)
                                split1, split2 = split_polyhedron(template, x, dimension)
                            to_split.append(split1)
                            to_split.append(split2)
                        else:
                            new_frontier.append((t, to_analyse))
                            # plot_frontier(new_frontier)
                            graph.add_edge(x, to_analyse, action="split")
                    print("finished splitting")
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
                        print(successor)
                        is_contained, contained_item = find_contained(successor, seen)
                        if not is_contained:
                            unsafe = check_unsafe(template, x_prime_results, unsafe_zone, env_input_size)
                            if not unsafe:
                                new_frontier.append((t + 1, successor))
                                if show_graph:
                                    vertices_list[t].append(successor)
                            else:
                                print("unsafe")
                                print(successor)
                            graph.add_edge(x, successor, action=chosen_action, lb=ranges_probs[chosen_action][0], ub=ranges_probs[chosen_action][1])
                            seen.append(successor)
                            if unsafe:
                                graph.nodes[successor]["safe"] = not unsafe
                                terminal.append(successor)
                        else:
                            print("skipped")
                            graph.add_edge(x, contained_item, action=chosen_action, lb=ranges_probs[chosen_action][0], ub=ranges_probs[chosen_action][1])
                if exit_flag:
                    break
            # update prism
            gateway, mc, mdp, mapping = recreate_prism_PPO(graph, root)
            for t, x in new_frontier:
                minmin, minmax, maxmin, maxmax = calculate_target_probabilities(gateway, mc, mdp, mapping, targets=[x])
                if maxmax > 1e-6:  # check if state is irrelevant
                    frontier.append((t, x))  # next computation only considers relevant states
                else:
                    graph.nodes[x]["irrelevant"] = True
            new_frontier = []
            if exit_flag:
                break
        if show_graph:
            fig, simple_vertices = show_polygon_list3(vertices_list, "x_lead", "x_lead-x_ego", template, template_2d)
            fig.show()

            width = 2560
            height = 1440
            scale = 1
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

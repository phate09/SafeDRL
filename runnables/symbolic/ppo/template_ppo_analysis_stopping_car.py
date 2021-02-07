import os
from collections import defaultdict
from typing import List, Tuple

import ray
import torch
from py4j.java_collections import ListConverter
from py4j.java_gateway import JavaGateway, get_field

from agents.ppo.train_PPO_car import get_PPO_trainer
from agents.ray_utils import convert_DQN_ray_policy_to_sequential, convert_ray_policy_to_sequential
import gurobi as grb
import networkx

from polyhedra.plot_utils import show_polygon_list3
from polyhedra.runnable.experiment.run_experiment_stopping_car import StoppingCarExperiment
from symbolic import unroll_methods
from polyhedra.experiments_nn_analysis import Experiment, contained
import numpy as np

from utility.standard_progressbar import StandardProgressBar


def get_nn():
    global config, l0
    config, trainer = get_PPO_trainer(use_gpu=0)
    trainer.restore("/home/edoardo/ray_results/PPO_StoppingCar_2020-12-30_17-06-3265yz3d63/checkpoint_65/checkpoint-65")
    policy = trainer.get_policy()
    sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
    l0 = torch.nn.Linear(6, 2, bias=False)
    l0.weight = torch.nn.Parameter(torch.tensor([[0, 0, 1, -1, 0, 0], [1, -1, 0, 0, 0, 0]], dtype=torch.float32))
    layers = [l0]
    for l in sequential_nn:
        layers.append(l)
    nn = torch.nn.Sequential(*layers)
    return nn


def get_range_bounds(input, nn):
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


def recreate_prism_PPO(graph, root, max_t: int = None):
    gateway = JavaGateway(auto_field=True)
    mc = gateway.jvm.explicit.IMDPModelChecker(None)
    mdp = gateway.entry_point.reset_mdp()
    eval = gateway.jvm.prism.Evaluator.createForDoubleIntervals()
    mdp.setEvaluator(eval)
    mdp.addState()
    mdp.addState()
    mdp.addState()
    dist = gateway.jvm.explicit.Distribution(eval)
    dist.add(1, gateway.jvm.common.Interval(0.2, 0.4))
    dist.add(2, gateway.jvm.common.Interval(0.6, 0.8))
    mdp.addActionLabelledChoice(0, dist, "a")
    dist = gateway.jvm.explicit.Distribution(eval)
    dist.add(1, gateway.jvm.common.Interval(0.1, 0.3))
    dist.add(2, gateway.jvm.common.Interval(0.7, 0.9))
    mdp.addActionLabelledChoice(0, dist, "b")
    mdp.findDeadlocks(True)
    mdp.exportToDotFile("imdppy.dot")
    target = gateway.jvm.java.util.BitSet()
    target.set(2)
    res = mc.computeReachProbs(mdp, target, gateway.jvm.explicit.MinMax.min().setMinUnc(True))
    print(f"minmin: {res.soln[0]}")
    res = mc.computeReachProbs(mdp, target, gateway.jvm.explicit.MinMax.min().setMinUnc(False))
    print(f"minmax: {res.soln[0]}")
    res = mc.computeReachProbs(mdp, target, gateway.jvm.explicit.MinMax.max().setMinUnc(True))
    print(f"maxmin: {res.soln[0]}")
    res = mc.computeReachProbs(mdp, target, gateway.jvm.explicit.MinMax.max().setMinUnc(False))
    print(f"maxmax: {res.soln[0]}")
    gateway.entry_point.add_states(graph.number_of_nodes())
    path_length = networkx.shortest_path_length(graph, source=root)
    descendants_dict = defaultdict(bool)
    descendants_true = []
    for descendant in path_length.keys():
        if max_t is None or path_length[descendant] <= max_t * 2:  # limit descendants to depth max_t
            descendants_dict[descendant] = True
            descendants_true.append(descendant)
    mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))
    with StandardProgressBar(prefix="Updating Prism ", max_value=len(descendants_dict) + 1).start() as bar:
        for parent_id, successors in graph.adjacency():  # generate the edges
            if descendants_dict[parent_id]:
                if len(successors.items()) != 0:  # filter out non-reachable states
                    if parent_id.action is None:  # action choice (probabilistic)
                        distribution = gateway.newDistribution()
                        for successor in successors:
                            eattr = successors[successor]
                            p = (eattr.get("p_ub") + eattr.get("p_lb")) / 2
                            assert p is not None
                            distribution.add(int(mapping[successor]), p)
                        mdp.addActionLabelledChoice(int(mapping[parent_id]), distribution, parent_id.action)
                    else:  # action transition
                        for successor in successors:
                            distribution = gateway.newDistribution()
                            eattr = successors[successor]
                            p = (eattr.get("p_ub") + eattr.get("p_lb")) / 2
                            assert p is not None
                            distribution.add(int(mapping[successor]), p)
                            mdp.addActionLabelledChoice(int(mapping[parent_id]), distribution, parent_id.action)

                else:
                    # zero successors
                    pass
                bar.update(bar.value + 1)  # else:  # print(f"Non descending item found")  # to_remove.append(parent_id)  # pass

    print("done")
    #
    # terminal_states = [mapping[x] for x in self.get_terminal_states_ids(dict_filter=descendants_dict)]
    # half_terminal_states = [mapping[x] for x in self.get_terminal_states_ids(half=True, dict_filter=descendants_dict)]
    # terminal_states_java = ListConverter().convert(terminal_states, gateway._gateway_client)
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
    return mdp, gateway


if __name__ == '__main__':
    ray.init(local_mode=False)
    nn = get_nn()

    output_flag = False
    gurobi_model = grb.Model()
    gurobi_model.setParam('OutputFlag', output_flag)
    env_input_size = 6
    rounding_value = 1024
    input_boundaries = tuple([50, -40, 10, -0, 28, -28, 36, -36, 0, -0, 0, -0])
    template_2d: np.ndarray = np.array([[1, 0, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0]])
    distance = [Experiment.e(6, 0) - Experiment.e(6, 1)]
    collision_distance = 10
    unsafe_zone: List[Tuple] = [(distance, np.array([collision_distance]))]
    input_template = Experiment.box(env_input_size)
    _, template = StoppingCarExperiment.get_template(1)
    input = Experiment.generate_input_region(gurobi_model, input_template, input_boundaries, env_input_size)
    root = tuple(optimise(template, gurobi_model, input, env_input_size))
    vertices_list = defaultdict(list)
    vertices_list[0].append(root)
    frontier = [(0, root)]
    seen = [root]
    graph = networkx.DiGraph()

    while len(frontier) != 0:
        t, x = frontier.pop(0)
        if t == 4:
            break
        ranges_probs = None
        for chosen_action in range(2):
            gurobi_model = grb.Model()
            gurobi_model.setParam('OutputFlag', output_flag)
            input = Experiment.generate_input_region(gurobi_model, template, x, env_input_size)
            x_prime = StoppingCarExperiment.apply_dynamic(input, gurobi_model, action=chosen_action, env_input_size=env_input_size)
            if ranges_probs is None:
                ranges = get_range_bounds(input, nn)
                ranges_probs = unroll_methods.softmax_interval(ranges)
                print(ranges_probs)
            x_prime_results = optimise(template, gurobi_model, x_prime, env_input_size)
            successor = tuple(x_prime_results)
            successor = Experiment.round_tuple(successor, rounding_value)
            is_contained, contained_item = find_contained(successor, seen)
            if not is_contained:
                unsafe = check_unsafe(template, x_prime_results, unsafe_zone, env_input_size)
                if not unsafe:
                    frontier.append((t + 1, successor))
                    vertices_list[t].append(successor)
                else:
                    print("unsafe")
                    print(successor)
                    exit(0)
                graph.add_edge(x, successor, action=chosen_action, lb=ranges_probs[chosen_action][0], ub=ranges_probs[chosen_action][1])
                seen.append(successor)
            else:
                print("skipped")
                graph.add_edge(x, contained_item, action=chosen_action, lb=ranges_probs[chosen_action][0], ub=ranges_probs[chosen_action][1])

    fig, simple_vertices = show_polygon_list3(vertices_list, "x_lead", "x_lead-x_ego", template, template_2d)
    # fig.show()
    save_dir = "/home/edoardo/Development/SafeDRL/"
    width = 2560
    height = 1440
    scale = 1
    fig.write_image(os.path.join(save_dir, "plot.svg"), width=width, height=height, scale=scale)
    mdp, gateway = recreate_prism_PPO(graph, root, max_t=6)

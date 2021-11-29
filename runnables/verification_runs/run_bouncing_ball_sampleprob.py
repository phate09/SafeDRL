import heapq

import networkx
import numpy as np
import progressbar
import ray
import torch
from py4j.java_gateway import JavaGateway
from ray.rllib.agents.ppo import ppo

from environment.bouncing_ball_old import BouncingBall
from polyhedra.experiments_nn_analysis import Experiment
from runnables.runnable.templates import polytope
from runnables.runnable.templates.dikin_walk_simplified import plot_points_and_prediction
from training.ppo.tune.tune_train_PPO_bouncing_ball import get_PPO_config
from training.ray_utils import convert_ray_policy_to_sequential

ray.init()
nn_path = "/home/edoardo/ray_results/tune_PPO_bouncing_ball/PPO_BouncingBall_71684_00004_4_2021-01-18_23-48-21/checkpoint_10/checkpoint-10"
config = get_PPO_config(1234, use_gpu=0)
trainer = ppo.PPOTrainer(config=config)
trainer.restore(nn_path)
policy = trainer.get_policy()
sequential_nn = convert_ray_policy_to_sequential(policy).cpu()
layers = []
for l in sequential_nn:
    layers.append(l)
nn = torch.nn.Sequential(*layers)
horizon = 10

gateway = JavaGateway(auto_field=True)
mc = gateway.jvm.explicit.MDPModelChecker(None)
analysis_template = Experiment.box(2)
boundaries = [6, -5, 1, 1]
samples = polytope.sample(2000, analysis_template, np.array(boundaries, dtype=float))
point_probabilities = []
for i, point in enumerate(samples):
    # generate prism graph

    frontier = [(0, point)]
    root = point
    graph = networkx.DiGraph()
    widgets = [progressbar.Variable('frontier'), ", ", progressbar.Variable('max_t'), ", ", progressbar.widgets.Timer()]
    # with progressbar.ProgressBar(widgets=widgets) as bar_main:
    while len(frontier) != 0:
        t, state = heapq.heappop(frontier)
        # bar_main.update(frontier=len(frontier), max_t=t)
        if t > horizon:
            break
        action_prob = torch.softmax(nn(torch.tensor(state, dtype=torch.float)), 0)
        for action in range(2):
            successor, cost, done, _ = BouncingBall.calculate_successor(state, action)
            graph.add_edge(tuple(state), tuple(successor), p=action_prob[action].item())
            graph.nodes[tuple(successor)]["done"] = done
            if not done:
                heapq.heappush(frontier, (t + 1, tuple(successor)))
            else:
                print("Terminal state found")
    gateway = JavaGateway(auto_field=True, python_proxy_port=25334)
    mdp = gateway.entry_point.reset_mdp()
    gateway.entry_point.add_states(graph.number_of_nodes())
    mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))
    # with StandardProgressBar(prefix="Updating Prism ", max_value=graph.number_of_nodes()).start() as bar:
    for parent_id, successors in graph.adjacency():  # generate the edges
        if len(successors.items()) != 0:  # filter out non-reachable states
            distribution = gateway.newDistribution()
            for successor_id, eattr in successors.items():
                p = eattr.get("p")
                distribution.add(int(mapping[successor_id]), p)
            mdp.addActionLabelledChoice(int(mapping[parent_id]), distribution, 0)
        else:
            # zero successors
            pass
        # bar.update(bar.value + 1)  # else:  # print(f"Non descending item found")  # to_remove.append(parent_id)  # pass
    print(f"Point {i} done")
    terminal = []
    for node, attr in graph.nodes.items():
        if attr.get("done"):
            terminal.append(node)
    terminal_states = [mapping[x] for x in terminal]
    if len(terminal_states) != 0:
        # terminal_states_java = ListConverter().convert(terminal_states, gateway._gateway_client)
        mdpsimple = gateway.entry_point.getMdpSimple()
        mdpsimple.findDeadlocks(True)
        mc = gateway.jvm.explicit.MDPModelChecker(None)
        if (mc.getSettings() is None):
            mc.setSettings(gateway.jvm.prism.PrismSettings())
        target = gateway.jvm.java.util.BitSet()
        for id in terminal_states:
            target.set(id)
        res1 = mc.computeReachProbs(mdpsimple, target, False)
        sol1 = res1.soln
        maxprob = list(sol1)[0]
        res2 = mc.computeReachProbs(mdpsimple, target, True)
        sol2 = res2.soln
        minprob = list(sol2)[0]
    else:
        maxprob = 0
        minprob = 0
    point_probabilities.append((minprob, maxprob))
template_2d: np.ndarray = np.array([[0, 1], [1, 0]])
# show_polygons(analysis_template, samples, template_2d, [x[0] for x in point_probabilities])
plot_points_and_prediction(samples @ template_2d.T, np.array([x[0] for x in point_probabilities]))

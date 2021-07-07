from collections import defaultdict
from typing import List, Tuple

import networkx
from py4j.java_gateway import JavaGateway

from utility.standard_progressbar import StandardProgressBar


def calculate_target_probabilities(gateway, model_checker, mdp, mapping, targets: List[Tuple]):
    terminal_states = [mapping[x] for x in targets]
    minmin, minmax, maxmin, maxmax = extract_probabilities(terminal_states, gateway, model_checker, mdp)
    # print(f"minmin: {minmin}")
    # print(f"minmax: {minmax}")
    # print(f"maxmin: {maxmin}")
    # print(f"maxmax: {maxmax}")
    return minmin, minmax, maxmin, maxmax


def extract_probabilities(targets: list, gateway, mc, mdp, root_id=0):
    mdp.findDeadlocks(True)
    target = gateway.jvm.java.util.BitSet()
    for id in targets:
        target.set(id)
    steps = 10
    res = mc.computeBoundedReachProbs(mdp, target, steps, gateway.jvm.explicit.MinMax.min().setMinUnc(True))
    minmin = res.soln[root_id]
    res = mc.computeBoundedReachProbs(mdp, target, steps, gateway.jvm.explicit.MinMax.min().setMinUnc(False))
    minmax = res.soln[root_id]
    res = mc.computeBoundedReachProbs(mdp, target, steps, gateway.jvm.explicit.MinMax.max().setMinUnc(True))
    maxmin = res.soln[root_id]
    res = mc.computeBoundedReachProbs(mdp, target, steps, gateway.jvm.explicit.MinMax.max().setMinUnc(False))
    maxmax = res.soln[root_id]
    return minmin, minmax, maxmin, maxmax


def recreate_prism_PPO(graph, root, max_t: int = None):
    gateway = JavaGateway(auto_field=True)
    mc = gateway.jvm.explicit.IMDPModelChecker(None)
    mdp = gateway.entry_point.reset_mdp()
    eval = gateway.jvm.prism.Evaluator.createForDoubleIntervals()
    mdp.setEvaluator(eval)
    group_by_action = True
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
                    if group_by_action:  # the action represent the non-deterministic choice and the probability distribution lies beneath it
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


    # mdp.exportToDotFile("imdppy.dot")
    return gateway, mc, mdp, mapping

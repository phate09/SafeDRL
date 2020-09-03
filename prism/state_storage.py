import os
from collections import defaultdict
from typing import Tuple, List

import networkx as nx
from py4j.java_collections import ListConverter
from py4j.java_gateway import JavaGateway

from mosaic.hyperrectangle import HyperRectangle
from utility.standard_progressbar import StandardProgressBar


class StateStorage:
    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.root = None

    def reset(self):
        print("Resetting the StateStorage")
        self.graph = nx.DiGraph()
        self.root = None

    def store_successor_multi(self, items: List[Tuple[HyperRectangle, HyperRectangle]]):
        # first element is parent
        self.graph.add_edges_from(items, p=1.0)

    def store_sticky_successors(self, successor: HyperRectangle, sticky_successor: HyperRectangle, parent: HyperRectangle):
        # we use a="a" to mark the successors belonging to the same distribution (as opposed to the successors of the split operation)
        self.graph.add_edge(parent, successor, p=0.8, a="a")
        self.graph.add_edge(parent, sticky_successor, p=0.2, a="a")  # same action

    def save_state(self, folder_path):
        nx.write_gpickle(self.graph, folder_path)
        print("Mdp Saved")

    def load_state(self, folder_path):
        if os.path.exists(folder_path):
            self.graph = nx.read_gpickle(folder_path)
            print("Mdp Loaded")
            return True
        else:
            print(f"{folder_path} does not exist")
            return False

    def mark_as_half_fail(self, fail_states: List[HyperRectangle]):
        for item in fail_states:
            self.graph.add_node(item)
            self.graph.nodes[item]['half_fail'] = True

    def mark_as_fail(self, fail_states: List[HyperRectangle]):
        for item in fail_states:
            self.graph.add_node(item)
            self.graph.nodes[item]['fail'] = True

    def get_terminal_states_ids(self, half=False, dict_filter=None):
        result = []
        is_half_str = "half " if half else " "
        with StandardProgressBar(prefix=f"Fetching {is_half_str}terminal states ", max_value=self.graph.number_of_nodes()) as bar:
            for node, attr in self.graph.nodes.items():
                if not half:
                    if attr.get("fail"):
                        if dict_filter is None or dict_filter[node]:
                            result.append(node)  # possible_fail_states = self.graph.nodes.data(data='fail', default=False)
                else:
                    if attr.get("half_fail") or attr.get("fail"):
                        if dict_filter is None or dict_filter[node]:
                            result.append(node)
                bar.update(bar.value + 1)
        # return list([x[0] for x in possible_fail_states if x[1]])
        return result

    def remove_unreachable(self):
        descendants = list(nx.algorithms.descendants(self.graph, self.root))  # descendants from 0
        descendants.insert(0, self.root)
        descendants_dict = defaultdict(bool)
        for descendant in descendants:
            descendants_dict[descendant] = True
        to_remove = []
        for parent_id, successors in self.graph.adjacency():  # generate the edges
            if not descendants_dict[parent_id]:
                to_remove.append(parent_id)
        for id in to_remove:
            print(f"removed {id}")
            self.graph.remove_node(id)

    def get_leaves(self, shortest_path, unsafe_threshold, horizon):
        leaves = [(interval, len(shortest_path[interval]) - 1, attributes.get('lb'), attributes.get('ub')) for interval, attributes in self.graph.nodes.data() if
                  self.graph.out_degree(interval) == 0 and not attributes.get('half_fail') and not attributes.get('fail') and interval.action is None and not attributes.get(
                      'ignore') and attributes.get('ub') is not None and interval in shortest_path and len(shortest_path[interval]) - 1 <= horizon]
        return leaves

    def recreate_prism(self, max_t: int = None):
        gateway = JavaGateway()
        # gateway.entry_point.reset_mdp()
        # mdp = self.gateway.entry_point.getMdpSimple()
        mdp = gateway.entry_point.reset_mdp()
        gateway.entry_point.add_states(self.graph.number_of_nodes())
        # fail_states_ids = self.get_terminal_states_ids()
        # java_list = ListConverter().convert(fail_states_ids, gateway._gateway_client)
        # gateway.entry_point.update_fail_label_list(java_list)
        path_length = nx.shortest_path_length(self.graph, source=self.root)
        # descendants = list(path_length.keys())  # descendants from 0
        # descendants.insert(0, self.root)
        descendants_dict = defaultdict(bool)
        descendants_true = []  # = list(descendants_dict.keys())
        for descendant in path_length.keys():
            if max_t is None or path_length[descendant] <= max_t * 2:  # limit descendants to depth max_t
                descendants_dict[descendant] = True
                descendants_true.append(descendant)
        to_remove = []
        mapping = dict(zip(self.graph.nodes(), range(self.graph.number_of_nodes())))
        with StandardProgressBar(prefix="Updating Prism ", max_value=len(descendants_dict) + 1).start() as bar:
            for parent_id, successors in self.graph.adjacency():  # generate the edges
                if descendants_dict[parent_id]:
                    if len(successors.items()) != 0:  # filter out non-reachable states
                        values = set()  # extract action names
                        for successor_id, eattr in successors.items():
                            values.add(eattr.get("a"))
                        group_by_action = [(x, [(successor_id, eattr) for successor_id, eattr in successors.items() if eattr.get("a") == x]) for x in values]  # group successors by action names
                        for action, successors_grouped in group_by_action:

                            if action is None:  # a new action for each successor
                                for successor_id, eattr in successors_grouped:
                                    distribution = gateway.newDistribution()
                                    distribution.add(int(mapping[successor_id]), 1.0)
                                    mdp.addActionLabelledChoice(int(mapping[parent_id]), distribution, action)
                            else:
                                distribution = gateway.newDistribution()
                                for successor_id, eattr in successors_grouped:
                                    p = eattr.get("p")
                                    assert p is not None
                                    distribution.add(int(mapping[successor_id]), p)
                                mdp.addActionLabelledChoice(int(mapping[parent_id]), distribution, action)  # int(mapping[successor_id])
                    else:
                        # zero successors
                        pass
                    bar.update(bar.value + 1)  # else:  # print(f"Non descending item found")  # to_remove.append(parent_id)  # pass
        # for id in to_remove:
        # print(f"removed {id}")
        # self.graph.remove_node(id)
        terminal_states = [mapping[x] for x in self.get_terminal_states_ids(dict_filter=descendants_dict)]
        half_terminal_states = [mapping[x] for x in self.get_terminal_states_ids(half=True, dict_filter=descendants_dict)]
        terminal_states_java = ListConverter().convert(terminal_states, gateway._gateway_client)
        half_terminal_states_java = ListConverter().convert(half_terminal_states, gateway._gateway_client)
        # get probabilities from prism to encounter a terminal state
        solution_min = list(gateway.entry_point.check_state_list(terminal_states_java, True))
        solution_max = list(gateway.entry_point.check_state_list(half_terminal_states_java, False))
        # update the probabilities in the graph
        with StandardProgressBar(prefix="Updating probabilities in the graph ", max_value=len(descendants_true)) as bar:
            for descendant in descendants_true:
                self.graph.nodes[descendant]['ub'] = solution_max[mapping[descendant]]
                self.graph.nodes[descendant]['lb'] = solution_min[mapping[descendant]]
                bar.update(bar.value + 1)
        print("Prism updated with new data")
        self.prism_needs_update = False
        return mdp, gateway

# def get_storage():
#     Pyro5.api.config.SERIALIZER = "marshal"
#     storage = Pyro5.api.Proxy("PYRONAME:prism.statestorage")
#     return storage
#
#
# if __name__ == '__main__':
#     Pyro5.api.config.SERIALIZER = "marshal"
#     Pyro5.api.config.SERVERTYPE = "multiplex"
#     Pyro5.api.Daemon.serveSimple({StateStorage: "prism.statestorage"}, ns=True)

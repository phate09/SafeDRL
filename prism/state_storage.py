# import multiprocessing
import pickle
from collections import defaultdict
import threading
from typing import Tuple, List

import progressbar
import zmq
from py4j.java_gateway import JavaGateway
from bidict import bidict
from py4j.java_collections import ListConverter
from utility.bidict_multi import bidict_multi
import networkx as nx
import Pyro5.api

from utility.standard_progressbar import StandardProgressBar


# @Pyro5.api.expose
# @Pyro5.api.behavior(instance_mode="single")
class StateStorage:
    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.root = None

    def reset(self):
        print("Resetting the StateStorage")
        self.graph = nx.DiGraph()
        self.root = None

    def store_successor_multi(self, items: List[Tuple[Tuple[float, float]]], parent: Tuple[Tuple[float, float]]):
        self.graph.add_edges_from([(parent, x) for x in items], p=1.0)

    def store_sticky_successors(self, successor: Tuple[Tuple[float, float]], sticky_successor: Tuple[Tuple[float, float]], parent: Tuple[Tuple[float, float]]):
        self.graph.add_edge(parent, successor, p=0.8, a=(successor,sticky_successor,parent))
        self.graph.add_edge(parent, sticky_successor, p=0.2, a=(successor,sticky_successor,parent))  # same action

    def save_state(self, folder_path):
        nx.write_gpickle(self.graph, folder_path + "/nx_graph.p")
        print("Mdp Saved")

    def load_state(self, folder_path):
        self.graph = nx.read_gpickle(folder_path + "/nx_graph.p")
        print("Mdp Loaded")

    def mark_as_fail(self, fail_states: List[Tuple[Tuple[float, float]]]):
        for item in fail_states:
            self.graph.add_node(item)
            self.graph.nodes[item]['fail'] = True

    def get_terminal_states_ids(self):
        possible_fail_states = self.graph.nodes.data(data='fail', default=False)
        return list([x[0] for x in possible_fail_states if x[1]])

    def get_terminal_states_dict(self):
        return dict(self.graph.nodes.data(data='fail', default=False))

    def recreate_prism(self):  # todo remake after using  networkx.relabel.convert_node_labels_to_integers
        gateway = JavaGateway()
        gateway.entry_point.reset_mdp()
        # mdp = self.gateway.entry_point.getMdpSimple()
        mdp = gateway.entry_point.reset_mdp()
        gateway.entry_point.add_states(self.graph.number_of_nodes())
        # fail_states_ids = self.get_terminal_states_ids()
        # java_list = ListConverter().convert(fail_states_ids, gateway._gateway_client)
        # gateway.entry_point.update_fail_label_list(java_list)
        descendants = list(nx.algorithms.descendants(self.graph, self.root))  # descendants from 0
        descendants.insert(0, self.root)
        descendants_dict = defaultdict(bool)
        for descendant in descendants:
            descendants_dict[descendant] = True
        to_remove = []
        mapping = dict(zip(self.graph.nodes(), range(self.graph.number_of_nodes())))
        with StandardProgressBar(prefix="Updating Prism ", max_value=len(descendants) + 1) as bar:
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
                                    mdp.addActionLabelledChoice(int(mapping[parent_id]), distribution, int(mapping[successor_id]))
                            else:
                                last_distribution_name = None
                                distribution = gateway.newDistribution()
                                last_successor = None  # this is used just for naming the action
                                for successor_id, eattr in successors_grouped:
                                    p = eattr.get("p", 1.0 / len(successors.items()))
                                    distribution.add(int(mapping[successor_id]), p)
                                    last_successor = successor_id
                                mdp.addActionLabelledChoice(int(mapping[parent_id]), distribution, int(mapping[successor_id]))
                    else:
                        # zero successors
                        pass
                    bar.update(bar.value + 1)
                else:
                    # print(f"Non descending item found")
                    to_remove.append(parent_id)
                    pass
        for id in to_remove:
            self.graph.remove_node(id)
        terminal_states = [mapping[x] for x in self.get_terminal_states_ids()]
        terminal_states_java = ListConverter().convert(terminal_states, gateway._gateway_client)
        # get probabilities from prism to encounter a terminal state
        solution_min = gateway.entry_point.check_state_list(terminal_states_java, True)
        solution_max = gateway.entry_point.check_state_list(terminal_states_java, False)
        # update the probabilities in the graph
        with StandardProgressBar(prefix="Updating probabilities in the graph ", max_value=len(descendants)) as bar:
            for descendant in descendants:
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

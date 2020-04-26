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


@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode="single")
class StateStorage:
    def __init__(self):
        self.dictionary = bidict()
        # self.t_dictionary = defaultdict(list)
        self.last_index = 0

        self.graph: nx.DiGraph = nx.DiGraph()
        self.prism_needs_update = False  # self.terminal_dictionary = defaultdict(bool)

    def reset(self):
        print("Resetting the StateStorage")
        self.dictionary = bidict()
        # self.t_dictionary = defaultdict(list)
        self.last_index = 0
        self.graph = nx.DiGraph()
        self.prism_needs_update = False

    @property
    def needs_update(self):
        return self.prism_needs_update

    def store(self, item: Tuple[Tuple[float, float]]) -> int:
        self.prism_needs_update = True
        if self.dictionary.inverse.get(item) is None:
            id = self.last_index
            self.dictionary[id] = item
            self.last_index = self.last_index + 1  # adds a state to mdpSimple, retrieve index
            return id
        else:
            return self.dictionary.inverse.get(item)

    def store_multi(self, items: List[Tuple[Tuple[float, float]]]) -> List[int]:
        return [self.store(item) for item in items]

    # def assign_t(self, item_id: int, t):
    #     # if item_id not in self.t_dictionary[t]:
    #     #     self.t_dictionary[t].append(item_id)
    #     pass

    # def assign_t_multi(self, items_id: List[int], t):
    #     for item in items_id:
    #         self.assign_t(item, t)

    def store_successor(self, item: Tuple[Tuple[float, float]], parent_id: int) -> int:
        successor_id = self.store(item)
        self.graph.add_edge(parent_id, successor_id, p=1.0, a=f"{parent_id}_{successor_id}")
        self.prism_needs_update = True
        return successor_id

    def store_successor_multi2(self, items: List[Tuple[Tuple[float, float]]], parent_id: int):
        ids = self.store_multi(items)
        self.graph.add_edges_from([(parent_id, x) for x in ids])
        self.prism_needs_update = True
        return ids

    def store_successor_multi(self, items: List[Tuple[Tuple[float, float]]], parent_id: int):
        return [self.store_successor(interval, parent_id) for interval in items]

    def store_sticky_successors(self, successor: Tuple[Tuple[float, float]], sticky_successor: Tuple[Tuple[float, float]], parent_id: int):
        successor_id = self.store(successor)
        sticky_successor_id = self.store(sticky_successor)
        self.graph.add_edge(parent_id, successor_id, p=0.8, a=successor_id)
        self.graph.add_edge(parent_id, sticky_successor_id, p=0.2, a=successor_id)
        self.prism_needs_update = True
        return successor_id, sticky_successor_id

    def save_state(self, folder_path):
        pickle.dump(self.dictionary, open(folder_path + "/dictionary.p", "wb+"))
        pickle.dump(self.last_index, open(folder_path + "/last_index.p", "wb+"))
        nx.write_gpickle(self.graph, folder_path + "/nx_graph.p")
        print("Mdp Saved")

    def load_state(self, folder_path):
        self.dictionary = pickle.load(open(folder_path + "/dictionary.p", "rb"))
        # self.t_dictionary = pickle.load(open(folder_path + "/t_dictionary.p", "rb"))
        self.last_index = pickle.load(open(folder_path + "/last_index.p", "rb"))
        self.graph = nx.read_gpickle(folder_path + "/nx_graph.p")
        # terminal_dictionary = pickle.load(open(folder_path + "/terminal_dictionary.p", "rb"))
        # for terminal in [key for key in terminal_dictionary.keys() if terminal_dictionary[key] is True]:
        #     self.graph.nodes[terminal]['fail'] = True
        self.prism_needs_update = True
        print("Mdp Loaded")

    def mark_as_fail(self, fail_states_ids: List[int]):
        for id in fail_states_ids:
            self.graph.nodes[id]['fail'] = True

    def get_inverse(self, interval):
        interval = tuple([tuple(x) for x in interval])
        return self.dictionary.inverse[interval]

    def get_forward(self, id):
        return self.dictionary[id]

    def get_terminal_states_ids(self):
        possible_fail_states = self.graph.nodes.data(data='fail', default=False)
        return list([x[0] for x in possible_fail_states if x[1]])
    def get_terminal_states_dict(self):
        return dict(self.graph.nodes.data(data='fail', default=False))
    # def get_t_layer(self, t: int) -> List[int]:
    #     return self.t_dictionary[t]

    def dictionary_get(self, id) -> Tuple[Tuple[float, float]]:
        return self.dictionary[id]

    def recreate_prism(self):
        gateway = JavaGateway()
        gateway.entry_point.reset_mdp()
        # mdp = self.gateway.entry_point.getMdpSimple()
        mdp = gateway.entry_point.reset_mdp()
        gateway.entry_point.add_states(self.last_index)
        fail_states_ids = self.get_terminal_states_ids()
        java_list = ListConverter().convert(fail_states_ids, gateway._gateway_client)
        gateway.entry_point.update_fail_label_list(java_list)
        descendants = nx.algorithms.descendants(self.graph, 0)  # descendants from 0
        descendants_dict = defaultdict(bool)

        for descendant in descendants:
            descendants_dict[descendant] = True
        to_remove = []
        with StandardProgressBar(prefix="Updating Prism ", max_value=len(descendants) + 1) as bar:
            for parent_id, successors in self.graph.adjacency():  # generate the edges
                if parent_id == 0 or descendants_dict[parent_id]:
                    if len(successors.items()) != 0:  # filter out non-reachable states
                        values = set()  # extract action names
                        for successor_id, eattr in successors.items():
                            values.add(eattr.get("a"))
                        group_by_action = [(x, [(successor_id, eattr) for successor_id, eattr in successors.items() if eattr.get("a") == x]) for x in values]  # group successors by action names
                        for action, successors_grouped in group_by_action:

                            if action is None:  # a new action for each successor
                                for successor_id, eattr in successors_grouped:
                                    distribution = gateway.newDistribution()
                                    distribution.add(int(successor_id), 1.0)
                                    mdp.addActionLabelledChoice(int(parent_id), distribution, int(successor_id))
                            else:
                                last_distribution_name = None
                                distribution = gateway.newDistribution()
                                last_successor = None  # this is used just for naming the action
                                for successor_id, eattr in successors_grouped:
                                    p = eattr.get("p", 1.0 / len(successors.items()))
                                    distribution.add(int(successor_id), p)
                                    last_successor = successor_id
                                mdp.addActionLabelledChoice(int(parent_id), distribution, int(successor_id))
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
        terminal_states = self.get_terminal_states_ids()
        terminal_states_java = ListConverter().convert(terminal_states, gateway._gateway_client)
        # get probabilities from prism to encounter a terminal state
        solution_min = gateway.entry_point.check_state_list(terminal_states_java, True)
        solution_max = gateway.entry_point.check_state_list(terminal_states_java, False)
        # update the probabilities in the graph
        with StandardProgressBar(prefix="Updating probabilities in the graph ", max_value=len(descendants)) as bar:
            for descendant in descendants:
                self.graph.nodes[descendant]['ub'] = solution_max[descendant]
                self.graph.nodes[descendant]['lb'] = solution_min[descendant]
                bar.update(bar.value + 1)
        print("Prism updated with new data")
        self.prism_needs_update = False
        return mdp, gateway


def get_storage():
    Pyro5.api.config.SERIALIZER = "marshal"
    storage = Pyro5.api.Proxy("PYRONAME:prism.statestorage")
    return storage


if __name__ == '__main__':
    Pyro5.api.config.SERIALIZER = "marshal"
    Pyro5.api.config.SERVERTYPE = "multiplex"
    Pyro5.api.Daemon.serveSimple({StateStorage: "prism.statestorage"}, ns=True)

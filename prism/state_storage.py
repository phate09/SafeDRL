import multiprocessing
import pickle
from collections import defaultdict
import threading
from typing import Tuple, List
import zmq
from py4j.java_gateway import JavaGateway
from bidict import bidict
from py4j.java_collections import ListConverter
from utility.bidict_multi import bidict_multi
import networkx as nx
import Pyro5.api


@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode="single")
class StateStorage():
    def __init__(self):
        self.dictionary = bidict()
        self.t_dictionary = defaultdict(list)
        self.last_index = 0
        self.gateway = JavaGateway()
        self.gateway.entry_point.reset_mdp()
        self.mdp = self.gateway.entry_point.getMdpSimple()
        self.graph = nx.DiGraph()
        self.prism_needs_update = False
        self.lock = multiprocessing.RLock()

    def reset(self):
        with self.lock:
            print("Resetting the StateStorage")
            self.dictionary = bidict()
            self.t_dictionary = defaultdict(list)
            self.last_index = 0
            self.gateway.entry_point.reset_mdp()
            self.mdp = self.gateway.entry_point.getMdpSimple()
            self.graph = nx.DiGraph()
            self.prism_needs_update = False

    @property
    def needs_update(self):
        with self.lock:
            return self.prism_needs_update

    def store(self, item, t) -> int:
        with self.lock:
            # item = tuple([tuple(x) for x in item])
            # print(f"store {item}")
            self.prism_needs_update = True
            if self.dictionary.inverse.get(item) is None:
                # if self.last_index != 0:  # skip the first one as it starts already with a single state
                self.dictionary[self.last_index] = item
                self.t_dictionary[t].append(self.last_index)
                self.last_index = self.last_index + 1  # self.mdp.addState()  # adds a state to mdpSimple, retrieve index
                # if self.last_index == 0:
                #     self.last_index += 1
                return self.last_index
            else:
                return self.dictionary.inverse.get(item)

    def store_successor(self, item: Tuple[Tuple[float, float]], t: int, parent_id: int) -> int:
        with self.lock:
            successor_id = self.store(item, t)
            self.graph.add_edge(parent_id, successor_id, p=1.0, a=successor_id)
            # distribution = self.gateway.newDistribution()
            # distribution.add(successor_id, 1.0)
            # self.mdp.addActionLabelledChoice(parent_id, distribution, successor_id)
            self.prism_needs_update = True
            return successor_id

    def store_sticky_successors(self, successor: Tuple[Tuple[float, float]], sticky_successor: Tuple[Tuple[float, float]], t: int, parent_id: int):
        with self.lock:
            successor_id = self.store(successor, t)
            sticky_successor_id = self.store(sticky_successor, t)
            self.graph.add_edge(parent_id, successor_id, p=0.8, a=successor_id)
            self.graph.add_edge(parent_id, sticky_successor_id, p=0.2, a=successor_id)
            # distribution = self.gateway.newDistribution()
            # distribution.add(successor_id, 0.8)
            # distribution.add(sticky_successor_id, 0.2)
            # self.mdp.addActionLabelledChoice(parent_id, distribution, successor_id)
            self.prism_needs_update = True
            return successor_id, sticky_successor_id

    def save_state(self, folder_path):
        with self.lock:
            pickle.dump(self.dictionary, open(folder_path + "/dictionary.p", "wb+"))
            pickle.dump(self.t_dictionary, open(folder_path + "/t_dictionary.p", "wb+"))
            pickle.dump(self.last_index, open(folder_path + "/last_index.p", "wb+"))
            self.mdp.exportToPrismExplicit(folder_path + "/last_save.prism")
            nx.write_gml(self.graph, folder_path + "/nx_graph.gml")
            print("Mdp Saved")

    def load_state(self, folder_path):
        with self.lock:
            self.dictionary = pickle.load(open(folder_path + "/dictionary.p", "rb"))
            self.t_dictionary = pickle.load(open(folder_path + "/t_dictionary.p", "rb"))
            self.last_index = pickle.load(open(folder_path + "/last_index.p", "rb"))
            self.mdp.buildFromPrismExplicit(folder_path + "/last_save.prism.tra")
            self.graph = nx.read_gml(folder_path + "/nx_graph.gml")
            self.prism_needs_update = True
            print("Mdp Loaded")

    def mark_as_fail(self, fail_states_ids: List[int]):
        if self.prism_needs_update:
            self.recreate_prism()
        java_list = ListConverter().convert(fail_states_ids, self.gateway._gateway_client)
        self.gateway.entry_point.update_fail_label_list(java_list)

    def get_inverse(self, interval):
        with self.lock:
            interval = tuple([tuple(x) for x in interval])
            return self.dictionary.inverse[interval]

    def get_forward(self, id):
        with self.lock:
            return self.dictionary[id]

    # def reversed_t_dictionary(self) -> dict:
    #     print("Reverse")
    #     inv_map = defaultdict(list)  # defaults to empty list
    #     for k, v in self.t_dictionary.items():
    #         inv_map[v].append(k)
    #     return inv_map

    def get_t_layer(self, t: int) -> List[int]:
        with self.lock:
            return self.t_dictionary[t]

    # def purge(self, parent_id: int, target_states_id: List[int]):
    #     java_list = ListConverter().convert(target_states_id, self.gateway._gateway_client)
    #     self.gateway.purge_states(parent_id, java_list)
    #     for id in target_states_id:
    #         self.dictionary.pop(id)
    #         self.t_dictionary.pop(id)
    #         print(f"Purged id {id}")

    def dictionary_get(self, id) -> Tuple[Tuple[float, float]]:
        with self.lock:
            return self.dictionary[id]

    def purge_branch(self, index_to_remove, initial_state=0):
        """Removes the given index and all the unconnected successors from the initial state after it"""
        with self.lock:
            self.graph.remove_node(index_to_remove)
            for component in nx.connected_components(self.graph.to_undirected()):
                if initial_state not in component:
                    self.graph.remove_nodes_from(component)
            self.prism_needs_update = True

    def recreate_prism(self):
        with self.lock:

            self.mdp = self.gateway.entry_point.reset_mdp()
            for i in range(self.last_index):  # generate the states
                index = self.mdp.addState()
            for parent_id, successors in self.graph.adjacency():  # generate the edges
                last_distribution_name = None
                distribution = None
                last_successor = None
                if len(successors.items()) != 0:
                    for successor_id, eattr in successors.items():
                        # print(f"u:{parent_id} v:{successor_id} wt:{eattr}")
                        p = eattr.get("p", 1.0 / len(successors.items()))
                        a = eattr.get("a")
                        if last_distribution_name != a:
                            if distribution is not None:  # save the previous distribution to prism
                                self.mdp.addActionLabelledChoice(int(parent_id), distribution, int(last_successor))
                            distribution = self.gateway.newDistribution()
                            last_distribution_name = a
                        last_successor = successor_id
                        distribution.add(int(successor_id), p)
                    self.mdp.addActionLabelledChoice(int(parent_id), distribution, int(successor_id))
            print("Prism updated with new data")
            self.prism_needs_update = False


def get_storage():
    # c = zerorpc.Client(timeout=99999999, heartbeat=9999999)
    # c.connect("ipc:///tmp/state_storage")
    # # c.connect("tcp://127.0.0.1:4242")
    # return c
    Pyro5.api.config.SERIALIZER = "marshal"
    storage = Pyro5.api.Proxy("PYRONAME:prism.statestorage")
    return storage


if __name__ == '__main__':
    # s = zerorpc.Server(StateStorage())
    # s.bind("ipc:///tmp/state_storage")
    # # s.bind("tcp://0.0.0.0:4242")
    # print("Storage server started")
    # s.run()
    Pyro5.api.config.SERIALIZER = "marshal"
    Pyro5.api.Daemon.serveSimple({StateStorage: "prism.statestorage"}, ns=True)

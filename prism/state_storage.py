import pickle
from typing import Tuple, List
import zmq
from py4j.java_gateway import JavaGateway
from bidict import bidict
import zerorpc
from py4j.java_collections import ListConverter
from utility.bidict_multi import bidict_multi


class StateStorage():
    def __init__(self):
        self.dictionary = bidict()
        self.t_dictionary = bidict_multi()
        self.last_index = 0
        self.gateway = JavaGateway()
        self.gateway.entry_point.reset_mdp()
        self.mdp = self.gateway.entry_point.getMdpSimple()

    def reset(self):
        print("Resetting the StateStorage")
        self.dictionary = bidict()
        self.t_dictionary = bidict_multi()
        self.last_index = 0
        self.gateway.entry_point.reset_mdp()
        self.mdp = self.gateway.entry_point.getMdpSimple()

    def store(self, item, t) -> int:
        item = tuple([tuple(x) for x in item])
        print(f"store {item}")
        if self.dictionary.inverse.get(item) is None:
            # if self.last_index != 0:  # skip the first one as it starts already with a single state
            self.last_index = self.mdp.addState()  # adds a state to mdpSimple, retrieve index
            self.dictionary[self.last_index] = item
            self.t_dictionary[self.last_index] = t
            # if self.last_index == 0:
            #     self.last_index += 1
            return self.last_index
        else:
            return self.dictionary.inverse.get(item)

    def store_successor(self, item: Tuple[Tuple[float, float]], t: int, parent: int) -> int:
        last_index = self.store(item, t)
        self.add_successor(parent, last_index)
        return last_index

    def store_sticky_successors(self, successor: Tuple[Tuple[float, float]], sticky_successor: Tuple[Tuple[float, float]], t: int, parent_id: int):
        successor_id = self.store(successor, t)
        sticky_successor_id = self.store(sticky_successor, t)
        distribution = self.gateway.newDistribution()
        distribution.add(successor_id, 0.8)
        distribution.add(sticky_successor_id, 0.2)
        self.mdp.addActionLabelledChoice(parent_id, distribution, successor_id)
        return successor_id, sticky_successor_id

    def add_successor(self, parent_id, successor_id):
        distribution = self.gateway.newDistribution()
        distribution.add(successor_id, 1.0)
        self.mdp.addActionLabelledChoice(parent_id, distribution, successor_id)

    def save_state(self, folder_path):
        pickle.dump(self.dictionary, open(folder_path + "/dictionary.p", "wb+"))
        pickle.dump(self.t_dictionary, open(folder_path + "/t_dictionary.p", "wb+"))
        pickle.dump(self.last_index, open(folder_path + "/last_index.p", "wb+"))
        self.mdp.exportToPrismExplicit(folder_path + "/last_save.prism")
        print("Mdp Saved")

    def load_state(self, folder_path):
        self.dictionary = pickle.load(open(folder_path + "/dictionary.p", "rb"))
        self.t_dictionary = pickle.load(open(folder_path + "/t_dictionary.p", "rb"))
        self.last_index = pickle.load(open(folder_path + "/last_index.p", "rb"))
        self.mdp.buildFromPrismExplicit(folder_path + "/last_save.prism.tra")
        print("Mdp Loaded")

    def mark_as_fail(self, fail_states_ids: List[int]):
        java_list = ListConverter().convert(fail_states_ids, self.gateway._gateway_client)
        self.gateway.entry_point.update_fail_label_list(java_list)

    def get_inverse(self, interval):
        interval = tuple([tuple(x) for x in interval])
        return self.dictionary.inverse[interval]

    def get_forward(self, id):
        return self.dictionary[id]


def get_storage():
    c = zerorpc.Client(timeout=99999999, heartbeat=9999999)
    c.connect("ipc:///tmp/state_storage")
    # c.connect("tcp://127.0.0.1:4242")
    return c


if __name__ == '__main__':
    s = zerorpc.Server(StateStorage())
    s.bind("ipc:///tmp/state_storage")
    # s.bind("tcp://0.0.0.0:4242")
    print("Storage server started")
    s.run()

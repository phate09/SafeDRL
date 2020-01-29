import pickle
from typing import Tuple

from py4j.java_gateway import JavaGateway
from bidict import bidict


class StateStorage():
    def __init__(self):
        self.dictionary = bidict()
        self.last_index = 0
        self.gateway = JavaGateway()
        self.mdp = self.gateway.entry_point.getMdpSimple()

    def store(self, item):
        if self.dictionary.inverse.get(item) is None:
            # if self.last_index != 0:  # skip the first one as it starts already with a single state
            self.last_index = self.mdp.addState()  # adds a state to mdpSimple, retrieve index
            self.dictionary[self.last_index] = item
            # if self.last_index == 0:
            #     self.last_index += 1
            return self.last_index
        else:
            return self.dictionary.inverse.get(item)

    def store_successor(self, item, parent):
        last_index = self.store(item)
        self.add_successor(parent, last_index)

    def store_sticky_successors(self, successor: Tuple[Tuple[float, float]], sticky_successor: Tuple[Tuple[float, float]], parent_id: int):
        successor_id = self.store(successor)
        sticky_successor_id = self.store(sticky_successor)
        distribution = self.gateway.newDistribution()
        distribution.add(successor_id, 0.8)
        distribution.add(sticky_successor_id, 0.2)
        self.mdp.addActionLabelledChoice(parent_id, distribution, successor_id)
        return successor_id, sticky_successor_id

    def add_successor(self, parent_id, successor_id):
        distribution = self.gateway.newDistribution()
        distribution.add(successor_id, 1.0)
        self.mdp.addActionLabelledChoice(parent_id, distribution, successor_id)

    def save_state(self,folder_path):
        pickle.dump(self.dictionary, open(folder_path+"/dictionary.p", "wb+"))
        pickle.dump(self.last_index, open(folder_path+"/last_index.p", "wb+"))
        self.mdp.exportToPrismExplicit(folder_path+"/last_save.prism")
        print("Mdp Saved")

    def load_state(self):
        self.dictionary = pickle.load(open("/tmp/dictionary.p", "rb"))
        self.last_index = pickle.load(open("/tmp/last_index.p", "rb"))
        self.mdp.buildFromPrismExplicit("/tmp/last_save.prism.tra")
        print("Mdp Loaded")

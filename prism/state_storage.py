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
            self.last_index = self.mdp.addState()  # adds a state to mdpSimple, retrieve index
            self.dictionary[self.last_index] = item
            return self.last_index
        else:
            return self.dictionary.inverse.get(item)

    def store_successor(self, item, parent):
        last_index = self.store(item)
        self.add_successor(parent, last_index)

    def add_successor(self, parent_id, successor_id):
        distribution = self.gateway.newDistribution()
        distribution.add(successor_id, 1.0)
        self.mdp.addActionLabelledChoice(parent_id, distribution, successor_id)

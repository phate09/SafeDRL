import os
from typing import List, Tuple

import Pyro5.api
from rtree import index
from threading import Lock
lock = Lock()

@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode="single")
class SharedRtree:
    def __init__(self):
        with lock:
            os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
            self.p = index.Property(dimension=4)
            self.tree = index.Index('save/rtree', interleaved=False, properties=self.p, overwrite=True)

    def add_single(self, id: int, interval: Tuple[Tuple[Tuple[float, float]], bool]):
        with lock:
            coordinates = flatten_interval(interval[0])
            self.tree.insert(id, coordinates, interval)

    def add_many(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]):
        pass

    def load(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]):
        with lock:
            print("Building the tree")
            helper = bulk_load_rtree_helper(intervals)
            self.tree.close()
            self.tree = index.Index('save/rtree', helper, interleaved=False, properties=self.p, overwrite=True)
            self.tree.flush()
            print("Finished building the tree")

    def filter_relevant_intervals3(self, current_interval: Tuple[Tuple[float, float]]) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
        """Filter the intervals relevant to the current_interval"""
        with lock:
            result = self.tree.intersection(flatten_interval(current_interval), objects='raw')
            return list(result)

    def flush(self):
        with lock:
            self.tree.flush()


def get_rtree() -> SharedRtree:
    Pyro5.api.config.SERIALIZER = "marshal"
    storage = Pyro5.api.Proxy("PYRONAME:prism.rtree")
    return storage


def bulk_load_rtree_helper(data: List[Tuple[Tuple[Tuple[float, float]], bool]]):
    for i, obj in enumerate(data):
        interval = obj[0]
        yield (i, flatten_interval(interval), obj)


def rebuild_tree(union_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]], n_workers: int = 8) -> Tuple[index.Index, List[Tuple[Tuple[Tuple[float, float]], bool]]]:
    p = index.Property(dimension=4)
    # union_states_total = merge_list_tuple(union_states_total, n_workers)  # aggregate intervals
    print("Building the tree")
    helper = bulk_load_rtree_helper(union_states_total)
    rtree = index.Index('save/rtree', helper, interleaved=False, properties=p, overwrite=True)
    rtree.flush()
    print("Finished building the tree")
    return rtree, union_states_total


def flatten_interval(current_interval: Tuple[Tuple[float, float]]) -> Tuple:
    return (
        current_interval[0][0], current_interval[0][1], current_interval[1][0], current_interval[1][1], current_interval[2][0], current_interval[2][1], current_interval[3][0], current_interval[3][1])


if __name__ == '__main__':
    # s = zerorpc.Server(StateStorage())
    # s.bind("ipc:///tmp/state_storage")
    # # s.bind("tcp://0.0.0.0:4242")
    # print("Storage server started")
    # s.run()
    Pyro5.api.config.SERIALIZER = "marshal"
    Pyro5.api.Daemon.serveSimple({SharedRtree: "prism.rtree"}, ns=True)

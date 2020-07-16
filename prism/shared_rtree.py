import pandas as pd
import os
import pickle
from typing import List, Tuple

# import Pyro5.api
import progressbar
from rtree import index

from mosaic import utils
from mosaic.hyperrectangle import HyperRectangle_action, HyperRectangle
from mosaic.utils import round_tuples, flatten_interval


# @Pyro5.api.expose
# @Pyro5.api.behavior(instance_mode="single")
class SharedRtree:
    def __init__(self):
        self.tree: index.Index = None
        self.union_states_total: List[HyperRectangle] = []  # a list representing the content of the tree

    def reset(self, dimension):
        print("Resetting the tree")
        self.dimension = dimension
        self.p = index.Property(dimension=self.dimension)
        self.tree = index.Index(interleaved=False, properties=self.p, overwrite=True)
        self.union_states_total: List[HyperRectangle] = []  # a list representing the content of the tree

    # def add_single(self, interval: HyperRectangle_action, rounding: int):
    #     id = len(self.union_states_total)
    #     # interval = (round_tuple(interval[0], rounding), interval[1])  # rounding
    #     relevant_intervals = self.filter_relevant_intervals3(interval[0], rounding)
    #     if len(relevant_intervals) != 0:
    #         print(len(relevant_intervals))
    #         assert len(relevant_intervals) == 0, f"There is an intersection with the intervals already present in the tree! {relevant_intervals} against {interval}"
    #     self.union_states_total.append(interval)
    #     coordinates = flatten_interval(interval[0])
    #     action = interval[1]
    #     self.tree.insert(id, coordinates, (interval[0], action))

    def tree_intervals(self) -> List[HyperRectangle]:
        return self.union_states_total

    def load(self, intervals: List[HyperRectangle]):

        # with self.lock:
        print("Building the tree")
        helper = bulk_load_rtree_helper(intervals)
        self.tree.close()
        self.union_states_total = intervals
        if len(intervals) != 0:
            self.tree = index.Index(helper, interleaved=False, properties=self.p, overwrite=True)
        else:
            self.tree = index.Index(interleaved=False, properties=self.p, overwrite=True)
        self.tree.flush()
        print("Finished building the tree")

    def load_from_file(self, filename, rounding):
        if os.path.exists(filename):
            # with self.lock:
            print("Loading from file")
            union_states_total = pickle.load(open(filename, "rb"))
            # print("Loaded from file")
            self.union_states_total = union_states_total  # round_tuples(union_states_total, rounding=rounding)
            print("Rounded intervals")
            self.load(self.union_states_total)
        else:
            print(f"{filename} does not exist")

    def save_to_file(self, file_name):
        # with self.lock:
        pickle.dump(self.union_states_total, open(file_name, "wb+"))
        print("Saved RTree")

    def filter_relevant_intervals_multi(self, current_intervals: List[HyperRectangle]) -> List[List[HyperRectangle]]:
        """Filter the intervals relevant to the current_interval"""
        # current_interval = inflate(current_interval, rounding)
        result_return: List[List[HyperRectangle]] = []
        for current_interval in current_intervals:
            results = list(self.tree.intersection(current_interval.to_coordinates(), objects='raw'))
            total: List[HyperRectangle] = []
            for result in results:
                if not result.intersect(current_interval).empty(): #if it intersects
                    total.append(result)
                # suitable = all([x[1] != y[0] and x[0] != y[1] for x, y in zip(result, current_interval)])  #
                # if suitable:
                #     total.append(result)
            result_return.append(total)
        return result_return

    def flush(self):
        # with self.lock:
        self.tree.flush()


# def get_rtree() -> SharedRtree:
#     Pyro5.api.config.SERIALIZER = "marshal"
#     storage = Pyro5.api.Proxy("PYRONAME:prism.rtree")
#     return storage


def bulk_load_rtree_helper(data: List[HyperRectangle]):
    for i, obj in enumerate(data):
        yield (i, obj.to_coordinates(), obj)

# if __name__ == '__main__':
#     Pyro5.api.config.SERIALIZER = "marshal"
#     Pyro5.api.config.SERVERTYPE = "multiplex"
#     Pyro5.api.Daemon.serveSimple({SharedRtree: "prism.rtree"}, ns=True)

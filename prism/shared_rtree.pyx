import os
import pickle
from typing import List
from rtree import index

from mosaic.hyperrectangle import HyperRectangle


class SharedRtree:
    def __init__(self):
        self.tree = None #: index.Index
        self.union_states_total = []  # a list representing the content of the tree : List[HyperRectangle]

    def reset(self, dimension):
        print("Resetting the tree")
        self.dimension = dimension
        self.p = index.Property(dimension=self.dimension)
        self.tree = index.Index(interleaved=False, properties=self.p, overwrite=True)
        self.union_states_total = []  # a list representing the content of the tree : List[HyperRectangle]

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
            print("Loading from file")
            union_states_total = pickle.load(open(filename, "rb"))
            # print("Loaded from file")
            self.union_states_total = union_states_total  # round_tuples(union_states_total, rounding=rounding)
            print("Rounded intervals")
            self.load(self.union_states_total)
        else:
            print(f"{filename} does not exist")

    def save_to_file(self, file_name):
        pickle.dump(self.union_states_total, open(file_name, "wb+"))
        print("Saved RTree")

    def filter_relevant_intervals_multi(self, current_intervals: List[HyperRectangle]) -> List[List[HyperRectangle]]:
        """Filter the intervals relevant to the current_interval"""
        # current_interval = inflate(current_interval, rounding)
        result_return = [] #: List[List[HyperRectangle]]
        for current_interval in current_intervals:
            results = list(self.tree.intersection(current_interval.to_coordinates(), objects='raw'))
            total = [] #: List[HyperRectangle]
            for result in results:
                if not result.intersect(current_interval).empty():  # if it intersects
                    total.append(result)  # suitable = all([x[1] != y[0] and x[0] != y[1] for x, y in zip(result, current_interval)])  #  # if suitable:  #     total.append(result)
            result_return.append(total)
        return result_return

    def flush(self):
        # with self.lock:
        self.tree.flush()


def bulk_load_rtree_helper(data: List[HyperRectangle]):
    for i, obj in enumerate(data):
        yield (i, obj.to_coordinates(), obj)

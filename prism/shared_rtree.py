import pickle
from typing import List, Tuple

import Pyro5.api
import progressbar
from rtree import index

from mosaic.utils import round_tuples, flatten_interval


@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode="single")
class SharedRtree:
    def __init__(self):
        self.p = index.Property(dimension=4)
        self.tree = index.Index(interleaved=False, properties=self.p, overwrite=True)
        self.union_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]] = []  # a list representing the content of the tree

    def reset(self):
        print("Resetting the tree")
        self.p = index.Property(dimension=4)
        self.tree = index.Index(interleaved=False, properties=self.p, overwrite=True)
        self.union_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]] = []  # a list representing the content of the tree

    def add_single(self, interval: Tuple[Tuple[Tuple[float, float]], bool], rounding: int):
        id = len(self.union_states_total)
        # interval = (round_tuple(interval[0], rounding), interval[1])  # rounding
        relevant_intervals = self.filter_relevant_intervals3(interval[0], rounding)
        if len(relevant_intervals) != 0:
            print(len(relevant_intervals))
            assert len(relevant_intervals) == 0, f"There is an intersection with the intervals already present in the tree! {relevant_intervals} against {interval}"
        self.union_states_total.append(interval)
        coordinates = flatten_interval(interval[0])
        action = interval[1]
        self.tree.insert(id, coordinates, (interval[0], action))

    def tree_intervals(self) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
        return self.union_states_total

    def add_many(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int):
        """
        Store all the intervals in the tree with the same action, assumes no overlap between input intervals
        :param intervals: the intervals to add, assumes no overlap
        :param action: the action to be assigned to all the intervals
        :return:
        """
        # with progressbar.ProgressBar(prefix="Add many:", max_value=len(intervals), is_terminal=True, term_width=200) as bar:
        #     for i, interval in enumerate(intervals):
        #         self.add_single(interval, rounding)
        #         bar.update(i)
        self.add_many_rebuild(intervals, rounding)

    def add_many_rebuild(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int):
        with progressbar.ProgressBar(prefix="Add many:", max_value=len(intervals), is_terminal=True, term_width=200) as bar:
            for i, interval in enumerate(intervals):
                relevant_intervals = self.filter_relevant_intervals3(interval[0], rounding)
                if len(relevant_intervals) != 0:
                    print(len(relevant_intervals))
                    assert len(relevant_intervals) == 0, f"There is an intersection with the intervals already present in the tree! {relevant_intervals} against {interval}"
                bar.update(i)
        self.union_states_total.extend(intervals)
        self.load(self.union_states_total)
        for i, interval in enumerate(intervals):
            relevant_intervals = self.filter_relevant_intervals3(interval[0], rounding)
            if len(relevant_intervals) == 0:
                assert len(relevant_intervals) != 0, f"The tree did not recognise the newly added interval"

    def load(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]):
        # with self.lock:
        print("Building the tree")
        helper = bulk_load_rtree_helper(intervals)
        self.tree.close()
        self.tree = index.Index(helper, interleaved=False, properties=self.p)
        self.tree.flush()
        print("Finished building the tree")

    def load_from_file(self, filename, rounding):
        # with self.lock:
        print("Loading from file")
        union_states_total = pickle.load(open(filename, "rb"))
        # print("Loaded from file")
        self.union_states_total = union_states_total  # round_tuples(union_states_total, rounding=rounding)
        print("Rounded intervals")
        self.load(self.union_states_total)

    def save_to_file(self, file_name):
        # with self.lock:
        pickle.dump(self.union_states_total, open(file_name, "wb+"))
        print("Saved RTree")

    def filter_relevant_intervals3(self, current_interval: Tuple[Tuple[float, float]], rounding: int) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
        """Filter the intervals relevant to the current_interval"""
        # current_interval = inflate(current_interval, rounding)
        results = list(self.tree.intersection(flatten_interval(current_interval), objects='raw'))
        total = []
        for result in results:
            suitable = all([x[1] != y[0] and x[0] != y[1] for x, y in zip(result[0], current_interval)])  #
            if suitable:
                total.append(result)
        return sorted(total)

    def flush(self):
        # with self.lock:
        self.tree.flush()

    def get_rtree_temp(self):
        Pyro5.api.config.SERIALIZER = "marshal"
        thing = SharedRtree()
        self._pyroDaemon.register(thing)
        return thing  # just return it, no need to return a proxy


def get_rtree() -> SharedRtree:
    Pyro5.api.config.SERIALIZER = "marshal"
    storage = Pyro5.api.Proxy("PYRONAME:prism.rtree")
    return storage


def bulk_load_rtree_helper(data: List[Tuple[Tuple[Tuple[float, float]], bool]]):
    for i, obj in enumerate(data):
        interval = obj[0]
        yield (i, flatten_interval(interval), obj)


if __name__ == '__main__':
    Pyro5.api.config.SERIALIZER = "marshal"
    Pyro5.api.config.SERVERTYPE = "multiplex"
    Pyro5.api.Daemon.serveSimple({SharedRtree: "prism.rtree"}, ns=True)

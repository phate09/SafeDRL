import pickle
from typing import List, Tuple

import Pyro5.api
import progressbar
from rtree import index

from mosaic.utils import round_tuples, round_tuple, flatten_interval, inflate, open_close_tuple
from mosaic.workers.RemainingWorker import compute_remaining_intervals3


@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode="session")
class SharedRtree_temp:
    def __init__(self):
        self.p = index.Property(dimension=4)
        self.tree = index.Index(interleaved=False, properties=self.p, overwrite=True)
        self.union_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]] = []  # a list representing the content of the tree

    def add_single(self, interval: Tuple[Tuple[Tuple[float, float]], bool], rounding: int):
        id = len(self.union_states_total)
        interval = (open_close_tuple(interval[0]), interval[1])
        relevant_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = self.filter_relevant_intervals3(interval[0], rounding)
        relevant_intervals = [x for x in relevant_intervals if x != interval[0]]  # remove itself
        remaining, intersection_safe, intersection_unsafe = compute_remaining_intervals3(interval[0], relevant_intervals, False)
        for remaining_interval in [(x, interval[1]) for x in remaining]:
            self.union_states_total.append(remaining_interval)
            coordinates = flatten_interval(remaining_interval[0])
            action = remaining_interval[1]
            self.tree.insert(id, coordinates, (remaining_interval[0], action))

    def tree_intervals(self):
        return self.union_states_total

    def add_many(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int):
        """
        Store all the intervals in the tree with the same action
        :param intervals:
        :param action: the action to be assigned to all the intervals
        :return:
        """
        with progressbar.ProgressBar(prefix="Add many:", max_value=len(intervals), is_terminal=True, term_width=200) as bar:
            for i, interval in enumerate(intervals):
                self.add_single(interval, rounding)
                bar.update(i)

    def filter_relevant_intervals3(self, current_interval: Tuple[Tuple[float, float]], rounding: int) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
        """Filter the intervals relevant to the current_interval"""
        # current_interval = inflate(current_interval, rounding)
        results = list(self.tree.intersection(flatten_interval(current_interval), objects='raw'))
        total = []
        for result in results:
            suitable = all([x[1] != y[0] and x[0] != y[1] for x, y in zip(result[0], current_interval)])
            if suitable:
                total.append(result)
        return total


def get_rtree_temp() -> SharedRtree_temp:
    Pyro5.api.config.SERIALIZER = "marshal"
    storage = Pyro5.api.Proxy("PYRONAME:prism.rtreetemp")
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
    rtree = index.Index(helper, interleaved=False, properties=p, overwrite=True)
    rtree.flush()
    print("Finished building the tree")
    return rtree, union_states_total


if __name__ == '__main__':
    Pyro5.api.config.SERIALIZER = "marshal"
    Pyro5.api.config.SERVERTYPE = "multiplex"
    Pyro5.api.Daemon.serveSimple({SharedRtree_temp: "prism.rtreetemp"}, ns=True)

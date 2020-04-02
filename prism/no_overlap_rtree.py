from itertools import cycle
from math import ceil
from typing import List, Tuple

import Pyro5.api
import progressbar
import ray
from rtree import index

from mosaic.utils import flatten_interval, chunks
from mosaic.workers.RemainingWorker import compute_remaining_intervals3
from prism.shared_dictionary import get_shared_dictionary


@Pyro5.api.expose
@Pyro5.api.behavior(instance_mode="session")
class NoOverlapRtree:
    def __init__(self):
        self.p = index.Property(dimension=4)
        self.tree = index.Index(interleaved=False, properties=self.p, overwrite=True)
        self.union_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]] = []  # a list representing the content of the tree

    def load(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]):
        # with self.lock:
        print("Building the tree")
        helper = bulk_load_rtree_helper(intervals)
        self.tree.close()
        self.tree = index.Index(helper, interleaved=False, properties=self.p)
        self.tree.flush()
        print("Finished building the tree")

    def add_single(self, interval: Tuple[Tuple[Tuple[float, float]], bool], rounding: int):
        id = len(self.union_states_total)
        # interval = (open_close_tuple(interval[0]), interval[1])
        relevant_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = self.filter_relevant_intervals3(interval[0], rounding)
        # relevant_intervals = [x for x in relevant_intervals if x != interval[0]]  # remove itself todo needed?
        remaining, intersection_safe, intersection_unsafe = compute_remaining_intervals3(interval[0], relevant_intervals, False)
        for remaining_interval in [(x, interval[1]) for x in remaining]:
            self.union_states_total.append(remaining_interval)
            coordinates = flatten_interval(remaining_interval[0])
            action = remaining_interval[1]
            self.tree.insert(id, coordinates, (remaining_interval[0], action))

    def tree_intervals(self) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
        return self.union_states_total

    def compute_no_overlaps(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int, n_workers, max_iter: int = -1) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
        aggregated_list = intervals
        completed_iterations = 0
        # self_proxy = self
        # self._pyroDaemon.register(self_proxy)
        workers = cycle([NoOverlapWorker.remote(self, rounding) for _ in range(n_workers)])
        with get_shared_dictionary() as shared_dict:
            while True:
                shared_dict.reset()  # reset the dictionary
                self.load(aggregated_list)
                old_size = len(aggregated_list)
                proc_ids = []
                chunk_size = 200
                with progressbar.ProgressBar(prefix="Starting workers", max_value=ceil(old_size / chunk_size), is_terminal=True, term_width=200) as bar:
                    for i, chunk in enumerate(chunks(aggregated_list, chunk_size)):
                        proc_ids.append(next(workers).no_overlap_worker.remote(chunk))
                        bar.update(i)
                aggregated_list = []
                found_list = []
                with progressbar.ProgressBar(prefix="Computing no overlap intervals", max_value=len(proc_ids), is_terminal=True, term_width=200) as bar:
                    while len(proc_ids) != 0:
                        ready_ids, proc_ids = ray.wait(proc_ids)
                        result = ray.get(ready_ids[0])
                        if result is not None:
                            aggregated_list.extend(result[0])
                            found_list.extend(result[1])
                        bar.update(bar.value + 1)
                n_founds = sum(x is True for x in found_list)
                new_size = len(aggregated_list)
                if n_founds == 0:
                    print(f"Reduced overlaps to {n_founds / new_size:.2%}")
                else:
                    print("Finished!")
                if n_founds == 0:
                    break
                completed_iterations += 1
                if completed_iterations >= max_iter != -1:
                    break
        return aggregated_list

    def add_many(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int):
        """
        Store all the intervals in the tree with the same action
        :param intervals:
        :param action: the action to be assigned to all the intervals
        :return:
        """
        with progressbar.ProgressBar(prefix="Add many temp:", max_value=len(intervals), is_terminal=True, term_width=200) as bar:
            for i, interval in enumerate(intervals):
                self.add_single(interval, rounding)
                bar.update(i)

    def filter_relevant_intervals3(self, current_interval: Tuple[Tuple[float, float]], rounding: int) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
        """Filter the intervals relevant to the current_interval"""
        # current_interval = inflate(current_interval, rounding)
        results = list(self.tree.intersection(flatten_interval(current_interval), objects='raw'))
        total = []
        for result in results:  # turn the intersection in an intersection with intervals which are closed only on the left
            suitable = all([x[1] != y[0] and x[0] != y[1] for x, y in zip(result[0], current_interval)])
            if suitable:
                total.append(result)
        return sorted(total)


@ray.remote
class NoOverlapWorker:
    def __init__(self, tree: NoOverlapRtree, rounding):
        self.rounding = rounding
        self.tree: NoOverlapRtree = tree
        self.handled_intervals = get_shared_dictionary()

    def no_overlap_worker(self, intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]) -> Tuple[List[Tuple[Tuple[Tuple[float, float]], bool]], List[bool]]:
        aggregated_list = []
        found_list = [False] * len(intervals)
        for i, interval in enumerate(intervals):
            handled = self.handled_intervals.get(interval, False)
            if not handled:
                relevant_intervals: List[Tuple[Tuple[Tuple[float, float]], bool]] = self.tree.filter_relevant_intervals3(interval[0], self.rounding)
                found_list[i] = len(relevant_intervals) != 0
                relevant_intervals = [x for x in relevant_intervals if not self.handled_intervals.get(x, False)]
                self.handled_intervals.set_multiple([interval] + relevant_intervals, True)  # mark the interval as handled
                remaining, intersection_safe, intersection_unsafe = compute_remaining_intervals3(interval[0], relevant_intervals, False)
                aggregated_list.extend([(x, interval[1]) for x in remaining])
            else:
                # already handled previously
                pass
        return aggregated_list, found_list


def get_rtree_temp() -> NoOverlapRtree:
    Pyro5.api.config.SERIALIZER = "marshal"
    storage = Pyro5.api.Proxy("PYRONAME:prism.rtreetemp")
    return storage


def bulk_load_rtree_helper(data: List[Tuple[Tuple[Tuple[float, float]], bool]]):
    for i, obj in enumerate(data):
        interval = obj[0]
        yield (i, flatten_interval(interval), obj)


#
#
# def rebuild_tree(union_states_total: List[Tuple[Tuple[Tuple[float, float]], bool]], n_workers: int = 8) -> Tuple[index.Index, List[Tuple[Tuple[Tuple[float, float]], bool]]]:
#     p = index.Property(dimension=4)
#     # union_states_total = merge_list_tuple(union_states_total, n_workers)  # aggregate intervals
#     print("Building the tree")
#     helper = bulk_load_rtree_helper(union_states_total)
#     rtree = index.Index(helper, interleaved=False, properties=p, overwrite=True)
#     rtree.flush()
#     print("Finished building the tree")
#     return rtree, union_states_total


if __name__ == '__main__':
    local_mode = False
    if not ray.is_initialized():
        ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
    n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
    Pyro5.api.config.SERIALIZER = "marshal"
    Pyro5.api.config.SERVERTYPE = "multiplex"
    Pyro5.api.Daemon.serveSimple({NoOverlapRtree: "prism.rtreetemp"}, ns=True)

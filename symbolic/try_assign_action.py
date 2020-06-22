import math
import os
import pickle

from rtree import Rtree

import verification_runs.domain_explorers_load
from plnn.bab_explore import DomainExplorer
from prism.shared_rtree import SharedRtree
from symbolic import unroll_methods
from symbolic.unroll_methods import assign_action_to_blank_intervals
import mosaic.utils as utils
import ray

rounding = 3
precision = 10 ** (-rounding)
n_workers = 8
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
n_dimensions = 2


def try1():
    '''Assign action to interval and plot'''

    remainings_merged = [((-0.785, 0.785), (-2.0, 2.0))]
    ray.init(log_to_driver=False)
    explorer, verification_model, env, current_interval, state_size, env_class = verification_runs.domain_explorers_load.generatePendulumDomainExplorer(precision, rounding, sym=True)
    assigned_intervals, ignore_intervals = assign_action_to_blank_intervals(remainings_merged, explorer, verification_model, n_workers, rounding)
    safe = [x for x in assigned_intervals if x[1]]
    unsafe = [x for x in assigned_intervals if not x[1]]
    pickle.dump(safe, open("safe.p", "wb"))
    pickle.dump(unsafe, open("unsafe.p", "wb"))
    utils.show_plot(safe, unsafe, remainings_merged)


def try2():
    safe = pickle.load(open("safe.p", "rb"))
    unsafe = pickle.load(open("unsafe.p", "rb"))
    union = safe + unsafe
    rtree = SharedRtree()
    rtree.reset(n_dimensions)
    rtree.load(union)
    merged_list = []
    window = ((-0.785, 0.785), (-2.0, 2.0))
    subwindows = [window]
    while len(subwindows) != 0:
        print(len(subwindows))
        sub = subwindows.pop()
        filtered = rtree.filter_relevant_intervals_multi([sub])[0]
        if len(filtered) != 0:
            area_filtered = sum([utils.area_tuple(x[0]) for x in filtered])
            area_sub = utils.area_tuple(sub)
            first_action = filtered[0][1]
            all_same = all([x[1] == first_action for x in filtered])
            if all_same and area_filtered<area_sub and not math.isclose(area_filtered, area_sub, abs_tol=1e-10):  # check for empty areas
                all_same = False
            if all_same:
                merged_list.append((sub, first_action))
            else:
                subwindows.extend(DomainExplorer.box_split_tuple(sub, rounding))
    safe = [x for x in merged_list if x[1] == True]
    unsafe = [x for x in merged_list if x[1] == False]
    utils.show_plot(safe, unsafe)

if __name__ == '__main__':
    try2()
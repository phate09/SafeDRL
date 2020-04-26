import os

import gym

from symbolic.unroll_methods import *
from symbolic.unroll_methods import get_rtree_temp
from verification_runs.aggregate_abstract_domain import merge_simple


def try_load():
    gym.logger.set_level(40)
    os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
    local_mode = False
    if not ray.is_initialized():
        ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
    n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
    storage: StateStorage = get_storage()
    storage.reset()
    env = CartPoleEnv_abstract()
    s = env.reset()
    current_interval = s
    rounding = 6
    explorer, verification_model = generateCartpoleDomainExplorer(1e-1, rounding)
    # reshape with tuples
    current_interval = tuple([(float(x.a), float(x.b)) for i, x in enumerate(current_interval)])
    precision = 1e-6
    print(f"Building the tree")
    rtree = get_rtree()
    rtree.reset()
    print(f"Finished building the tree")
    # rtree = get_rtree()
    remainings = [current_interval]
    t = 0

    remainings_new1 = analysis_iteration(remainings, n_workers, rtree, env, explorer, rounding)
    # remainings_new1_2 = analysis_iteration(remainings, t, n_workers, rtree, env, explorer, rounding)
    # assert_lists_equal(remainings_new1, remainings_new1_2)
    remainings_new2 = analysis_iteration(remainings_new1, n_workers, rtree, env, explorer, rounding)
    # remainings_new2_2 = analysis_iteration(remainings_new1, t + 1, n_workers, rtree, env, explorer, rounding)
    # assert_lists_equal(remainings_new2, remainings_new2_2)
    remainings_new3 = analysis_iteration(remainings_new2, n_workers, rtree, env, explorer, rounding)
    rtree.save_to_file("/home/edoardo/Development/SafeDRL/save/union_states_total.p")
    previous_union = [x[0] for x in rtree.tree_intervals()]
    previous_length = len(rtree.tree_intervals())
    print("Saved")
    print("-------------------------AFTER LOADING-------------------------")
    rtree.reset()
    storage.reset()
    rtree.load_from_file("/home/edoardo/Development/SafeDRL/save/union_states_total.p", rounding)
    assert_lists_equal(previous_union, [x[0] for x in rtree.tree_intervals()])
    after_length = len(rtree.tree_intervals())
    assert after_length == previous_length, f"The size of the tree before and after load do not match: {previous_length} vs {after_length}"
    remainings_after_first1 = analysis_iteration(remainings, n_workers, rtree, env, explorer, rounding)
    # remainings_after_first1_2 = analysis_iteration(remainings, t, n_workers, rtree, env, explorer, rounding)
    # assert_lists_equal(remainings_after_first1, remainings_after_first1_2)
    assert_lists_equal(remainings_new1, remainings_after_first1)
    remainings_after_first2 = analysis_iteration(remainings_after_first1, n_workers, rtree, env, explorer, rounding)
    assert_lists_equal(remainings_new2, remainings_after_first2)
    # remainings_after_first2_2 = analysis_iteration(remainings_after_first1, t + 1, n_workers, rtree, env, explorer,
    #                                                rounding)
    # assert_lists_equal(remainings_after_first2, remainings_after_first2_2)
    remainings_after_first3 = analysis_iteration(remainings_after_first2, n_workers, rtree, env, explorer, rounding)
    assert_lists_equal(remainings_new3, remainings_after_first3)


def try_merge():
    gym.logger.set_level(40)
    os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
    local_mode = True
    if not ray.is_initialized():
        ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
    n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
    storage: StateStorage = get_storage()
    storage.reset()
    env = CartPoleEnv_abstract()
    s = env.reset()
    current_interval = s
    rounding = 6
    explorer, verification_model = generateCartpoleDomainExplorer(1e-1, rounding)
    # reshape with tuples
    current_interval = tuple([(float(x.a), float(x.b)) for i, x in enumerate(current_interval)])
    precision = 1e-6
    print(f"Building the tree")
    rtree = get_rtree()
    rtree.reset()
    # rtree.load_from_file("/home/edoardo/Development/SafeDRL/save/union_states_total.p", rounding)
    remainings = [current_interval]
    t = 0
    # remainings = pickle.load(open("/home/edoardo/Development/SafeDRL/save/remainings.p", "rb"))
    remainings_new1 = analysis_iteration(remainings, n_workers, rtree, env, explorer, rounding)
    no_overlaps1 = remove_overlaps([(x, True) for x in remainings_new1], rounding, n_workers)
    no_overlaps1 = [x[0] for x in no_overlaps1]
    remainings_new2 = analysis_iteration(no_overlaps1, n_workers, rtree, env, explorer, rounding)
    no_overlaps2 = remove_overlaps([(x, True) for x in remainings_new2], rounding, n_workers)
    no_overlaps2 = [x[0] for x in no_overlaps2]
    remainings_new3 = analysis_iteration(remainings_new2, n_workers, rtree, env, explorer, rounding)
    no_overlaps3 = remove_overlaps([(x, True) for x in remainings_new3], rounding, n_workers)
    no_overlaps3 = [x[0] for x in no_overlaps3]
    print("-------------------------AFTER MERGING-------------------------")
    union_states_total = rtree.tree_intervals()
    union_states_total_no_overlaps = remove_overlaps(union_states_total, rounding, n_workers)
    total_area_before = sum([area_tuple(remaining[0]) for remaining in union_states_total_no_overlaps])
    union_states_total_merged = merge_simple(union_states_total, rounding)
    total_area_after = sum([area_tuple(remaining[0]) for remaining in union_states_total_merged])
    assert math.isclose(total_area_before, total_area_after), f"The areas do not match: {total_area_before} vs {total_area_after}"
    rtree.load(union_states_total_merged)
    remainings_after_first1 = analysis_iteration(remainings, n_workers, rtree, env, explorer, rounding)
    no_overlaps_after1 = remove_overlaps([(x, True) for x in remainings_after_first1], rounding, n_workers)
    no_overlaps_after1 = [x[0] for x in no_overlaps_after1]
    assert_area_equal(no_overlaps1, no_overlaps_after1)
    remainings_after_first2 = analysis_iteration(no_overlaps_after1, n_workers, rtree, env, explorer, rounding)
    no_overlaps_after2 = remove_overlaps([(x, True) for x in remainings_after_first2], rounding, n_workers)
    no_overlaps_after2 = [x[0] for x in no_overlaps_after2]
    assert_area_equal(no_overlaps2, no_overlaps_after2)

    # remainings_after_first2 = analysis_iteration(remainings_after_first1, t + 1, n_workers, rtree, env, explorer, rounding)
    # assert_area_equal(remainings_new2, remainings_after_first2)
    remainings_after_first3 = analysis_iteration(remainings_after_first2, n_workers, rtree, env, explorer, rounding)
    no_overlaps_after3 = remove_overlaps([(x, True) for x in remainings_after_first3], rounding, n_workers)
    no_overlaps_after3 = [x[0] for x in no_overlaps_after3]
    assert_area_equal(no_overlaps3, no_overlaps_after3)
    # assert_area_equal(remainings_new3, remainings_after_first3)  # total_area_before = sum([area_tuple(remaining) for remaining in remainings_new1])
    # total_area_after = sum([area_tuple(remaining) for remaining in remainings_after_first1])
    # assert math.isclose(total_area_before, total_area_after), f"The areas do not match: {total_area_before} vs {total_area_after}"


def assert_area_equal(list1, list2):
    total_area1 = sum([area_tuple(remaining) for remaining in list1])
    total_area2 = sum([area_tuple(remaining) for remaining in list2])
    if not math.isclose(total_area1, total_area2):
        assert math.isclose(total_area1, total_area2), f"The areas do not match: {total_area1} vs {total_area2}"


def assert_lists_equal(list1, list2):
    # total_area1 = sum([area_tuple(remaining) for remaining in list1])
    # total_area2 = sum([area_tuple(remaining) for remaining in list2])
    # if not math.isclose(total_area1,total_area2):
    #     assert math.isclose(total_area1,total_area2)
    len1 = len(list1)
    len2 = len(list2)
    if len1 != len2:
        assert len1 == len2, f"The two lists do not have the same length: {len1} vs {len2}"
    list1_sorted = sorted(list1)
    list2_sorted = sorted(list2)
    for index, (x, y) in enumerate(zip(list1_sorted, list2_sorted)):
        for i, j in zip(x, y):
            if i != j:
                assert i == j, f"Two elements are not the same at index {index}: {i} vs {j}"


def try_temp_tree():
    temp1 = get_rtree_temp()
    temp2 = get_rtree_temp()
    temp2.add_single((((-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005)), True), 6)
    results1 = temp1.filter_relevant_intervals3(((0, 0.005), (0, 0.005), (0, 0.005), (0, 0.005)), 6)
    results2 = temp2.filter_relevant_intervals3(((0, 0.005), (0, 0.005), (0, 0.005), (0, 0.005)), 6)
    assert results1 != results2


if __name__ == '__main__':
    # try_temp_tree()
    # try_load()
    try_merge()

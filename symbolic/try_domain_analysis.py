import os
import gym
from py4j.java_gateway import JavaGateway
from prism.shared_rtree import get_rtree
from prism.shared_rtree_temp import get_rtree_temp
from symbolic.unroll_methods import *


def try_load():
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
    print(f"Finished building the tree")
    # rtree = get_rtree()
    remainings = [current_interval]
    t = 0

    remainings_new1 = analysis_iteration(remainings, t, n_workers, rtree, env, explorer, rounding)
    # remainings_new1_2 = analysis_iteration(remainings, t, n_workers, rtree, env, explorer, rounding)
    # assert_lists_equal(remainings_new1, remainings_new1_2)
    remainings_new2 = analysis_iteration(remainings_new1, t + 1, n_workers, rtree, env, explorer, rounding)
    # remainings_new2_2 = analysis_iteration(remainings_new1, t + 1, n_workers, rtree, env, explorer, rounding)
    # assert_lists_equal(remainings_new2, remainings_new2_2)
    remainings_new3 = analysis_iteration(remainings_new2, t + 2, n_workers, rtree, env, explorer, rounding)
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
    remainings_after_first1 = analysis_iteration(remainings, t, n_workers, rtree, env, explorer, rounding)
    # remainings_after_first1_2 = analysis_iteration(remainings, t, n_workers, rtree, env, explorer, rounding)
    # assert_lists_equal(remainings_after_first1, remainings_after_first1_2)
    assert_lists_equal(remainings_new1, remainings_after_first1)
    remainings_after_first2 = analysis_iteration(remainings_after_first1, t + 1, n_workers, rtree, env, explorer, rounding)
    assert_lists_equal(remainings_new2, remainings_after_first2)
    # remainings_after_first2_2 = analysis_iteration(remainings_after_first1, t + 1, n_workers, rtree, env, explorer, rounding)
    # assert_lists_equal(remainings_after_first2, remainings_after_first2_2)
    remainings_after_first3 = analysis_iteration(remainings_after_first2, t + 2, n_workers, rtree, env, explorer, rounding)
    assert_lists_equal(remainings_new3, remainings_after_first3)


def assert_lists_equal(list1, list2):
    total_area1 = sum([area_tuple(remaining) for remaining in list1])
    total_area2 = sum([area_tuple(remaining) for remaining in list2])
    if not math.isclose(total_area1,total_area2):
        assert math.isclose(total_area1,total_area2)
    len1 = len(list1)
    len2 = len(list2)
    if len1 !=len2:
        assert len1 == len2, f"The two lists do not have the same length: {len1} vs {len2}"
    list1_sorted = sorted(list1)
    list2_sorted = sorted(list2)
    for x, y in zip(list1_sorted, list2_sorted):
        for i, j in zip(x, y):
            if i != j:
                assert i == j, f"Two elements are not the same: {x} vs {y}"


def try_temp_tree():
    temp1 = get_rtree_temp()
    temp2 = get_rtree_temp()
    temp2.add_single((((-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005)), True), 6)
    results1 = temp1.filter_relevant_intervals3(((0, 0.005), (0, 0.005), (0, 0.005), (0, 0.005)), 6)
    results2 = temp2.filter_relevant_intervals3(((0, 0.005), (0, 0.005), (0, 0.005), (0, 0.005)), 6)
    assert results1 != results2


if __name__ == '__main__':
    # try_temp_tree()
    try_load()

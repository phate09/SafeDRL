import os
import gym
from py4j.java_gateway import JavaGateway
from prism.shared_rtree import get_rtree
from symbolic.unroll_methods import *


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

    remainings_new1 = analysis_iteration(remainings, t, n_workers, rtree, env, explorer, rounding)
    remainings_new2 = analysis_iteration(remainings_new1, t + 1, n_workers, rtree, env, explorer, rounding)
    remainings_new3 = analysis_iteration(remainings_new2, t + 2, n_workers, rtree, env, explorer, rounding)
    rtree.save_to_file("/home/edoardo/Development/SafeDRL/save/union_states_total.p")
    print("Saved")
    print("-------------------------AFTER LOADING-------------------------")
    rtree.reset()
    storage.reset()
    rtree.load_from_file("/home/edoardo/Development/SafeDRL/save/union_states_total.p", rounding)
    remainings_after_first1 = analysis_iteration(remainings, t, n_workers, rtree, env, explorer, rounding)
    remainings_after_first2 = analysis_iteration(remainings_after_first1, t + 1, n_workers, rtree, env, explorer, rounding)
    remainings_after_first3 = analysis_iteration(remainings_after_first2, t + 2, n_workers, rtree, env, explorer, rounding)

    assert remainings_new1 == remainings_after_first1
    assert remainings_new2 == remainings_after_first2
    assert remainings_new3 == remainings_after_first3


if __name__ == '__main__':
    try_load()

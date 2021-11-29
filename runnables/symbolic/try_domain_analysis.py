import os

import gym
import numpy as np
import ray
import torch

import mosaic.hyperrectangle_serialisation as serialisation
import prism.state_storage
import symbolic.unroll_methods as unroll_methods
import utility.domain_explorers_load
from mosaic.hyperrectangle import HyperRectangle_action, HyperRectangle
from prism.shared_rtree import SharedRtree


def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def try_load():
    list_assigned_action, verification_model, rounding = setup()

    print("Obtained coverage")
    count_wrong = 0
    test_performed = 0
    list_wrong_samples = set()
    list_wrong_interval = set()
    for interval_action in list_assigned_action:
        for i in range(10000):
            sample_np = interval_action.sample()
            sample_np = trunc(sample_np, rounding)
            sample_tensor = torch.from_numpy(sample_np)
            action = np.argmax(verification_model.base_network.forward(sample_tensor).data.numpy())
            # action = agent.act(sample_np)
            if not (action == interval_action.action):
                count_wrong += 1
                list_wrong_samples.add(tuple(sample_np))
                list_wrong_interval.add(interval_action)
            test_performed += 1
    print(f"wrong count {count_wrong}/{test_performed} = {count_wrong / test_performed:.2%}")
    print(list_wrong_samples)
    print(list_wrong_interval)
    print("passed tests")


def setup():
    gym.logger.set_level(40)
    os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
    local_mode = False
    allow_compute = True
    allow_save = True
    allow_load = False
    if not ray.is_initialized():
        ray.init(local_mode=local_mode, include_dashboard=True, log_to_driver=False)
    serialisation.register_serialisers()
    n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
    storage = prism.state_storage.StateStorage()
    storage.reset()
    rounding = 2
    precision = 10 ** (-rounding)
    explorer, verification_model, env, current_interval, state_size, env_class = utility.domain_explorers_load.generatePendulumDomainExplorer(precision, rounding, sym=True)
    print(f"Building the tree")
    rtree = SharedRtree()
    rtree.reset(state_size)
    # rtree.load_from_file(f"/home/edoardo/Development/SafeDRL/save/union_states_total_e{rounding}.p", rounding)
    print(f"Finished building the tree")
    current_interval = HyperRectangle.from_tuple(tuple([(0.45, 0.52), (0.02, 0.18)]))
    # current_interval = HyperRectangle.from_tuple(tuple([(0.51, 0.52), (0.17, 0.18)]))
    # current_interval = HyperRectangle.from_tuple(tuple([(0.51, 0.52), (0.16, 0.17)]))
    current_interval = current_interval.round(rounding)
    remainings = [current_interval]
    root = HyperRectangle_action.from_hyperrectangle(current_interval, None)
    storage.root = root
    storage.graph.add_node(storage.root)
    horizon = 4
    t = 0
    # agent = Agent(state_size, 2)
    # agent.load(os.path.expanduser("~/Development") + "/SafeDRL/save/Pendulum_Apr07_12-17-45_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_final.pth")
    intersected_intervals = unroll_methods.check_tree_coverage(True, False, explorer, [current_interval], n_workers, rounding, rtree, verification_model)
    list_assigned_action = []
    for interval_noaction, successors in intersected_intervals:
        list_assigned_action.extend(successors)
    return list_assigned_action, verification_model, rounding


if __name__ == '__main__':
    # try_temp_tree()
    try_load()  # try_merge()

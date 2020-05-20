# %%
import itertools
import os
import time
import numpy as np
import gym
import ray
import mosaic.utils as utils
from prism.shared_rtree import SharedRtree
import prism.state_storage
import symbolic.unroll_methods as unroll_methods
import verification_runs.domain_explorers_load
import networkx as nx
from datetime import datetime

from symbolic.unroll_methods import get_n_states


def experiment(env_name="cartpole", horizon: int = 8, abstract: bool = True, rounding: int = 3, *, folder_path="/home/edoardo/Development/SafeDRL/save", max_iterations=-1, load_only=False):
    gym.logger.set_level(40)
    os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
    local_mode = False
    if not ray.is_initialized():
        ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
    n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
    storage = prism.state_storage.StateStorage()
    storage.reset()
    precision = 10 ** (-rounding)
    if env_name == "cartpole":
        explorer, verification_model, env, current_interval, state_size, env_class = verification_runs.domain_explorers_load.generateCartpoleDomainExplorer(precision, rounding)
    elif env_name == "pendulum":
        explorer, verification_model, env, current_interval, state_size, env_class = verification_runs.domain_explorers_load.generatePendulumDomainExplorer(precision, rounding)
    else:
        raise Exception("Invalid choice")
    print(f"Building the tree")
    rtree = SharedRtree()
    rtree.reset(state_size)
    print(f"Finished building the tree")
    storage.root = (utils.round_tuple(current_interval, rounding), None)
    storage.graph.add_node(storage.root)
    env_type = "concrete" if not abstract else "abstract"
    if abstract:
        rtree.load_from_file(f"{folder_path}/union_states_total_{environment_name}_e{rounding}_{env_type}.p", rounding)
        storage.load_state(f"{folder_path}/nx_graph_{environment_name}_e{rounding}_{env_type}.p")
    else:
        rtree.load_from_file(f"{folder_path}/union_states_total_{environment_name}_e{rounding}_{env_type}.p", rounding)
        loaded = storage.load_state(f"{folder_path}/nx_graph_{environment_name}_e{rounding}_{env_type}.p")
        if not loaded:
            # generate every possible permutation within the boundaries of current_interval
            remainings = []
            grid = np.mgrid[tuple(slice(current_interval[d][0], current_interval[d][1], precision) for d in range(state_size))]

            l = [list(range(x)) for x in grid[0].shape]
            permutations = [tuple(x) for x in itertools.product(*l)]
            for indices in permutations:
                new_index = (slice(None),) + indices
                values = grid[new_index]
                interval = tuple([(float(round(x, rounding)), float(round(x, rounding))) for x in values])
                remainings.append(interval)
            assigned_intervals, ignore_intervals = unroll_methods.assign_action_to_blank_intervals(remainings, explorer, verification_model, n_workers,
                                                                                                   rounding)  # compute the action in each single state
            storage.graph.add_edges_from([(storage.root, x) for x in assigned_intervals], p=1.0)  # assign single intervals as direct successors of root
            next_to_compute = unroll_methods.compute_successors(env_class, assigned_intervals, n_workers, rounding, storage)  # compute successors and store result in graph
            storage.save_state(f"{folder_path}/nx_graph_{environment_name}_e{rounding}_{env_type}.p")
    if not load_only:
        # %%
        print(f"Start time: {datetime.now():%d/%m/%Y %H:%M:%S}")
        iterations = 0
        time_from_last_save = time.time()
        while True:
            print(f"Iteration {iterations}")
            split_performed = unroll_methods.probability_iteration(storage, rtree, precision, rounding, env_class, n_workers, explorer, verification_model, state_size, horizon=horizon,
                                                                   allow_assign_actions=True, allow_merge=abstract,allow_refine = False)
            if time.time() - time_from_last_save >= 60 * 2:
                storage.save_state(f"{folder_path}/nx_graph_{environment_name}_e{rounding}_{env_type}.p")
                rtree.save_to_file(f"{folder_path}/union_states_total_{environment_name}_e{rounding}_{env_type}.p")
                print("Graph Saved - Checkpoint")
                time_from_last_save = time.time()
            if not split_performed or (0 <= max_iterations == iterations):
                if not split_performed:
                    print("No more split performed")
                break
            iterations += 1
        # %%
        storage.save_state(f"{folder_path}/nx_graph_{environment_name}_e{rounding}_{env_type}.p")
        rtree.save_to_file(f"{folder_path}/union_states_total_{environment_name}_e{rounding}_{env_type}.p")
        print(f"End time: {datetime.now():%d/%m/%Y %H:%M:%S}")
    return storage, rtree


if __name__ == '__main__':
    folder_path = "/home/edoardo/Development/SafeDRL/save"
    horizon = 7
    precision = 2
    environment_name = "pendulum"
    storage_abstract, tree_abstract = experiment(environment_name, horizon, True, precision, load_only=True, folder_path=folder_path)
    n_states_abstract = get_n_states(storage_abstract, horizon)
    print(f"Number of states at different timesteps:")
    print(n_states_abstract)
    # storage_concrete, tree_concrete = experiment(environment_name, horizon, False, precision, load_only=False, folder_path=folder_path)
    # shortest_path_concrete = nx.shortest_path(storage_concrete.graph, source=storage_concrete.root)
    # n_states_concrete = get_n_states(storage_concrete, horizon)
    import matplotlib.pyplot as plt

    # line 1 points
    time_steps = list(range(1, horizon+1))
    plt.plot(time_steps, n_states_abstract, label="abstract")
    # plt.plot(time_steps, n_states_concrete, label="concrete")
    plt.xlabel('timesteps')
    plt.ylabel('# states')
    plt.title(f'Comparison of # of states in {environment_name} environment')
    # show a legend on the plot
    plt.legend()
    plt.xticks(np.arange(1, horizon+1, step=1))  # Set x label locations.
    # Display a figure.
    plt.savefig(f"{folder_path}/plot_states_{environment_name}_e{precision}_h{horizon}_{time.time()}.png")
    plt.show()

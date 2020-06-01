# %%
import math
import os
import pickle

import gym

from prism.shared_dictionary import get_shared_dictionary
from prism.shared_rtree import get_rtree
from verification_runs.aggregate_abstract_domain import merge_list_tuple

gym.logger.set_level(40)
from py4j.java_gateway import JavaGateway

from symbolic.unroll_methods import *
"""
Calculates the actions to assign to an initial interval and save the result to a file
"""
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
gateway = JavaGateway()
storage: StateStorage = get_storage()
storage.reset()
dictionary = get_shared_dictionary()
dictionary.reset()
tree = get_rtree()
env = CartPoleEnv_abstract()
s = env.reset()
explorer, verification_model = generateCartpoleDomainExplorer(precision=1e-1)
theta_threshold_radians = 12 * 2 * math.pi / 360  # maximum angle allowed
x_threshold = 2.4  # maximum distance allowed
current_interval: Tuple[Tuple[float, float]] = tuple(
    [(-x_threshold, x_threshold), (-3 * x_threshold, 3 * x_threshold), (-theta_threshold_radians, theta_threshold_radians), (-3 * theta_threshold_radians, 3 * theta_threshold_radians)])
remainings: List[Tuple[Tuple[float, float]]] = [current_interval]
local_mode = False
if not ray.is_initialized():
    ray.init(local_mode=local_mode, include_webui=True, log_to_driver=False)
n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
union_states_total = []
# %% Calculate the action
safe_states_current, unsafe_states_current, ignored = assign_action_to_blank_intervals(remainings, n_workers)
union_states_total.extend([(x, True) for x in safe_states_current])
union_states_total.extend([(x, False) for x in unsafe_states_current])
# %% Merge the result
union_states_merged = merge_list_tuple(union_states_total, n_workers)
# %% Save the result
pickle.dump(union_states_merged, open("/home/edoardo/Development/SafeDRL/save/union_states_total.p", "wb+"))

import os
import time
import gym
import ray
from sklearn.linear_model import LogisticRegression

import mosaic.hyperrectangle_serialisation as serialisation
import mosaic.utils as utils
import prism.state_storage
import symbolic.unroll_methods as unroll_methods
import utility.domain_explorers_load
from mosaic.hyperrectangle import HyperRectangle_action, HyperRectangle
from prism.shared_rtree import SharedRtree
import numpy as np
import torch

gym.logger.set_level(40)
os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
local_mode = False
if not ray.is_initialized():
    ray.init(local_mode=local_mode, include_dashboard=True, log_to_driver=False)
serialisation.register_serialisers()
n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
rounding = 2
precision = 10 ** (-rounding)
explorer, verification_model, env, current_interval, state_size, env_class = utility.domain_explorers_load.generatePendulumDomainExplorer(precision, rounding, sym=True)
storage = prism.state_storage.StateStorage()
storage.reset()
rtree = SharedRtree()
rtree.reset(state_size)
current_interval: HyperRectangle = HyperRectangle.from_tuple(((0.35, 0.79), (-1, 1)))

# %% show (approximated) true decision boundary
unroll_methods.check_tree_coverage(True, True, explorer, [current_interval], 8, rounding, rtree, verification_model)
intervals = rtree.tree_intervals()
utils.show_plot([x.to_tuple() for x in intervals if x.action], [x.to_tuple() for x in intervals if not x.action])
# %% sample 10k points from the area and feed to the nn
samples = np.stack([current_interval.sample() for i in range(10000)])
actions = torch.argmax(verification_model.base_network(torch.from_numpy(samples)), dim=1)
utils.scatter_plot([tuple(x) for i,x in enumerate(samples) if bool(actions[i].item())],[tuple(x) for i,x in enumerate(samples) if not bool(actions[i].item())])
#%%
clf1 = LogisticRegression(random_state=0).fit(samples, actions)
coeff = clf1.coef_
intercept = clf1.intercept_
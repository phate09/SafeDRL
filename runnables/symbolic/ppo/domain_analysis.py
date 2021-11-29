# %%
import os

import numpy as np
import ray
import torch
from sklearn.model_selection import ParameterGrid

import mosaic.utils as utils
import utility.domain_explorers_load
from mosaic.hyperrectangle import HyperRectangle
from symbolic.symbolic_interval import Symbolic_interval, Interval_network
from training.ray_utils import load_sequential_from_ray, get_pendulum_ppo_agent

#
# gym.logger.set_level(40)
# os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
local_mode = False
# allow_compute = True
# allow_save = False
# allow_load = False
if not ray.is_initialized():
    ray.init(local_mode=local_mode, include_dashboard=True, log_to_driver=False)
# serialisation.register_serialisers()
# n_workers = int(ray.cluster_resources()["CPU"]) if not local_mode else 1
# storage = prism.state_storage.StateStorage()
# storage.reset()
rounding = 2
precision = 10 ** (-rounding)
explorer, verification_model, env, current_interval, state_size, env_class = utility.domain_explorers_load.generatePendulumDomainExplorerPPO(precision, rounding, sym=True)
delta_x = precision * 2
delta_y = precision * 2
param_grid = {'param1': list(np.arange(-0.79, 0.79, delta_x).round(rounding)), 'param2': list(np.arange(-1, 1, delta_y).round(rounding))}
grid = ParameterGrid(param_grid)
sequential_nn = load_sequential_from_ray(os.path.expanduser("~/Development") + "/SafeDRL/save/PPO_PendulumEnv_2020-09-18_11-23-17wpwqe3zd/checkpoint_25/checkpoint-25",get_pendulum_ppo_agent())
# net = TestNetwork2()
sequential_nn.add_module("softmax", torch.nn.Softmax())
sequential_nn.cpu()
# verif = SymVerificationNetwork(net.sequential)
current_intervals = []
for params in grid:
    state = np.array((params['param1'], params['param2']))
    current_intervals.append(torch.tensor(state))
tensor_intervals = torch.stack(current_intervals)
# ix = Symbolic_interval(lower=torch.tensor([[0, 0]], dtype=torch.float64, requires_grad=False), upper=torch.tensor([[0.01, 0.01]], dtype=torch.float64, requires_grad=False))
ix2 = Symbolic_interval(lower=tensor_intervals, upper=tensor_intervals + np.array([delta_x, delta_y]))
# verif.get_boundaries(ix, 0)
inet = Interval_network(sequential_nn.double(), None)
result_interval = inet(ix2)
upper_bound = result_interval.u
lower_bound = result_interval.l
upper_bound_rectangles = []
for x, y in zip(tensor_intervals, upper_bound[:, 0]):
    x_numpy = x.numpy()
    x_numpy = np.stack([x_numpy, x_numpy + np.array([delta_x, delta_y])])
    from_numpy = HyperRectangle.from_numpy(x_numpy)
    upper_bound_rectangles.append((from_numpy, y.item()))
lower_bound_rectangles = []
for x, y in zip(tensor_intervals, lower_bound[:, 0]):
    x_numpy = x.numpy()
    x_numpy = np.stack([x_numpy, x_numpy + np.array([delta_x, delta_y])])
    from_numpy = HyperRectangle.from_numpy(x_numpy)
    lower_bound_rectangles.append((from_numpy, y.item()))
utils.show_heatmap(upper_bound_rectangles, title="Upper bound probability of action=0", rounding=3)
utils.show_heatmap(lower_bound_rectangles, title="Lower bound probability of action=0", rounding=3)
print("Finished")

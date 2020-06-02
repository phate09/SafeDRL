import os
import torch
import numpy as np
from dqn.dqn_agent import Agent
from environment.cartpole_abstract import CartPoleEnv_abstract
from symbolic.symbolic_interval.symbolic_network import Interval_Bound

state_size = 4
agent = Agent(state_size, 2)
agent.load(os.path.expanduser("~/Development") + "/SafeDRL/save/Sep19_12-42-51_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_5223.pth")
net = agent.qnetwork_local.sequential.cpu()
# true_class_index = 1
# n_classes = net[-1].out_features
# cases = []
# for i in range(n_classes):
#     if i == true_class_index:
#         continue
#     case = [0] * n_classes  # list of zeroes
#     case[true_class_index] = 1  # sets the property to 1
#     case[i] = -1
#     cases.append(case)
# weights = np.array(cases)
# #         print(f'weight={weights}')
# weight_tensor = torch.from_numpy(weights).float()
# property_layer = torch.nn.Linear(2, 1, bias=False)
# with torch.no_grad():
#     property_layer.weight = torch.nn.Parameter(weight_tensor)
# layers = []
# layers.extend(net)
# layers.append(property_layer)
# layers = torch.nn.Sequential(*layers)
epsilon = 0.1
use_cuda = False
norm = "linf"
env = CartPoleEnv_abstract()
X = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
y = torch.tensor([1])
interval_bound = Interval_Bound(net, epsilon, method="sym", use_cuda=use_cuda, norm=norm)
wc = interval_bound.forward(X, y)  # worst case
print(wc)

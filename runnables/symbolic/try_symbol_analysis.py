import os

import numpy as np
import torch
from dqn.dqn_agent import Agent
from models.model_critic_sequential import TestNetwork

from plnn.verification_network import VerificationNetwork
from plnn.verification_network_sym import SymVerificationNetwork
from symbolic.symbolic_interval import Symbolic_interval
from symbolic.symbolic_interval.symbolic_network import Interval_network


def try1():
    """Reproduces example from reluval paper"""
    net = TestNetwork().sequential.cuda()
    # X = torch.tensor([[5, 3]], dtype=torch.float64)
    # result = net(X)
    # epsilon = torch.tensor([1, 2], dtype=torch.float64)
    use_cuda = True
    norm = "linf"
    # X = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    # y = torch.tensor([[0, 20]], dtype=torch.float64)
    # interval_bound = Interval_Bound(net, epsilon, method="sym", use_cuda=use_cuda, norm=norm, worst_case=False)

    inet = Interval_network(net, None)
    ix = Symbolic_interval(lower=torch.tensor([[4, 1]], dtype=torch.float64).cuda(), upper=torch.tensor([[6, 5]], dtype=torch.float64).cuda(), use_cuda=use_cuda)
    # ix2 = Symbolic_interval(lower=torch.tensor([[4, 3]], dtype=torch.float64).cuda(), upper=torch.tensor([[6, 5]], dtype=torch.float64).cuda(), use_cuda=use_cuda)
    ic = inet(ix)
    # ic2 = inet(ix2)
    print(ic)


def try2():
    """Use symintervals with an agent's networks plus action layer"""
    use_cuda = False
    state_size = 4
    agent = Agent(state_size, 2)
    agent.load(os.path.expanduser("~/Development") + "/SafeDRL/save/Sep19_12-42-51_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_5223.pth")
    net = agent.qnetwork_local.sequential.cpu().double()
    true_class_index = 1
    n_classes = net[-1].out_features
    cases = []
    for i in range(n_classes):
        if i == true_class_index:
            continue
        case = [0] * n_classes  # list of zeroes
        case[true_class_index] = 1  # sets the property to 1
        case[i] = -1
        cases.append(case)
    weights = np.array(cases)
    #         print(f'weight={weights}')
    weight_tensor = torch.from_numpy(weights).double()
    property_layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        property_layer.weight = torch.nn.Parameter(weight_tensor)
    layers = []
    layers.extend(net)
    layers.append(property_layer)
    layers = torch.nn.Sequential(*layers)
    inet = Interval_network(layers, None)
    ix = Symbolic_interval(lower=torch.tensor([[-0.05, -0.05, -0.05, -0.05]], dtype=torch.float64), upper=torch.tensor([[0.05, 0.05, 0.05, 0.05]], dtype=torch.float64), use_cuda=use_cuda)
    ic = inet(ix)
    print(ic)


def try3():
    use_cuda = False
    state_size = 4
    agent = Agent(state_size, 2)
    agent.load(os.path.expanduser("~/Development") + "/SafeDRL/save/Sep19_12-42-51_alpha=0.6, min_eps=0.01, eps_decay=0.2/checkpoint_5223.pth")
    symnet = SymVerificationNetwork(agent.qnetwork_local.sequential.cpu().double())
    ix = Symbolic_interval(lower=torch.tensor([[-0.05, -0.05, -0.05, -0.05], [-0.01, -0.01, -0.01, -0.01]], dtype=torch.float64),
                           upper=torch.tensor([[0.05, 0.05, 0.05, 0.05], [0.01, 0.01, 0.01, 0.01]], dtype=torch.float64), use_cuda=use_cuda)
    u, l = symnet.get_boundaries(ix, 1)
    print(f"u:{u}")
    print(f"l:{l}")
    net = VerificationNetwork(agent.qnetwork_local.cpu().double())
    dom_ub, dom_lb = net.get_boundaries(torch.tensor([[-0.05, -0.05, -0.05, -0.05], [0.05, 0.05, 0.05, 0.05]], dtype=torch.float64).t(), 1, False)
    dom_ub, dom_lb = net.get_boundaries(torch.tensor([[-0.01, -0.01, -0.01, -0.01], [0.01, 0.01, 0.01, 0.01]], dtype=torch.float64).t(), 1, False)
    print(dom_lb)


try3()

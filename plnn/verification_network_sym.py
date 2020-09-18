import hashlib
import os
import pickle
import time
import sympy
# import gurobipy as grb
import numpy as np
import torch
from sympy import symarray
from torch import nn as nn

from agents.dqn.dqn_sequential import TestNetwork
from plnn.flatten_layer import Flatten
from symbolic.symbolic_interval import Interval_network, Symbolic_interval

use_cuda = False
device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")


class SymVerificationNetwork(nn.Module):
    def __init__(self, base_network):
        """Base network any nn.sequential network"""
        super(SymVerificationNetwork, self).__init__()
        self.base_network = base_network

    '''need to  repeat this method for each class so that it describes the distance between the corresponding class 
    and the closest other class'''

    def attach_property_layers(self, true_class_index: int):
        n_classes = self.base_network[-1].out_features
        cases = []
        for i in range(n_classes):
            if i == true_class_index:
                continue
            case = [0] * n_classes  # list of zeroes
            case[true_class_index] = 1  # sets the property to 1
            case[i] = -1
            cases.append(case)
        weight_tensor = torch.tensor(cases, dtype=self.base_network[-1].weight.dtype).to(device)
        property_layer = torch.nn.Linear(n_classes, 1, bias=False)
        with torch.no_grad():
            property_layer.weight = torch.nn.Parameter(weight_tensor)
        return property_layer

    def forward_verif(self, x, true_class_index):
        verif_network = []
        verif_network.extend(self.base_network)
        verif_network.append(self.attach_property_layers(true_class_index))
        net = torch.nn.Sequential(*verif_network)
        result = net(x)
        return torch.min(result, dim=1, keepdim=True)[0]

    def forward(self, x):
        #         x = self.base_network(x)
        x = self.base_network(x)
        return x

    def substitute_array(self, prefix, array):
        substitution_dict = dict()
        for index, x in np.ndenumerate(array):
            key = prefix + "".join([f"_{i}" for i in index])
            substitution_dict[key] = array[index]
        return substitution_dict

    def get_boundaries(self, interval: Symbolic_interval, true_class_index):
        '''
        input_domain: Tensor containing in each row the lower and upper bound
                      for the corresponding dimension
        '''
        verif_network = []
        verif_network.extend(self.base_network)
        verif_network.append(self.attach_property_layers(true_class_index))
        net = torch.nn.Sequential(*verif_network)
        inet = Interval_network(net, None)
        result_interval = inet(interval)
        upper_bound = result_interval.u
        lower_bound = result_interval.l
        return upper_bound, lower_bound


if __name__ == '__main__':
    net = TestNetwork()
    verif = SymVerificationNetwork(net.sequential)
    ix = Symbolic_interval(lower=torch.tensor([[4, -3]], dtype=torch.float64,requires_grad=False), upper=torch.tensor([[6, 5]], dtype=torch.float64,requires_grad=False))
    verif.get_boundaries(ix, 0)

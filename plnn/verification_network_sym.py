import hashlib
import os
import pickle
import time
import sympy
import gurobipy as grb
import numpy as np
import torch
from sympy import symarray
from torch import nn as nn

from models.model_critic_sequential import TestNetwork
from plnn.flatten import Flatten

use_cuda = False
device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")


class SymVerificationNetwork(nn.Module):
    def __init__(self, base_network):
        super(SymVerificationNetwork, self).__init__()
        self.base_network = base_network

    '''need to  repeat this method for each class so that it describes the distance between the corresponding class 
    and the closest other class'''

    def attach_property_layers(self, true_class_index: int):
        n_classes = self.base_network.layers[-1].out_features
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
        weight_tensor = torch.from_numpy(weights).float().to(device)
        # print(f'final weightTensor size={weight_tensor.size()}')
        return weight_tensor

    def forward_verif(self, x, true_class_index):
        x = self.base_network(x)
        property_layer = self.attach_property_layers(true_class_index)
        result = torch.matmul(x, torch.t(property_layer))
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

    def get_boundaries(self, domain, true_class_index):
        '''
        input_domain: Tensor containing in each row the lower and upper bound
                      for the corresponding dimension
        '''
        # now try to do the lower bound

        batch_size = 1  # domain.size()[1]
        # for index in range(batch_size):
        input_domain = domain  # we use a single domain, not ready for parallelisation yet
        # print(f'input_domain.size()={input_domain.size()}')
        lower_bounds = []
        upper_bounds = []
        # These three are nested lists. Each of their elements will itself be a
        # list of the neurons after a layer.

        # Do the input layer, which is a special case
        inp_lb = symarray('lb', input_domain.shape[:-2])
        inp_ub = symarray('ub', input_domain.shape[:-2])
        lower_bounds.append(inp_lb)
        upper_bounds.append(inp_ub)

        layers = []
        layers.extend(self.base_network.layers)
        # layers.append(self.attach_property_layers(true_class_index))
        layer_idx = 1
        for layer in layers:
            if type(layer) is nn.Linear or type(layer) is torch.Tensor:
                if type(layer) is nn.Linear:
                    weight = layer.weight if len(layer.weight.shape) > 1 else layer.weight.unsqueeze(0)
                    bias = layer.bias
                else:
                    weight = layer
                    bias = None
                shape = np.array(weight.shape, dtype=int)
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]

                # out_size = np.array(list(shape[:-1]))
                # out_size = out_size.astype(dtype=int)
                new_layer_lb = np.empty(shape[:-1], dtype=object)
                new_layer_ub = np.empty(shape[:-1], dtype=object)
                for row in range(shape[0]):  # j
                    if bias is None:
                        ub = 0
                        lb = 0
                    else:
                        ub = bias.data[row].item()
                        lb = bias.data[row].item()
                    for column in range(shape[1]):
                        coeff = weight.data[row][column].item()  # picks the weight between the two neurons
                        if coeff == 0:
                            continue
                        if coeff >= 0:
                            ub = ub + coeff * old_layer_ub[column]  # multiplies the ub
                            lb = lb + coeff * old_layer_lb[column]  # multiplies the lb
                        else:  # inverted
                            ub = ub + coeff * old_layer_lb[column]  # multiplies the lb
                            lb = lb + coeff * old_layer_ub[column]  # multiplies the ub
                    new_layer_lb[row] = lb
                    new_layer_ub[row] = ub
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
            elif type(layer) == nn.ReLU:
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                substi_dict_ub = self.substitute_array("ub", domain[:, 1, 0].numpy())
                substi_dict_lb = self.substitute_array("lb", domain[:, 0, 0].numpy())
                # concrete_ub = np.array([xi.evalf(subs=substi_dict_ub) for xi in old_layer_ub])
                # concrete_lb = np.array([xi.evalf(subs=substi_dict_lb) for xi in old_layer_lb])
                previous_layer_size = np.array(upper_bounds[-1].shape)
                new_layer_lb = np.empty(previous_layer_size, dtype=object)
                new_layer_ub = np.empty(previous_layer_size, dtype=object)
                for row in range(previous_layer_size[0]):
                    pre_lb = old_layer_lb[row].evalf(subs=substi_dict_lb)
                    pre_ub = old_layer_ub[row].evalf(subs=substi_dict_ub)
                    if pre_lb >= 0 and pre_ub >= 0:
                        # The ReLU is always passing
                        lb = old_layer_lb[row]
                        ub = old_layer_ub[row]
                    elif pre_lb <= 0 and pre_ub <= 0:
                        lb = 0
                        ub = 0
                    else:
                        # here the magic happens
                        slope = pre_ub / (pre_ub - pre_lb)
                        lb = slope * old_layer_lb[row]
                        ub = slope * (old_layer_ub[row] - pre_lb)
                    new_layer_lb[row] = lb
                    new_layer_ub[row] = ub
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
            layer_idx += 1

        # last layer, minimise
        substi_dict_ub = self.substitute_array("ub", domain[:, 1, 0].numpy())
        substi_dict_lb = self.substitute_array("lb", domain[:, 0, 0].numpy())
        substi_dict = dict()
        substi_dict.update(substi_dict_ub)
        substi_dict.update(substi_dict_lb)
        concrete_ub = np.array([xi.evalf(subs=substi_dict,chop=True) for xi in upper_bounds[-1]])
        concrete_lb = np.array([xi.evalf(subs=substi_dict,chop=True) for xi in lower_bounds[-1]])
        lower_bound = min(lower_bounds[-1])
        upper_bound = max(upper_bounds[-1])
        assert lower_bound <= upper_bound
        return upper_bound, lower_bound


if __name__ == '__main__':
    net = TestNetwork()
    result = net(torch.tensor([6, 5], dtype=torch.float64))
    print(result)
    verif = SymVerificationNetwork(net)
    verif.get_boundaries(torch.tensor([[[4], [6]], [[1], [5]]], dtype=torch.float64), 1)

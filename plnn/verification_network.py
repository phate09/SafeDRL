import hashlib
import os
import pickle
import time

import gurobipy as grb
import numpy as np
import torch
from torch import nn as nn

from plnn.flatten_layer import Flatten

use_cuda = False
device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")


class VerificationNetwork(nn.Module):
    def __init__(self, base_network):
        super(VerificationNetwork, self).__init__()
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

    def get_upper_bound(self, domain, true_class_index):
        # we try get_upper_bound
        nb_samples = 1024
        nb_inp = domain.size()[0]  # get last dimensions
        # print(nb_inp)
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        rand_samples_size = [nb_samples, nb_inp]
        rand_samples = torch.zeros(rand_samples_size).to(device)
        rand_samples.uniform_(0, 1)
        domain_lb = domain.select(-1, 0).contiguous()
        domain_ub = domain.select(-1, 1).contiguous()
        domain_width = domain_ub - domain_lb
        # domain_lb = domain_lb.view([1] + list(nb_inp)).expand(
        #     [nb_samples] + list(nb_inp))  # expand the initial point for the number of examples
        # domain_width = domain_width.view([1] + list(nb_inp)).expand(
        #     [nb_samples] + list(nb_inp))  # expand the width for the number of examples
        inps = domain_lb + domain_width * rand_samples
        # now flatten the first dimension into the second
        # flattened_size = [inps.size(0) * inps.size(1)] + list(inps.size()[2:])
        # print(flattened_size)
        # rearrange the tensor so that is consumable by the model
        # print(self.input_size)
        # examples_data_size = [nb_samples] + list(self.input_size[1:])  # the expected dimension of the example tensor
        # print(examples_data_size)
        # var_inps = inps.view(examples_data_size)
        # if var_inps.size() != self.input_size: print(f"var_inps != input_size , {var_inps}/{self.input_size}")  # should match input_size
        outs = self.forward_verif(inps, true_class_index)  # gets the input for the values
        # print(outs.size())
        # print(outs[0])  # those two should be very similar but different because they belong to two different examples
        # print(outs[1])
        # print(outs.size())
        outs_true_class_resized = outs.squeeze(1)
        # print(outs_true_class_resized.size())  # resize outputs so that they each row is a different element of each batch
        upper_bound, idx = torch.min(outs_true_class_resized,
                                     dim=0)  # this returns the distance of the network output from the given class, it selects the class which is furthest from the current one
        # print(f'idx size {idx.size()}')
        # print(f'inps size {inps.size()}')
        # print(idx.item())
        # upper_bound = upper_bound[0]
        # unsqueezed_idx = idx.view(-1, 1)
        # print(f'single size {inps.select(0, idx.item()).size()}')
        ub_point = inps.select(0,
                               idx.item())  # torch.tensor([inps[x][idx[x]][:].cpu().numpy() for x in range(idx.size()[0])]).to(device)  # ub_point represents the input that amongst all examples returns the minimum response for the appropriate class
        return ub_point, upper_bound.item()

    def get_lower_bound(self, domain, true_class_index, save=True):
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
        gurobi_vars = []
        # These three are nested lists. Each of their elements will itself be a
        # list of the neurons after a layer.

        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', False)
        gurobi_model.setParam('Threads', 1)

        # Do the input layer, which is a special case
        inp_lb = np.ndarray(input_domain.shape[:-1])
        inp_ub = np.ndarray(input_domain.shape[:-1])
        inp_gurobi_vars = np.ndarray(input_domain.shape[:-1], dtype=grb.Var)
        # for channel in range(input_domain.shape[0]):
        #     for i in range(input_domain.shape[1]):
        #         for j in range(input_domain.shape[2]):
        #             ub = input_domain[channel][i][j][1].item()  # check this value, it can be messed up
        #             lb = input_domain[channel][i][j][0].item()
        #             assert ub > lb, "ub should be greater that lb"
        #             v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
        #                                     vtype=grb.GRB.CONTINUOUS,
        #                                     name=f'inp_{channel}_{i}_{j}')
        #             inp_gurobi_vars[channel][i][j] = v
        #             inp_lb[channel][i][j] = lb
        #             inp_ub[channel][i][j] = ub
        for i in range(input_domain.shape[0]):
            ub = input_domain[i][1].item()  # check this value, it can be messed up
            lb = input_domain[i][0].item()
            assert ub > lb, "ub should be greater that lb"
            v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                    vtype=grb.GRB.CONTINUOUS,
                                    name=f'inp_{i}')
            inp_gurobi_vars[i] = v
            inp_lb[i] = lb
            inp_ub[i] = ub

        gurobi_model.update()

        lower_bounds.append(inp_lb)
        upper_bounds.append(inp_ub)
        gurobi_vars.append(inp_gurobi_vars)

        layers = []
        layers.extend(self.base_network.layers)
        layers.append(self.attach_property_layers(true_class_index))
        layer_idx = 1
        for layer in layers:

            file_name = hashlib.sha224(domain.numpy().view(np.uint8)).hexdigest()
            if os.path.isfile(f'./data/{file_name}-{layer_idx}.mps'):
                # print(f'opening {file_name}-{layer_idx}')
                with open(f'./data/{file_name}-{layer_idx}.lb', 'rb') as f_lb:
                    new_layer_lb = pickle.load(f_lb)
                with open(f'./data/{file_name}-{layer_idx}.ub', 'rb') as f_ub:
                    new_layer_ub = pickle.load(f_ub)
                gurobi_model = grb.read(f'./data/{file_name}-{layer_idx}.mps')
                gurobi_model.setParam('OutputFlag', False)
                gurobi_model.setParam('Threads', 1)
                gurobi_model.update()
                new_layer_gurobi_vars = np.asarray([x for x in gurobi_model.getVars() if x.VarName.startswith(f'lay{layer_idx}')], dtype=grb.Var)
                new_layer_gurobi_vars = new_layer_gurobi_vars.reshape(new_layer_lb.shape)
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
                layer_idx += 1
                continue

            if type(layer) is nn.Linear:
                weight = layer.weight
                bias = layer.bias
                shape = np.array(weight.shape)
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]

                out_size = np.array(list(shape[:-1]))
                out_size = out_size.astype(dtype=int)
                new_layer_lb = np.zeros(out_size)
                new_layer_ub = np.zeros(out_size)
                new_layer_gurobi_vars = np.ndarray(out_size, dtype=grb.Var)
                for row in range(shape[0]):  # j
                    if bias is None:
                        ub = 0
                        lb = 0
                        lin_expr = 0
                    else:
                        ub = bias.data[row].item()
                        lb = bias.data[row].item()
                        lin_expr = bias.data[row].item()  # adds the bias to the linear expression
                    for column in range(shape[1]):
                        coeff = weight.data[row][column].item()  # picks the weight between the two neurons
                        if coeff == 0:
                            continue
                        if coeff >= 0:
                            ub = ub + coeff * old_layer_ub[column]  # multiplies the ub
                            lb = lb + coeff * old_layer_lb[column]  # multiplies the lb
                        else:  # inverted
                            ub = ub + coeff * old_layer_lb[column]  # multiplies the ub
                            lb = lb + coeff * old_layer_ub[column]  # multiplies the lb
                        lin_expr = lin_expr + coeff * old_layer_gurobi_vars[column]  # multiplies the unknown by the coefficient

                    v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{layer_idx}_{row}')
                    gurobi_model.addConstr(v == lin_expr)
                    gurobi_model.update()
                    #     print(f'v={v}')
                    gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                    gurobi_model.optimize()
                    #          print(f'gurobi status {gurobi_model.status}')
                    assert gurobi_model.status == 2, "LP wasn't optimally solved"
                    # We have computed a lower bound
                    lb = v.X
                    v.lb = lb

                    # Let's now compute an upper bound
                    gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                    gurobi_model.update()
                    gurobi_model.reset()
                    gurobi_model.optimize()
                    assert gurobi_model.status == 2, "LP wasn't optimally solved"
                    ub = v.X
                    v.ub = ub

                    new_layer_lb[row] = lb
                    new_layer_ub[row] = ub
                    new_layer_gurobi_vars[row] = v

                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) is torch.Tensor:
                weight = layer
                bias = None
                shape = np.array(weight.shape)
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]

                out_size = np.array(list(shape[:-1]))
                out_size = out_size.astype(dtype=int)
                new_layer_lb = np.zeros(out_size)
                new_layer_ub = np.zeros(out_size)
                new_layer_gurobi_vars = np.ndarray(out_size, dtype=grb.Var)
                for row in range(shape[0]):  # j
                    if bias is None:
                        ub = 0
                        lb = 0
                        lin_expr = 0
                    else:
                        ub = bias.data[row].item()
                        lb = bias.data[row].item()
                        lin_expr = bias.data[row].item()  # adds the bias to the linear expression
                    for column in range(shape[1]):
                        coeff = weight.data[row][column].item()  # picks the weight between the two neurons
                        if coeff == 0:
                            continue
                        if coeff >= 0:
                            ub = ub + coeff * old_layer_ub[column]  # multiplies the ub
                            lb = lb + coeff * old_layer_lb[column]  # multiplies the lb
                        else:  # inverted
                            ub = ub + coeff * old_layer_lb[column]  # multiplies the ub
                            lb = lb + coeff * old_layer_ub[column]  # multiplies the lb
                        lin_expr = lin_expr + coeff * old_layer_gurobi_vars[column]  # multiplies the unknown by the coefficient

                    v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{layer_idx}_{row}')
                    gurobi_model.addConstr(v == lin_expr)
                    gurobi_model.update()
                    gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                    gurobi_model.optimize()
                    assert gurobi_model.status == 2, "LP wasn't optimally solved"
                    # We have computed a lower bound
                    lb = v.X
                    v.lb = lb

                    # Let's now compute an upper bound
                    gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                    gurobi_model.update()
                    gurobi_model.reset()
                    gurobi_model.optimize()
                    assert gurobi_model.status == 2, "LP wasn't optimally solved"
                    ub = v.X
                    v.ub = ub

                    new_layer_lb[row] = lb
                    new_layer_ub[row] = ub
                    new_layer_gurobi_vars[row] = v

                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) == nn.ReLU:
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]
                previous_layer_size = np.array(old_layer_gurobi_vars.shape)
                new_layer_lb = np.zeros(previous_layer_size)
                new_layer_ub = np.zeros(previous_layer_size)
                new_layer_gurobi_vars = np.ndarray(previous_layer_size, dtype=grb.Var)

                for row in range(previous_layer_size[0]):
                    pre_lb = old_layer_lb[row]
                    pre_ub = old_layer_ub[row]

                    v = gurobi_model.addVar(lb=max(0, pre_lb),
                                            ub=max(0, pre_ub),
                                            obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{layer_idx}_{row}')
                    if pre_lb >= 0 and pre_ub >= 0:
                        # The ReLU is always passing
                        gurobi_model.addConstr(v == old_layer_gurobi_vars[row])
                        lb = pre_lb
                        ub = pre_ub
                    elif pre_lb <= 0 and pre_ub <= 0:
                        lb = 0
                        ub = 0
                        # No need to add an additional constraint that v==0
                        # because this will be covered by the bounds we set on
                        # the value of v.
                    else:
                        lb = 0
                        ub = pre_ub
                        gurobi_model.addConstr(v >= old_layer_gurobi_vars[row])

                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        gurobi_model.addConstr(v <= slope * old_layer_gurobi_vars[row] + bias)

                    new_layer_lb[row] = lb
                    new_layer_ub[row] = ub
                    new_layer_gurobi_vars[row] = v
                gurobi_model.update()
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) == nn.MaxPool2d:
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]
                t1_start = time.perf_counter()
                t2_start = time.process_time()
                # Compute convolution
                ch_in, h_in, w_in = old_layer_lb.shape
                k_h = layer.kernel_size
                k_w = layer.kernel_size
                padding = layer.padding  # np.array(layer.padding)
                stride = layer.stride  # np.array(layer.stride)
                h_out = int((h_in + 2 * padding - k_h) / stride + 1)
                w_out = int((w_in + 2 * padding - k_w) / stride + 1)
                # out_channels = np.array(layer.out_channels)
                previous_layer_size = np.array(gurobi_vars[-1].shape)
                out_size = (ch_in, h_out, w_out)
                new_layer_lb = np.zeros(out_size)
                new_layer_ub = np.zeros(out_size)
                new_layer_gurobi_vars = np.ndarray(out_size, dtype=grb.Var)
                for channel in range(ch_in):
                    for row in range(h_out):
                        for col in range(w_out):
                            lin_expr = 0  # layer.bias[channel].item()
                            lower_bounds_section = old_layer_lb[channel, col * stride - padding:col * stride - padding + k_w, row * stride - padding:row * stride - padding + k_h]
                            lb = lower_bounds_section.max()
                            upper_bounds_section = old_layer_ub[channel, col * stride - padding:col * stride - padding + k_w, row * stride - padding:row * stride - padding + k_h]
                            ub = upper_bounds_section.max()
                            v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                                    vtype=grb.GRB.CONTINUOUS,
                                                    name=f'lay{layer_idx}_{channel}_{row}_{col}')
                            gurobi_model.addConstr(v >= lb)
                            gurobi_model.addConstr(v <= ub)
                            gurobi_model.update()
                            gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                            gurobi_model.optimize()
                            assert gurobi_model.status == 2, "LP wasn't optimally solved"
                            # We have computed a lower bound
                            lb = v.X
                            v.lb = lb

                            # Let's now compute an upper bound
                            gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                            gurobi_model.update()
                            gurobi_model.reset()
                            gurobi_model.optimize()
                            assert gurobi_model.status == 2, "LP wasn't optimally solved"
                            ub = v.X
                            v.ub = ub
                            new_layer_lb[channel][row][col] = lb
                            new_layer_ub[channel][row][col] = ub
                            new_layer_gurobi_vars[channel][row][col] = v
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
                t1_stop = time.perf_counter()
                t2_stop = time.process_time()
                # print(f"End MaxPool2d{layer_idx} {((t1_stop - t1_start)):.1f} [sec]")
            elif type(layer) == Flatten:
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]

                new_layer_lb = old_layer_lb.flatten()
                new_layer_ub = old_layer_ub.flatten()
                new_layer_gurobi_vars = np.ndarray(new_layer_ub.shape, grb.Var)  # old_layer_gurobi_vars.flatten()
                index = 0
                for var in old_layer_gurobi_vars.flatten():
                    v = gurobi_model.addVar(lb=var.lb, ub=var.ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{layer_idx}_{index}')
                    new_layer_gurobi_vars[index] = v
                    index += 1
                gurobi_model.update()
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)

            elif type(layer) == nn.Conv2d:
                # print(f"Start Conv2d_{layer_idx}")
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]
                t1_start = time.perf_counter()
                t2_start = time.process_time()
                # Compute convolution
                ch_in, h_in, w_in = old_layer_lb.shape
                channels_out, channel_in, k_h, k_w = np.array(layer.weight.shape)
                padding = np.array(layer.padding)
                stride = np.array(layer.stride)
                h_out = int((h_in + 2 * padding[0] - k_h) / stride[0] + 1)
                w_out = int((w_in + 2 * padding[1] - k_w) / stride[1] + 1)
                # out_channels = np.array(layer.out_channels)
                previous_layer_size = np.array(gurobi_vars[-1].shape)
                out_size = (channels_out, h_out, w_out)
                new_layer_lb = np.zeros(out_size)
                new_layer_ub = np.zeros(out_size)
                new_layer_gurobi_vars = np.ndarray(out_size, dtype=grb.Var)

                # conv_index = torch.Tensor(range(nb_pre)).view(previous_layer_size[:-1])  # index of the gurobi vars in the flat representation
                for channel in range(channels_out):
                    for row in range(h_out):
                        for col in range(w_out):
                            lb = layer.bias[channel].item()
                            ub = layer.bias[channel].item()
                            lin_expr = layer.bias[channel].item()
                            for kernel_channel in range(channel_in):
                                for i in range(k_h):
                                    for j in range(k_w):
                                        row_prev_layer = row * stride[0] - padding[0] + i
                                        col_prev_layer = col * stride[1] - padding[1] + j
                                        if row_prev_layer < 0 or row_prev_layer > \
                                                old_layer_lb[kernel_channel].shape[0]:
                                            continue
                                        if col_prev_layer < 0 or col_prev_layer > \
                                                old_layer_lb[kernel_channel].shape[1]:
                                            continue
                                        coeff = layer.weight[channel][kernel_channel][i][j].item()
                                        if coeff > 0:
                                            lb += coeff * old_layer_lb[kernel_channel][row_prev_layer][col_prev_layer]
                                            ub += coeff * old_layer_ub[kernel_channel][row_prev_layer][col_prev_layer]
                                        else:  # invert lower bound and upper bound
                                            lb += coeff * old_layer_ub[kernel_channel][row_prev_layer][col_prev_layer]
                                            ub += coeff * old_layer_lb[kernel_channel][row_prev_layer][col_prev_layer]
                                        lin_expr = lin_expr + coeff * gurobi_vars[-1][kernel_channel][row_prev_layer][col_prev_layer]
                            v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                                    vtype=grb.GRB.CONTINUOUS,
                                                    name=f'lay{layer_idx}_{channel}_{row}_{col}')
                            gurobi_model.addConstr(v == lin_expr)
                            gurobi_model.update()
                            gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                            gurobi_model.optimize()
                            assert gurobi_model.status == 2, "LP wasn't optimally solved"
                            # We have computed a lower bound
                            lb = v.X
                            v.lb = lb

                            # Let's now compute an upper bound
                            gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                            gurobi_model.update()
                            gurobi_model.reset()
                            gurobi_model.optimize()
                            assert gurobi_model.status == 2, "LP wasn't optimally solved"
                            ub = v.X
                            v.ub = ub
                            new_layer_lb[channel][row][col] = lb
                            new_layer_ub[channel][row][col] = ub
                            new_layer_gurobi_vars[channel][row][col] = v
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
                t1_stop = time.perf_counter()
                t2_stop = time.process_time()
                # print(f"End Conv2d_{layer_idx} {((t1_stop - t1_start)):.1f} [sec]")
            else:
                raise Exception('Type of layer not supported')
            if save:
                with open(f'./data/{file_name}-{layer_idx}.ub', 'wb') as f_ub:
                    pickle.dump(new_layer_ub, f_ub)
                with open(f'./data/{file_name}-{layer_idx}.lb', 'wb') as f_lb:
                    pickle.dump(new_layer_lb, f_lb)
                gurobi_model.ModelName = f'{file_name}'
                gurobi_model.update()
                gurobi_model.write(f'./data/{file_name}-{layer_idx}.mps')
            layer_idx += 1
        # Assert that this is as expected a network with a single output
        # assert len(gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

        # last layer, minimise
        lower_bound = min(lower_bounds[-1])
        upper_bound = max(upper_bounds[-1])
        assert lower_bound <= upper_bound
        v = gurobi_model.addVar(lb=lower_bound, ub=upper_bound, obj=0,
                                vtype=grb.GRB.CONTINUOUS,
                                name=f'lay{layer_idx}_min')
        # gurobi_model.addConstr(v == min(gurobi_vars[-1]))
        gurobi_model.addGenConstrMin(v, gurobi_vars[-1], name="minconstr")
        gurobi_model.update()
        #     print(f'v={v}')
        gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
        gurobi_model.optimize()

        gurobi_model.update()
        gurobi_vars.append([v])

        # We will first setup the appropriate bounds for the elements of the
        # input
        # is it just to be sure?
        # for var_idx, inp_var in enumerate(gurobi_vars[0]):
        #     inp_var.lb = domain[var_idx, 0]
        #     inp_var.ub = domain[var_idx, 1]

        # We will make sure that the objective function is properly set up
        gurobi_model.setObjective(gurobi_vars[-1][0], grb.GRB.MINIMIZE)
        # print(f'gurobi_vars[-1][0].size()={len(gurobi_vars[-1])}')
        # We will now compute the requested lower bound
        gurobi_model.update()
        gurobi_model.optimize()
        assert gurobi_model.status == 2, "LP wasn't optimally solved"
        # print(f'gurobi status {gurobi_model.status}')
        # print(f'Result={gurobi_vars[-1][0].X}')
        # print(f'Result={gurobi_vars[-1]}')
        # print(f'Result -1={gurobi_vars[-2]}')
        return gurobi_vars[-1][0].X

    def get_upper_bound2(self, domain, true_class_index, save=True):
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
        gurobi_vars = []
        # These three are nested lists. Each of their elements will itself be a
        # list of the neurons after a layer.

        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', False)
        gurobi_model.setParam('Threads', 1)

        # Do the input layer, which is a special case
        inp_lb = np.ndarray(input_domain.shape[:-1])
        inp_ub = np.ndarray(input_domain.shape[:-1])
        inp_gurobi_vars = np.ndarray(input_domain.shape[:-1], dtype=grb.Var)
        for i in range(input_domain.shape[0]):
            ub = input_domain[i][1].item()  # check this value, it can be messed up
            lb = input_domain[i][0].item()
            assert ub > lb, "ub should be greater that lb"
            v = gurobi_model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'inp_{i}')
            inp_gurobi_vars[i] = v
            inp_lb[i] = lb
            inp_ub[i] = ub

        gurobi_model.update()

        lower_bounds.append(inp_lb)
        upper_bounds.append(inp_ub)
        gurobi_vars.append(inp_gurobi_vars)

        layers = []
        layers.extend(self.base_network.layers)
        layers.append(self.attach_property_layers(true_class_index))
        layer_idx = 1
        for layer in layers:

            file_name = hashlib.sha224(domain.numpy().view(np.uint8)).hexdigest()
            if os.path.isfile(f'./data/{file_name}-{layer_idx}.mps'):
                # print(f'opening {file_name}-{layer_idx}')
                with open(f'./data/{file_name}-{layer_idx}.lb', 'rb') as f_lb:
                    new_layer_lb = pickle.load(f_lb)
                with open(f'./data/{file_name}-{layer_idx}.ub', 'rb') as f_ub:
                    new_layer_ub = pickle.load(f_ub)
                gurobi_model = grb.read(f'./data/{file_name}-{layer_idx}.mps')
                gurobi_model.setParam('OutputFlag', False)
                gurobi_model.setParam('Threads', 1)
                gurobi_model.update()
                new_layer_gurobi_vars = np.asarray([x for x in gurobi_model.getVars() if x.VarName.startswith(f'lay{layer_idx}')], dtype=grb.Var)
                new_layer_gurobi_vars = new_layer_gurobi_vars.reshape(new_layer_lb.shape)
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
                layer_idx += 1
                continue

            if type(layer) is nn.Linear:
                weight = layer.weight
                bias = layer.bias
                shape = np.array(weight.shape)
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]

                out_size = np.array(list(shape[:-1]))
                out_size = out_size.astype(dtype=int)
                new_layer_lb = np.zeros(out_size)
                new_layer_ub = np.zeros(out_size)
                new_layer_gurobi_vars = np.ndarray(out_size, dtype=grb.Var)
                for row in range(shape[0]):  # j
                    if bias is None:
                        ub = 0
                        lb = 0
                        lin_expr = 0
                    else:
                        ub = bias.data[row].item()
                        lb = bias.data[row].item()
                        lin_expr = bias.data[row].item()  # adds the bias to the linear expression
                    for column in range(shape[1]):
                        coeff = weight.data[row][column].item()  # picks the weight between the two neurons
                        if coeff == 0:
                            continue
                        if coeff >= 0:
                            ub = ub + coeff * old_layer_ub[column]  # multiplies the ub
                            lb = lb + coeff * old_layer_lb[column]  # multiplies the lb
                        else:  # inverted
                            ub = ub + coeff * old_layer_lb[column]  # multiplies the ub
                            lb = lb + coeff * old_layer_ub[column]  # multiplies the lb
                        lin_expr = lin_expr + coeff * old_layer_gurobi_vars[column]  # multiplies the unknown by the coefficient

                    v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{layer_idx}_{row}')
                    gurobi_model.addConstr(v == lin_expr)
                    gurobi_model.update()
                    #     print(f'v={v}')
                    gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                    gurobi_model.optimize()
                    #          print(f'gurobi status {gurobi_model.status}')
                    assert gurobi_model.status == 2, "LP wasn't optimally solved"
                    # We have computed a lower bound
                    lb = v.X
                    v.lb = lb

                    # Let's now compute an upper bound
                    gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                    gurobi_model.update()
                    gurobi_model.reset()
                    gurobi_model.optimize()
                    assert gurobi_model.status == 2, "LP wasn't optimally solved"
                    ub = v.X
                    v.ub = ub

                    new_layer_lb[row] = lb
                    new_layer_ub[row] = ub
                    new_layer_gurobi_vars[row] = v

                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) is torch.Tensor:
                weight = layer
                bias = None
                shape = np.array(weight.shape)
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]

                out_size = np.array(list(shape[:-1]))
                out_size = out_size.astype(dtype=int)
                new_layer_lb = np.zeros(out_size)
                new_layer_ub = np.zeros(out_size)
                new_layer_gurobi_vars = np.ndarray(out_size, dtype=grb.Var)
                for row in range(shape[0]):  # j
                    if bias is None:
                        ub = 0
                        lb = 0
                        lin_expr = 0
                    else:
                        ub = bias.data[row].item()
                        lb = bias.data[row].item()
                        lin_expr = bias.data[row].item()  # adds the bias to the linear expression
                    for column in range(shape[1]):
                        coeff = weight.data[row][column].item()  # picks the weight between the two neurons
                        if coeff == 0:
                            continue
                        if coeff >= 0:
                            ub = ub + coeff * old_layer_ub[column]  # multiplies the ub
                            lb = lb + coeff * old_layer_lb[column]  # multiplies the lb
                        else:  # inverted
                            ub = ub + coeff * old_layer_lb[column]  # multiplies the ub
                            lb = lb + coeff * old_layer_ub[column]  # multiplies the lb
                        lin_expr = lin_expr + coeff * old_layer_gurobi_vars[column]  # multiplies the unknown by the coefficient

                    v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{layer_idx}_{row}')
                    gurobi_model.addConstr(v == lin_expr)
                    gurobi_model.update()
                    gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                    gurobi_model.optimize()
                    assert gurobi_model.status == 2, "LP wasn't optimally solved"
                    # We have computed a lower bound
                    lb = v.X
                    v.lb = lb

                    # Let's now compute an upper bound
                    gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                    gurobi_model.update()
                    gurobi_model.reset()
                    gurobi_model.optimize()
                    assert gurobi_model.status == 2, "LP wasn't optimally solved"
                    ub = v.X
                    v.ub = ub

                    new_layer_lb[row] = lb
                    new_layer_ub[row] = ub
                    new_layer_gurobi_vars[row] = v

                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) == nn.ReLU:
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]
                previous_layer_size = np.array(old_layer_gurobi_vars.shape)
                new_layer_lb = np.zeros(previous_layer_size)
                new_layer_ub = np.zeros(previous_layer_size)
                new_layer_gurobi_vars = np.ndarray(previous_layer_size, dtype=grb.Var)

                for row in range(previous_layer_size[0]):
                    pre_lb = old_layer_lb[row]
                    pre_ub = old_layer_ub[row]

                    v = gurobi_model.addVar(lb=max(0, pre_lb),
                                            ub=max(0, pre_ub),
                                            obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{layer_idx}_{row}')
                    if pre_lb >= 0 and pre_ub >= 0:
                        # The ReLU is always passing
                        gurobi_model.addConstr(v == old_layer_gurobi_vars[row])
                        lb = pre_lb
                        ub = pre_ub
                    elif pre_lb <= 0 and pre_ub <= 0:
                        lb = 0
                        ub = 0
                        # No need to add an additional constraint that v==0
                        # because this will be covered by the bounds we set on
                        # the value of v.
                    else:
                        lb = 0
                        ub = pre_ub
                        gurobi_model.addConstr(v >= old_layer_gurobi_vars[row])

                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        gurobi_model.addConstr(v <= slope * old_layer_gurobi_vars[row] + bias)

                    new_layer_lb[row] = lb
                    new_layer_ub[row] = ub
                    new_layer_gurobi_vars[row] = v
                gurobi_model.update()
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) == nn.MaxPool2d:
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]
                t1_start = time.perf_counter()
                t2_start = time.process_time()
                # Compute convolution
                ch_in, h_in, w_in = old_layer_lb.shape
                k_h = layer.kernel_size
                k_w = layer.kernel_size
                padding = layer.padding  # np.array(layer.padding)
                stride = layer.stride  # np.array(layer.stride)
                h_out = int((h_in + 2 * padding - k_h) / stride + 1)
                w_out = int((w_in + 2 * padding - k_w) / stride + 1)
                # out_channels = np.array(layer.out_channels)
                previous_layer_size = np.array(gurobi_vars[-1].shape)
                out_size = (ch_in, h_out, w_out)
                new_layer_lb = np.zeros(out_size)
                new_layer_ub = np.zeros(out_size)
                new_layer_gurobi_vars = np.ndarray(out_size, dtype=grb.Var)
                for channel in range(ch_in):
                    for row in range(h_out):
                        for col in range(w_out):
                            lin_expr = 0  # layer.bias[channel].item()
                            lower_bounds_section = old_layer_lb[channel, col * stride - padding:col * stride - padding + k_w, row * stride - padding:row * stride - padding + k_h]
                            lb = lower_bounds_section.max()
                            upper_bounds_section = old_layer_ub[channel, col * stride - padding:col * stride - padding + k_w, row * stride - padding:row * stride - padding + k_h]
                            ub = upper_bounds_section.max()
                            v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                                    vtype=grb.GRB.CONTINUOUS,
                                                    name=f'lay{layer_idx}_{channel}_{row}_{col}')
                            gurobi_model.addConstr(v >= lb)
                            gurobi_model.addConstr(v <= ub)
                            gurobi_model.update()
                            gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                            gurobi_model.optimize()
                            assert gurobi_model.status == 2, "LP wasn't optimally solved"
                            # We have computed a lower bound
                            lb = v.X
                            v.lb = lb

                            # Let's now compute an upper bound
                            gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                            gurobi_model.update()
                            gurobi_model.reset()
                            gurobi_model.optimize()
                            assert gurobi_model.status == 2, "LP wasn't optimally solved"
                            ub = v.X
                            v.ub = ub
                            new_layer_lb[channel][row][col] = lb
                            new_layer_ub[channel][row][col] = ub
                            new_layer_gurobi_vars[channel][row][col] = v
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
                t1_stop = time.perf_counter()
                t2_stop = time.process_time()
                # print(f"End MaxPool2d{layer_idx} {((t1_stop - t1_start)):.1f} [sec]")
            elif type(layer) == Flatten:
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]

                new_layer_lb = old_layer_lb.flatten()
                new_layer_ub = old_layer_ub.flatten()
                new_layer_gurobi_vars = np.ndarray(new_layer_ub.shape, grb.Var)  # old_layer_gurobi_vars.flatten()
                index = 0
                for var in old_layer_gurobi_vars.flatten():
                    v = gurobi_model.addVar(lb=var.lb, ub=var.ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{layer_idx}_{index}')
                    new_layer_gurobi_vars[index] = v
                    index += 1
                gurobi_model.update()
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)

            elif type(layer) == nn.Conv2d:
                # print(f"Start Conv2d_{layer_idx}")
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]
                t1_start = time.perf_counter()
                t2_start = time.process_time()
                # Compute convolution
                ch_in, h_in, w_in = old_layer_lb.shape
                channels_out, channel_in, k_h, k_w = np.array(layer.weight.shape)
                padding = np.array(layer.padding)
                stride = np.array(layer.stride)
                h_out = int((h_in + 2 * padding[0] - k_h) / stride[0] + 1)
                w_out = int((w_in + 2 * padding[1] - k_w) / stride[1] + 1)
                # out_channels = np.array(layer.out_channels)
                previous_layer_size = np.array(gurobi_vars[-1].shape)
                out_size = (channels_out, h_out, w_out)
                new_layer_lb = np.zeros(out_size)
                new_layer_ub = np.zeros(out_size)
                new_layer_gurobi_vars = np.ndarray(out_size, dtype=grb.Var)

                # conv_index = torch.Tensor(range(nb_pre)).view(previous_layer_size[:-1])  # index of the gurobi vars in the flat representation
                for channel in range(channels_out):
                    for row in range(h_out):
                        for col in range(w_out):
                            lb = layer.bias[channel].item()
                            ub = layer.bias[channel].item()
                            lin_expr = layer.bias[channel].item()
                            for kernel_channel in range(channel_in):
                                for i in range(k_h):
                                    for j in range(k_w):
                                        row_prev_layer = row * stride[0] - padding[0] + i
                                        col_prev_layer = col * stride[1] - padding[1] + j
                                        if row_prev_layer < 0 or row_prev_layer > \
                                                old_layer_lb[kernel_channel].shape[0]:
                                            continue
                                        if col_prev_layer < 0 or col_prev_layer > \
                                                old_layer_lb[kernel_channel].shape[1]:
                                            continue
                                        coeff = layer.weight[channel][kernel_channel][i][j].item()
                                        if coeff > 0:
                                            lb += coeff * old_layer_lb[kernel_channel][row_prev_layer][col_prev_layer]
                                            ub += coeff * old_layer_ub[kernel_channel][row_prev_layer][col_prev_layer]
                                        else:  # invert lower bound and upper bound
                                            lb += coeff * old_layer_ub[kernel_channel][row_prev_layer][col_prev_layer]
                                            ub += coeff * old_layer_lb[kernel_channel][row_prev_layer][col_prev_layer]
                                        lin_expr = lin_expr + coeff * gurobi_vars[-1][kernel_channel][row_prev_layer][col_prev_layer]
                            v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                                    vtype=grb.GRB.CONTINUOUS,
                                                    name=f'lay{layer_idx}_{channel}_{row}_{col}')
                            gurobi_model.addConstr(v == lin_expr)
                            gurobi_model.update()
                            gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                            gurobi_model.optimize()
                            assert gurobi_model.status == 2, "LP wasn't optimally solved"
                            # We have computed a lower bound
                            lb = v.X
                            v.lb = lb

                            # Let's now compute an upper bound
                            gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                            gurobi_model.update()
                            gurobi_model.reset()
                            gurobi_model.optimize()
                            assert gurobi_model.status == 2, "LP wasn't optimally solved"
                            ub = v.X
                            v.ub = ub
                            new_layer_lb[channel][row][col] = lb
                            new_layer_ub[channel][row][col] = ub
                            new_layer_gurobi_vars[channel][row][col] = v
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
                t1_stop = time.perf_counter()
                t2_stop = time.process_time()
                # print(f"End Conv2d_{layer_idx} {((t1_stop - t1_start)):.1f} [sec]")
            else:
                raise Exception('Type of layer not supported')
            if save:
                with open(f'./data/{file_name}-{layer_idx}.ub', 'wb') as f_ub:
                    pickle.dump(new_layer_ub, f_ub)
                with open(f'./data/{file_name}-{layer_idx}.lb', 'wb') as f_lb:
                    pickle.dump(new_layer_lb, f_lb)
                gurobi_model.ModelName = f'{file_name}'
                gurobi_model.update()
                gurobi_model.write(f'./data/{file_name}-{layer_idx}.mps')
            layer_idx += 1
        # Assert that this is as expected a network with a single output
        # assert len(gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

        # last layer, minimise
        lower_bound = min(lower_bounds[-1])
        upper_bound = max(upper_bounds[-1])
        assert lower_bound <= upper_bound
        v = gurobi_model.addVar(lb=lower_bound, ub=upper_bound, obj=0,
                                vtype=grb.GRB.CONTINUOUS,
                                name=f'lay{layer_idx}_max')
        # gurobi_model.addConstr(v == min(gurobi_vars[-1]))
        gurobi_model.addGenConstrMin(v, gurobi_vars[-1], name="maxconstr")
        gurobi_model.update()
        #     print(f'v={v}')
        gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
        gurobi_model.optimize()

        gurobi_model.update()
        gurobi_vars.append([v])

        # We will first setup the appropriate bounds for the elements of the
        # input
        # is it just to be sure?
        # for var_idx, inp_var in enumerate(gurobi_vars[0]):
        #     inp_var.lb = domain[var_idx, 0]
        #     inp_var.ub = domain[var_idx, 1]

        # We will make sure that the objective function is properly set up
        gurobi_model.setObjective(gurobi_vars[-1][0], grb.GRB.MAXIMIZE)
        # print(f'gurobi_vars[-1][0].size()={len(gurobi_vars[-1])}')
        # We will now compute the requested lower bound
        gurobi_model.update()
        gurobi_model.optimize()
        assert gurobi_model.status == 2, "LP wasn't optimally solved"
        # print(f'gurobi status {gurobi_model.status}')
        # print(f'Result={gurobi_vars[-1][0].X}')
        # print(f'Result={gurobi_vars[-1]}')
        # print(f'Result -1={gurobi_vars[-2]}')
        return gurobi_vars[-1][0].X

    def get_boundaries(self, domain, true_class_index, save=True):
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
        gurobi_vars = []
        # These three are nested lists. Each of their elements will itself be a
        # list of the neurons after a layer.

        gurobi_model = grb.Model()
        gurobi_model.setParam('OutputFlag', False)
        gurobi_model.setParam('Threads', 1)

        # Do the input layer, which is a special case
        inp_lb = np.ndarray(input_domain.shape[:-1])
        inp_ub = np.ndarray(input_domain.shape[:-1])
        inp_gurobi_vars = np.ndarray(input_domain.shape[:-1], dtype=grb.Var)
        for i in range(input_domain.shape[0]):
            ub = input_domain[i][1].item()  # check this value, it can be messed up
            lb = input_domain[i][0].item()
            assert ub >= lb, f"ub should be greater that lb, ub:{ub} lb:{lb}"
            v = gurobi_model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'inp_{i}')
            inp_gurobi_vars[i] = v
            inp_lb[i] = lb
            inp_ub[i] = ub

        gurobi_model.update()

        lower_bounds.append(inp_lb)
        upper_bounds.append(inp_ub)
        gurobi_vars.append(inp_gurobi_vars)

        layers = []
        layers.extend(self.base_network.layers)
        layers.append(self.attach_property_layers(true_class_index))
        layer_idx = 1
        for layer in layers:

            # file_name = hashlib.sha224(domain.numpy().view(np.uint8)).hexdigest()
            # if os.path.isfile(f'./data/{file_name}-{layer_idx}.mps'):
            #     # print(f'opening {file_name}-{layer_idx}')
            #     with open(f'./data/{file_name}-{layer_idx}.lb', 'rb') as f_lb:
            #         new_layer_lb = pickle.load(f_lb)
            #     with open(f'./data/{file_name}-{layer_idx}.ub', 'rb') as f_ub:
            #         new_layer_ub = pickle.load(f_ub)
            #     gurobi_model = grb.read(f'./data/{file_name}-{layer_idx}.mps')
            #     gurobi_model.setParam('OutputFlag', False)
            #     gurobi_model.setParam('Threads', 1)
            #     gurobi_model.update()
            #     new_layer_gurobi_vars = np.asarray([x for x in gurobi_model.getVars() if x.VarName.startswith(f'lay{layer_idx}')], dtype=grb.Var)
            #     new_layer_gurobi_vars = new_layer_gurobi_vars.reshape(new_layer_lb.shape)
            #     lower_bounds.append(new_layer_lb)
            #     upper_bounds.append(new_layer_ub)
            #     gurobi_vars.append(new_layer_gurobi_vars)
            #     layer_idx += 1
            #     continue

            if type(layer) is nn.Linear:
                weight = layer.weight
                bias = layer.bias
                shape = np.array(weight.shape)
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]

                out_size = np.array(list(shape[:-1]))
                out_size = out_size.astype(dtype=int)
                new_layer_lb = np.zeros(out_size)
                new_layer_ub = np.zeros(out_size)
                new_layer_gurobi_vars = np.ndarray(out_size, dtype=grb.Var)
                for row in range(shape[0]):  # j
                    if bias is None:
                        ub = 0
                        lb = 0
                        lin_expr = 0
                    else:
                        ub = bias.data[row].item()
                        lb = bias.data[row].item()
                        lin_expr = bias.data[row].item()  # adds the bias to the linear expression
                    for column in range(shape[1]):
                        coeff = weight.data[row][column].item()  # picks the weight between the two neurons
                        if coeff == 0:
                            continue
                        if coeff >= 0:
                            ub = ub + coeff * old_layer_ub[column]  # multiplies the ub
                            lb = lb + coeff * old_layer_lb[column]  # multiplies the lb
                        else:  # inverted
                            ub = ub + coeff * old_layer_lb[column]  # multiplies the ub
                            lb = lb + coeff * old_layer_ub[column]  # multiplies the lb
                        lin_expr = lin_expr + coeff * old_layer_gurobi_vars[column]  # multiplies the unknown by the coefficient

                    v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{layer_idx}_{row}')
                    gurobi_model.addConstr(v == lin_expr)
                    gurobi_model.update()
                    #     print(f'v={v}')
                    gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                    gurobi_model.optimize()
                    #          print(f'gurobi status {gurobi_model.status}')
                    assert gurobi_model.status == 2, "LP wasn't optimally solved"
                    # We have computed a lower bound
                    lb = v.X
                    v.lb = lb

                    # Let's now compute an upper bound
                    gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                    gurobi_model.update()
                    gurobi_model.reset()
                    gurobi_model.optimize()
                    assert gurobi_model.status == 2, "LP wasn't optimally solved"
                    ub = v.X
                    v.ub = ub

                    new_layer_lb[row] = lb
                    new_layer_ub[row] = ub
                    new_layer_gurobi_vars[row] = v

                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) is torch.Tensor:
                weight = layer
                bias = None
                shape = np.array(weight.shape)
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]

                out_size = np.array(list(shape[:-1]))
                out_size = out_size.astype(dtype=int)
                new_layer_lb = np.zeros(out_size)
                new_layer_ub = np.zeros(out_size)
                new_layer_gurobi_vars = np.ndarray(out_size, dtype=grb.Var)
                for row in range(shape[0]):  # j
                    if bias is None:
                        ub = 0
                        lb = 0
                        lin_expr = 0
                    else:
                        ub = bias.data[row].item()
                        lb = bias.data[row].item()
                        lin_expr = bias.data[row].item()  # adds the bias to the linear expression
                    for column in range(shape[1]):
                        coeff = weight.data[row][column].item()  # picks the weight between the two neurons
                        if coeff == 0:
                            continue
                        if coeff >= 0:
                            ub = ub + coeff * old_layer_ub[column]  # multiplies the ub
                            lb = lb + coeff * old_layer_lb[column]  # multiplies the lb
                        else:  # inverted
                            ub = ub + coeff * old_layer_lb[column]  # multiplies the ub
                            lb = lb + coeff * old_layer_ub[column]  # multiplies the lb
                        lin_expr = lin_expr + coeff * old_layer_gurobi_vars[column]  # multiplies the unknown by the coefficient

                    v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{layer_idx}_{row}')
                    gurobi_model.addConstr(v == lin_expr)
                    gurobi_model.update()
                    gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                    gurobi_model.optimize()
                    assert gurobi_model.status == 2, "LP wasn't optimally solved"
                    # We have computed a lower bound
                    lb = v.X
                    v.lb = lb

                    # Let's now compute an upper bound
                    gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                    gurobi_model.update()
                    gurobi_model.reset()
                    gurobi_model.optimize()
                    assert gurobi_model.status == 2, "LP wasn't optimally solved"
                    ub = v.X
                    v.ub = ub

                    new_layer_lb[row] = lb
                    new_layer_ub[row] = ub
                    new_layer_gurobi_vars[row] = v

                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) == nn.ReLU:
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]
                previous_layer_size = np.array(old_layer_gurobi_vars.shape)
                new_layer_lb = np.zeros(previous_layer_size)
                new_layer_ub = np.zeros(previous_layer_size)
                new_layer_gurobi_vars = np.ndarray(previous_layer_size, dtype=grb.Var)

                for row in range(previous_layer_size[0]):
                    pre_lb = old_layer_lb[row]
                    pre_ub = old_layer_ub[row]

                    v = gurobi_model.addVar(lb=max(0, pre_lb),
                                            ub=max(0, pre_ub),
                                            obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{layer_idx}_{row}')
                    if pre_lb >= 0 and pre_ub >= 0:
                        # The ReLU is always passing
                        gurobi_model.addConstr(v == old_layer_gurobi_vars[row])
                        lb = pre_lb
                        ub = pre_ub
                    elif pre_lb <= 0 and pre_ub <= 0:
                        lb = 0
                        ub = 0
                        # No need to add an additional constraint that v==0
                        # because this will be covered by the bounds we set on
                        # the value of v.
                    else:
                        lb = 0
                        ub = pre_ub
                        gurobi_model.addConstr(v >= old_layer_gurobi_vars[row])

                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        gurobi_model.addConstr(v <= slope * old_layer_gurobi_vars[row] + bias)

                    new_layer_lb[row] = lb
                    new_layer_ub[row] = ub
                    new_layer_gurobi_vars[row] = v
                gurobi_model.update()
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
            elif type(layer) == nn.MaxPool2d:
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]
                t1_start = time.perf_counter()
                t2_start = time.process_time()
                # Compute convolution
                ch_in, h_in, w_in = old_layer_lb.shape
                k_h = layer.kernel_size
                k_w = layer.kernel_size
                padding = layer.padding  # np.array(layer.padding)
                stride = layer.stride  # np.array(layer.stride)
                h_out = int((h_in + 2 * padding - k_h) / stride + 1)
                w_out = int((w_in + 2 * padding - k_w) / stride + 1)
                # out_channels = np.array(layer.out_channels)
                previous_layer_size = np.array(gurobi_vars[-1].shape)
                out_size = (ch_in, h_out, w_out)
                new_layer_lb = np.zeros(out_size)
                new_layer_ub = np.zeros(out_size)
                new_layer_gurobi_vars = np.ndarray(out_size, dtype=grb.Var)
                for channel in range(ch_in):
                    for row in range(h_out):
                        for col in range(w_out):
                            lin_expr = 0  # layer.bias[channel].item()
                            lower_bounds_section = old_layer_lb[channel, col * stride - padding:col * stride - padding + k_w, row * stride - padding:row * stride - padding + k_h]
                            lb = lower_bounds_section.max()
                            upper_bounds_section = old_layer_ub[channel, col * stride - padding:col * stride - padding + k_w, row * stride - padding:row * stride - padding + k_h]
                            ub = upper_bounds_section.max()
                            v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                                    vtype=grb.GRB.CONTINUOUS,
                                                    name=f'lay{layer_idx}_{channel}_{row}_{col}')
                            gurobi_model.addConstr(v >= lb)
                            gurobi_model.addConstr(v <= ub)
                            gurobi_model.update()
                            gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                            gurobi_model.optimize()
                            assert gurobi_model.status == 2, "LP wasn't optimally solved"
                            # We have computed a lower bound
                            lb = v.X
                            v.lb = lb

                            # Let's now compute an upper bound
                            gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                            gurobi_model.update()
                            gurobi_model.reset()
                            gurobi_model.optimize()
                            assert gurobi_model.status == 2, "LP wasn't optimally solved"
                            ub = v.X
                            v.ub = ub
                            new_layer_lb[channel][row][col] = lb
                            new_layer_ub[channel][row][col] = ub
                            new_layer_gurobi_vars[channel][row][col] = v
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
                t1_stop = time.perf_counter()
                t2_stop = time.process_time()
                # print(f"End MaxPool2d{layer_idx} {((t1_stop - t1_start)):.1f} [sec]")
            elif type(layer) == Flatten:
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]

                new_layer_lb = old_layer_lb.flatten()
                new_layer_ub = old_layer_ub.flatten()
                new_layer_gurobi_vars = np.ndarray(new_layer_ub.shape, grb.Var)  # old_layer_gurobi_vars.flatten()
                index = 0
                for var in old_layer_gurobi_vars.flatten():
                    v = gurobi_model.addVar(lb=var.lb, ub=var.ub, obj=0,
                                            vtype=grb.GRB.CONTINUOUS,
                                            name=f'lay{layer_idx}_{index}')
                    new_layer_gurobi_vars[index] = v
                    index += 1
                gurobi_model.update()
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)

            elif type(layer) == nn.Conv2d:
                # print(f"Start Conv2d_{layer_idx}")
                old_layer_lb = lower_bounds[-1]
                old_layer_ub = upper_bounds[-1]
                old_layer_gurobi_vars = gurobi_vars[-1]
                t1_start = time.perf_counter()
                t2_start = time.process_time()
                # Compute convolution
                ch_in, h_in, w_in = old_layer_lb.shape
                channels_out, channel_in, k_h, k_w = np.array(layer.weight.shape)
                padding = np.array(layer.padding)
                stride = np.array(layer.stride)
                h_out = int((h_in + 2 * padding[0] - k_h) / stride[0] + 1)
                w_out = int((w_in + 2 * padding[1] - k_w) / stride[1] + 1)
                # out_channels = np.array(layer.out_channels)
                previous_layer_size = np.array(gurobi_vars[-1].shape)
                out_size = (channels_out, h_out, w_out)
                new_layer_lb = np.zeros(out_size)
                new_layer_ub = np.zeros(out_size)
                new_layer_gurobi_vars = np.ndarray(out_size, dtype=grb.Var)

                # conv_index = torch.Tensor(range(nb_pre)).view(previous_layer_size[:-1])  # index of the gurobi vars in the flat representation
                for channel in range(channels_out):
                    for row in range(h_out):
                        for col in range(w_out):
                            lb = layer.bias[channel].item()
                            ub = layer.bias[channel].item()
                            lin_expr = layer.bias[channel].item()
                            for kernel_channel in range(channel_in):
                                for i in range(k_h):
                                    for j in range(k_w):
                                        row_prev_layer = row * stride[0] - padding[0] + i
                                        col_prev_layer = col * stride[1] - padding[1] + j
                                        if row_prev_layer < 0 or row_prev_layer > \
                                                old_layer_lb[kernel_channel].shape[0]:
                                            continue
                                        if col_prev_layer < 0 or col_prev_layer > \
                                                old_layer_lb[kernel_channel].shape[1]:
                                            continue
                                        coeff = layer.weight[channel][kernel_channel][i][j].item()
                                        if coeff > 0:
                                            lb += coeff * old_layer_lb[kernel_channel][row_prev_layer][col_prev_layer]
                                            ub += coeff * old_layer_ub[kernel_channel][row_prev_layer][col_prev_layer]
                                        else:  # invert lower bound and upper bound
                                            lb += coeff * old_layer_ub[kernel_channel][row_prev_layer][col_prev_layer]
                                            ub += coeff * old_layer_lb[kernel_channel][row_prev_layer][col_prev_layer]
                                        lin_expr = lin_expr + coeff * gurobi_vars[-1][kernel_channel][row_prev_layer][col_prev_layer]
                            v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                                    vtype=grb.GRB.CONTINUOUS,
                                                    name=f'lay{layer_idx}_{channel}_{row}_{col}')
                            gurobi_model.addConstr(v == lin_expr)
                            gurobi_model.update()
                            gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
                            gurobi_model.optimize()
                            assert gurobi_model.status == 2, "LP wasn't optimally solved"
                            # We have computed a lower bound
                            lb = v.X
                            v.lb = lb

                            # Let's now compute an upper bound
                            gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
                            gurobi_model.update()
                            gurobi_model.reset()
                            gurobi_model.optimize()
                            assert gurobi_model.status == 2, "LP wasn't optimally solved"
                            ub = v.X
                            v.ub = ub
                            new_layer_lb[channel][row][col] = lb
                            new_layer_ub[channel][row][col] = ub
                            new_layer_gurobi_vars[channel][row][col] = v
                lower_bounds.append(new_layer_lb)
                upper_bounds.append(new_layer_ub)
                gurobi_vars.append(new_layer_gurobi_vars)
                t1_stop = time.perf_counter()
                t2_stop = time.process_time()
                # print(f"End Conv2d_{layer_idx} {((t1_stop - t1_start)):.1f} [sec]")
            else:
                raise Exception('Type of layer not supported')
            if save:
                with open(f'./data/{file_name}-{layer_idx}.ub', 'wb') as f_ub:
                    pickle.dump(new_layer_ub, f_ub)
                with open(f'./data/{file_name}-{layer_idx}.lb', 'wb') as f_lb:
                    pickle.dump(new_layer_lb, f_lb)
                gurobi_model.ModelName = f'{file_name}'
                gurobi_model.update()
                gurobi_model.write(f'./data/{file_name}-{layer_idx}.mps')
            layer_idx += 1
        # Assert that this is as expected a network with a single output
        # assert len(gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

        # last layer, minimise
        lower_bound = min(lower_bounds[-1])
        upper_bound = max(upper_bounds[-1])
        assert lower_bound <= upper_bound
        # v = gurobi_model.addVar(lb=lower_bound, ub=upper_bound, obj=0,
        #                         vtype=grb.GRB.CONTINUOUS,
        #                         name=f'lay{layer_idx}_max')
        # # gurobi_model.addConstr(v == min(gurobi_vars[-1]))
        # gurobi_model.addGenConstrMax(v, gurobi_vars[-1], name="maxconstr")
        # gurobi_model.update()
        # #     print(f'v={v}')
        # gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
        # gurobi_model.optimize()
        #
        # gurobi_model.update()
        # gurobi_vars.append([v])

        # We will first setup the appropriate bounds for the elements of the
        # input
        # is it just to be sure?
        # for var_idx, inp_var in enumerate(gurobi_vars[0]):
        #     inp_var.lb = domain[var_idx, 0]
        #     inp_var.ub = domain[var_idx, 1]

        # We will make sure that the objective function is properly set up
        # gurobi_model.setObjective(gurobi_vars[-1][0], grb.GRB.MAXIMIZE)
        # print(f'gurobi_vars[-1][0].size()={len(gurobi_vars[-1])}')
        # We will now compute the requested lower bound
        # gurobi_model.update()
        # gurobi_model.optimize()
        # assert gurobi_model.status == 2, "LP wasn't optimally solved"
        # print(f'gurobi status {gurobi_model.status}')
        # print(f'Result={gurobi_vars[-1][0].X}')
        # print(f'Result={gurobi_vars[-1]}')
        # print(f'Result -1={gurobi_vars[-2]}')
        return upper_bound,lower_bound

    def convert_ConvL_to_FCL(self, input: torch.Tensor, K: torch.Tensor, padding, stride):
        batch_size, channels_in, h_in, w_in = input.size()
        channels_out, channels_in, k_h, k_w = K.size()
        h_out = int((h_in + 2 * padding - k_h) / stride + 1)
        w_out = int((w_in + 2 * padding - k_w) / stride + 1)
        padding = stride * (h_out - 1) + k_h - h_in
        plefttop = int((padding - 1) / 2) if padding > 0 else 0
        prightbot = padding - plefttop
        padedinput = np.lib.pad(input.cpu(), ((0, 0), (plefttop, prightbot), (plefttop, prightbot), (0, 0)), 'constant',
                                constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
        # M = torch.tensor(np.ndarray(shape=(batch_size * h_out * w_out, k_h * k_w * channels_in)))
        stretchinput = np.zeros(shape=(batch_size * h_out * w_out, k_h * k_w * channels_in), dtype=np.float64)
        # L = torch.tensor(np.ndarray(shape=(k_h * k_w * channels_in, channels_out)))
        for j in range(stretchinput.shape[0]):
            batch_index = int(j / (h_out * w_out))
            patch_index = j % (h_out * w_out)
            ih2 = patch_index % w_out
            iw2 = int(patch_index / w_out)
            sih1 = iw2 * stride
            siw1 = ih2 * stride
            stretchinput[j, :] = padedinput[batch_index, :, sih1:sih1 + k_h, siw1:siw1 + k_w].flatten()
        return torch.tensor(stretchinput), (batch_size, channels_out, h_out, w_out)

    def recoverInput(self, input, kernel_size, stride, outshape):
        '''
        :param input: it is of the shape (height, width)
        :param kernel_size: it is the kernel shape we want
        :param stride:
        :param outshape: the shape of the output
        :return:
        '''
        H, W = input.shape
        batch, h, w, ch = outshape
        original_input = np.zeros(outshape)
        first_row_index = np.arange(0, w, kernel_size)
        first_col_index = np.arange(0, h, kernel_size)

        patches_row = int((w - kernel_size) / stride) + 1
        # patches_col = (h-kernel_size)/stride + 11_2
        rowend_index = kernel_size - (w - first_row_index[-1])
        colend_index = kernel_size - (h - first_col_index[-1])
        if first_row_index[-1] + kernel_size > w:
            first_row_index[-1] = first_row_index[-1] - (first_row_index[-1] + kernel_size - 1 - (w - 1))
        if first_col_index[-1] + kernel_size > h:
            first_col_index[-1] = first_col_index[-1] - (first_col_index[-1] + kernel_size - 1 - (h - 1))

        for k in range(batch):
            for i in range(len(first_col_index)):
                for j in range(len(first_row_index)):
                    w_index = first_row_index[j] + i * patches_row + \
                              k * (int((h - kernel_size) / stride) + 1) * (int((w - kernel_size) / stride) + 1)
                    # print('------------------------')
                    if i != len(first_col_index) - 1 and j != len(first_row_index) - 1:
                        # print( original_input[  k , first_row_index[j] : first_row_index[j]+kernel_size ,
                        #     first_col_index[i] :  first_col_index[i]+kernel_size ,:].shape)
                        # print(input[w_index,:].reshape(kernel_size,kernel_size,-11_2).shape)
                        original_input[k, first_row_index[j]: first_row_index[j] + kernel_size,
                        first_col_index[i]:  first_col_index[i] + kernel_size, :] \
                            = input[w_index, :].reshape(kernel_size, kernel_size, -1)
                    elif i == len(first_col_index) - 1 and j != len(first_row_index) - 1:
                        # print(original_input[k, first_col_index[-11_2] + colend_index : ,
                        #     first_row_index[i] :  first_row_index[i]+kernel_size, :].shape)
                        # print(input[w_index, :].reshape(kernel_size, kernel_size, -11_2)[rowend_index:,:,:].shape)
                        original_input[k, first_col_index[-1] + colend_index:,
                        first_row_index[i]:  first_row_index[i] + kernel_size, :] \
                            = input[w_index, :].reshape(kernel_size, kernel_size, -1)[rowend_index:, :, :]
                    elif i != len(first_col_index) - 1 and j == len(first_row_index) - 1:
                        # print(original_input[k, first_col_index[i]: first_col_index[i]+kernel_size, first_row_index[-11_2]+rowend_index : ,:].shape)
                        # print(input[w_index, :].reshape(kernel_size, kernel_size, -11_2)[:, colend_index : , :].shape)
                        original_input[k, first_col_index[i]: first_col_index[i] + kernel_size,
                        first_row_index[-1] + rowend_index:, :] \
                            = input[w_index, :].reshape(kernel_size, kernel_size, -1)[:, colend_index:, :]
                    else:
                        # print( original_input[k,first_col_index[-11_2] + colend_index : ,
                        #     first_row_index[-11_2] + rowend_index:, :].shape)
                        # print(input[w_index, :].reshape(kernel_size, kernel_size, -11_2)[
                        #       rowend_index :, colend_index :, :].shape)
                        original_input[k, first_col_index[-1] + colend_index:,
                        first_row_index[-1] + rowend_index:, :] \
                            = input[w_index, :].reshape(kernel_size, kernel_size, -1)[
                              rowend_index:, colend_index:, :]
        return original_input

    def stretchKernel(self, kernel: torch.Tensor):
        '''
        :param kernel: it has the shape (filters, height, width, channels) denoted as (filter_num, kh, kw, ch)
        :return: kernel of the shape (kh*kw*ch,filter_num)
        '''
        filter_num, ch, kh, kw = kernel.shape
        stretchkernel = np.zeros((kh * kw * ch, filter_num), dtype=np.float64)
        for i in range(filter_num):
            stretchkernel[:, i] = kernel[i, :, :, :].cpu().detach().numpy().flatten()
        return torch.tensor(stretchkernel)


def linear_layer(self, gurobi_model, gurobi_vars, weight, bias, layer_idx, lower_bounds, new_layer_gurobi_vars,
                 new_layer_lb, new_layer_ub, upper_bounds):
    for neuron_idx in range(weight.size(0)):
        if bias is None:
            ub = 0
            lb = 0
            lin_expr = 0
        else:
            ub = bias.data[neuron_idx]
            lb = bias.data[neuron_idx]
            lin_expr = bias.data[neuron_idx].item()  # adds the bias to the linear expression
        #     print(f'bias_ub={ub} bias_lb={lb}')

        for prev_neuron_idx in range(weight.size(1)):
            coeff = weight.data[neuron_idx, prev_neuron_idx]  # picks the weight between the two neurons
            if coeff >= 0:
                ub = ub + coeff * upper_bounds[-1][prev_neuron_idx]  # multiplies the ub
                lb = lb + coeff * lower_bounds[-1][prev_neuron_idx]  # multiplies the lb
            else:  # inverted
                ub = ub + coeff * lower_bounds[-1][prev_neuron_idx]  # multiplies the ub
                lb = lb + coeff * upper_bounds[-1][prev_neuron_idx]  # multiplies the lb
            #         print(f'ub={ub} lb={lb}')
            #                     assert ub!=lb
            lin_expr = lin_expr + coeff.item() * gurobi_vars[-1][
                prev_neuron_idx]  # multiplies the unknown by the coefficient
        #         print(lin_expr)
        v = gurobi_model.addVar(lb=lb, ub=ub, obj=0,
                                vtype=grb.GRB.CONTINUOUS,
                                name=f'lay{layer_idx}_{neuron_idx}')
        gurobi_model.addConstr(v == lin_expr)
        gurobi_model.update()
        #     print(f'v={v}')
        gurobi_model.setObjective(v, grb.GRB.MINIMIZE)
        gurobi_model.optimize()
        #          print(f'gurobi status {gurobi_model.status}')
        assert gurobi_model.status == 2, "LP wasn't optimally solved"
        # We have computed a lower bound
        lb = v.X
        v.lb = lb

        # Let's now compute an upper bound
        gurobi_model.setObjective(v, grb.GRB.MAXIMIZE)
        gurobi_model.update()
        gurobi_model.reset()
        gurobi_model.optimize()
        assert gurobi_model.status == 2, "LP wasn't optimally solved"
        ub = v.X
        v.ub = ub

        new_layer_lb.append(lb)
        new_layer_ub.append(ub)
        new_layer_gurobi_vars.append(v)

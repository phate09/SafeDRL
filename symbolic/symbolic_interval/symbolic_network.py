'''
Interval networks and symbolic interval propagations.
** Top contributor: Shiqi Wang
** This file is part of the symbolic interval analysis library.
** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
** and their institutional affiliations.
** All rights reserved.

Usage: 
for symbolic interval anlysis:
	from symbolic_interval.symbolic_network import sym_interval_analyze
for naive interval analysis:
	from symbolic_interval.symbolic_network import naive_interval_analyze
'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interval import Interval, Symbolic_interval, mix_interval, Inverse_interval, Center_symbolic_interval
from .interval import Symbolic_interval_proj1, Symbolic_interval_proj2


class Interval_network(nn.Module):
    '''Convert a nn.Sequential model to a network support symbolic
    interval propagations/naive interval propagations.
    '''

    def __init__(self, model, c):
        nn.Module.__init__(self)

        self.net = []
        first_layer = True
        last_layer = False

        for layer in model:
            if (isinstance(layer, nn.Linear)):
                if layer == model[-1]:
                    last_layer = True
                if last_layer and c is not None:
                    wc_matrix = c
                else:
                    wc_matrix = None
                self.net.append(Interval_Dense(layer, first_layer, wc_matrix=wc_matrix))
                first_layer = False
            if (isinstance(layer, nn.ReLU)):
                self.net.append(Interval_ReLU(layer))
            if (isinstance(layer, nn.Conv2d)):
                self.net.append(Interval_Conv2d(layer, first_layer))
                first_layer = False
            if (isinstance(layer, nn.Softmax)):
                self.net.append(Interval_Softmax(layer))
                first_layer = False
            if 'Flatten' in (str(layer.__class__.__name__)):
                self.net.append(Interval_Flatten())
            if 'bn' in (str(layer.__class__.__name__)):
                self.net.append(Interval_BN(layer))
        self.net = nn.Sequential(*self.net)

    '''Forward intervals for each layer.

    * :attr:`ix` is the input fore each layer. If ix is a naive
    interval, it will propagate naively. If ix is a symbolic 
    interval, it will propagate symbolicly.
    '''

    def forward(self, ix):
        return self.net(ix)
        '''
        for i, layer in enumerate(self.net):
            ix = layer(ix)
        return ix
        '''


class Interval_Dense(nn.Module):
    def __init__(self, layer, first_layer=False, wc_matrix=None):
        nn.Module.__init__(self)
        self.layer = layer
        self.first_layer = first_layer
        self.wc_matrix = wc_matrix

    def forward(self, ix):
        if (isinstance(ix, Center_symbolic_interval)):
            c = ix.c
            idep = ix.idep
            # print (ix.c.shape, self.layer.weight.shape)
            ix.c = F.linear(c, self.layer.weight, bias=self.layer.bias)
            ix.idep = F.linear(idep, self.layer.weight)
            ix.shape = list(ix.c.shape[1:])
            ix.n = list(ix.c[0].view(-1).size())[0]
            ix.concretize()
            return ix

        if (isinstance(ix, mix_interval)):
            # print (ix.c.shape, self.layer.weight.shape)
            ix.c = F.linear(ix.c, self.layer.weight, bias=self.layer.bias)
            ix.idep = F.linear(ix.idep, self.layer.weight)
            for i in range(len(ix.edep)):
                ix.edep[i] = F.linear(ix.edep[i], self.layer.weight)
            ix.shape = list(ix.c.shape[1:])
            ix.n = list(ix.c[0].view(-1).size())[0]

            c = ix.nc
            e = ix.ne
            if self.wc_matrix is None:
                c = F.linear(c, self.layer.weight, bias=self.layer.bias)
                e = F.linear(e, self.layer.weight.abs())
                ix.nc, ix.ne, ix.nl, ix.nu = c, e, c - e, c + e
            else:
                weight = self.wc_matrix.matmul(self.layer.weight)
                bias = self.wc_matrix.matmul(self.layer.bias)

                c = weight.matmul(c.unsqueeze(-1)) + bias.unsqueeze(-1)
                e = weight.abs().matmul(e.unsqueeze(-1))

                c, e = c.squeeze(-1), e.squeeze(-1)

                ix.nc, ix.ne, ix.nl, ix.nu = -c, -e, -c - e, -c + e

            ix.concretize()

            return ix

        if (isinstance(ix, Symbolic_interval)):  # normally we use this layer
            # print (ix.c.shape, self.layer.weight.shape)
            ix.c = F.linear(ix.c, self.layer.weight, bias=self.layer.bias)
            ix.idep = F.linear(ix.idep, self.layer.weight)
            for i in range(len(ix.edep)):
                ix.edep[i] = F.linear(ix.edep[i], self.layer.weight)
            ix.shape = list(ix.c.shape[1:])
            ix.n = list(ix.c[0].view(-1).size())[0]
            ix.concretize()
            return ix

        if (isinstance(ix, Symbolic_interval_proj1)):
            c = ix.c
            idep = ix.idep
            edep = ix.edep

            ix.c = F.linear(c, self.layer.weight, bias=self.layer.bias)
            ix.idep = F.linear(idep, self.layer.weight)
            ix.idep_proj = F.linear(ix.idep_proj, self.layer.weight.abs())

            for i in range(len(edep)):
                ix.edep[i] = F.linear(edep[i], self.layer.weight)
            ix.shape = list(ix.c.shape[1:])
            ix.n = list(ix.c[0].view(-1).size())[0]
            ix.concretize()
            return ix

        if (isinstance(ix, Symbolic_interval_proj2)):
            c = ix.c
            idep = ix.idep
            edep = ix.edep

            ix.c = F.linear(c, self.layer.weight, bias=self.layer.bias)
            ix.idep = F.linear(idep, self.layer.weight)
            ix.idep_proj = F.linear(ix.idep_proj, self.layer.weight.abs())

            ix.edep = F.linear(ix.edep, self.layer.weight.abs())

            ix.shape = list(ix.c.shape[1:])
            ix.n = list(ix.c[0].view(-1).size())[0]
            ix.concretize()
            return ix

        if (isinstance(ix, Inverse_interval)):
            c = ix.c
            e = ix.e
            c = F.linear(c, self.layer.weight, bias=self.layer.bias)
            e = F.linear(e, self.layer.weight)

            # print("naive e", e)
            # print("naive c", c)
            ix.update_lu(c - e, c + e)

            return ix

        if (isinstance(ix, Interval)):
            c = ix.c
            e = ix.e
            if self.wc_matrix is None:
                c = F.linear(c, self.layer.weight, bias=self.layer.bias)
                e = F.linear(e, self.layer.weight.abs())
            else:
                weight = self.wc_matrix.matmul(self.layer.weight)
                bias = self.wc_matrix.matmul(self.layer.bias)

                c = weight.matmul(c.unsqueeze(-1)) + bias.unsqueeze(-1)
                e = weight.abs().matmul(e.unsqueeze(-1))

                c, e = c.squeeze(-1), e.squeeze(-1)

            # print(c.shape, e.shape)
            # print("naive e", e)
            # print("naive c", c)
            ix.update_lu(c - e, c + e)

            return ix


class Interval_Conv2d(nn.Module):
    def __init__(self, layer, first_layer=False):
        nn.Module.__init__(self)
        self.layer = layer
        self.first_layer = first_layer

    # print ("conv2d:", self.layer.weight.shape)

    def forward(self, ix):
        if (isinstance(ix, Center_symbolic_interval)):
            ix.shrink()
            c = ix.c
            idep = ix.idep
            ix.c = F.conv2d(c, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding, bias=self.layer.bias)
            ix.idep = F.conv2d(idep, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding)

            ix.shape = list(ix.c.shape[1:])
            ix.n = list(ix.c[0].reshape(-1).size())[0]
            ix.concretize()
            return ix

        if (isinstance(ix, mix_interval)):
            ix.shrink()
            ix.c = F.conv2d(ix.c, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding, bias=self.layer.bias)
            ix.idep = F.conv2d(ix.idep, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding)

            for i in range(len(ix.edep)):
                ix.edep[i] = F.conv2d(ix.edep[i], self.layer.weight, stride=self.layer.stride, padding=self.layer.padding)
            ix.shape = list(ix.c.shape[1:])
            ix.n = list(ix.c[0].reshape(-1).size())[0]

            c, e = ix.nc, ix.ne

            c = F.conv2d(c, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding, bias=self.layer.bias)
            e = F.conv2d(e, self.layer.weight.abs(), stride=self.layer.stride, padding=self.layer.padding)

            ix.nc, ix.ne, ix.nl, ix.nu = c, e, c - e, c + e

            ix.concretize()

            return ix

        if (isinstance(ix, Symbolic_interval)):
            ix.shrink()
            ix.c = F.conv2d(ix.c, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding, bias=self.layer.bias)
            ix.idep = F.conv2d(ix.idep, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding)

            for i in range(len(ix.edep)):
                ix.edep[i] = F.conv2d(ix.edep[i], self.layer.weight, stride=self.layer.stride, padding=self.layer.padding)
            ix.shape = list(ix.c.shape[1:])
            ix.n = list(ix.c[0].reshape(-1).size())[0]
            ix.concretize()
            return ix

        if (isinstance(ix, Symbolic_interval_proj1)):
            # print(ix.idep.shape)
            ix.shrink()
            c = ix.c
            idep = ix.idep
            edep = ix.edep
            ix.c = F.conv2d(c, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding, bias=self.layer.bias)
            ix.idep = F.conv2d(idep, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding)
            ix.idep_proj = F.conv2d(ix.idep_proj, self.layer.weight.abs(), stride=self.layer.stride, padding=self.layer.padding)

            for i in range(len(edep)):
                ix.edep[i] = F.conv2d(edep[i], self.layer.weight, stride=self.layer.stride, padding=self.layer.padding)
            ix.shape = list(ix.c.shape[1:])
            ix.n = list(ix.c[0].reshape(-1).size())[0]
            ix.concretize()
            return ix

        if (isinstance(ix, Symbolic_interval_proj2)):
            # print(ix.idep.shape)
            ix.shrink()
            c = ix.c
            idep = ix.idep
            edep = ix.edep
            ix.c = F.conv2d(c, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding, bias=self.layer.bias)
            ix.idep = F.conv2d(idep, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding)
            ix.idep_proj = F.conv2d(ix.idep_proj, self.layer.weight.abs(), stride=self.layer.stride, padding=self.layer.padding)

            ix.edep = F.conv2d(ix.edep, self.layer.weight.abs(), stride=self.layer.stride, padding=self.layer.padding)

            ix.shape = list(ix.c.shape[1:])
            ix.n = list(ix.c[0].reshape(-1).size())[0]
            ix.concretize()
            return ix

        if (isinstance(ix, Inverse_interval)):
            c = ix.c
            e = ix.e
            c = F.conv2d(c, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding, bias=self.layer.bias)
            e = F.conv2d(e, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding)
            ix.update_lu(c - e, c + e)

            return ix

        if (isinstance(ix, Interval)):
            c = ix.c
            e = ix.e
            c = F.conv2d(c, self.layer.weight, stride=self.layer.stride, padding=self.layer.padding, bias=self.layer.bias)
            e = F.conv2d(e, self.layer.weight.abs(), stride=self.layer.stride, padding=self.layer.padding)
            ix.update_lu(c - e, c + e)

            return ix


class Interval_BN(nn.Module):
    def __init__(self, layer, first_layer=False):
        nn.Module.__init__(self)
        self.layer = layer
        self.first_layer = first_layer

    def forward(self, ix):
        if isinstance(ix, Interval):
            shape = ix.u.shape
            tmax = torch.where(ix.u > -ix.l, ix.u, ix.l).view(ix.batch_size, -1)
            # tmax = ix.c.view(ix.batch_size, -1)
            mean = tmax.mean(dim=0, keepdim=True)
            # print(mean.shape, tmax.shape, ix.u.shape)

            sigma = torch.norm(tmax - mean, dim=0, keepdim=True)
            # sigma = sigma*sigma

            # if self.layer.mean is None:
            # 	self.layer.mean = mean
            # 	self.layer.sigma = sigma
            # else:

            # self.layer.mean = self.layer.mean * (1-self.layer.momentum) + self.layer.momentum * mean
            # self.layer.sigma = self.layer.sigma * (1-self.layer.momentum) + self.layer.momentum * sigma

            self.layer.mean = mean
            self.layer.sigma = sigma
            # print(mean, sigma)

            ix.u = (ix.u.view(ix.batch_size, -1) - mean) / sigma
            ix.l = (ix.l.view(ix.batch_size, -1) - mean) / sigma
            ix.u, ix.l = ix.u.view(shape), ix.l.view(shape)
            return ix


class Interval_ReLU(nn.Module):
    def __init__(self, layer):
        nn.Module.__init__(self)
        self.layer = layer

    def forward(self, ix):
        # print(ix.u)
        # print(ix.l)
        if (isinstance(ix, Center_symbolic_interval)):
            lower = ix.l
            upper = ix.u
            # print("sym u", upper)
            # print("sym l", lower)

            appr_condition = ((lower < 0) * (upper > 0)).detach()
            mask = (lower > 0).type_as(lower)
            mask[appr_condition] = upper[appr_condition] / (upper[appr_condition] - lower[appr_condition]).detach()

            new_mask = (ix.c >= 0).type_as(ix.c)

            ix.c = ix.c * mask
            ix.idep = ix.idep * new_mask.view(ix.batch_size, 1, ix.n)

            return ix

        if (isinstance(ix, mix_interval)):

            lower = ix.l.clamp(max=0)
            upper = ix.u.clamp(min=0)
            upper = torch.max(upper, lower + 1e-8)
            mask = upper / (upper - lower)
            appr_condition = ((ix.l < 0) * (ix.u > 0))

            m = int(appr_condition.sum().item())

            appr_ind = appr_condition.view(-1, ix.n).nonzero()

            appr_err = mask * (-lower) / 2.0

            if (m != 0):

                if (ix.use_cuda):
                    error_row = torch.zeros((m, ix.n), device=lower.get_device())
                else:
                    error_row = torch.zeros((m, ix.n))

                error_row = error_row.scatter_(1, appr_ind[:, 1, None], appr_err[appr_condition][:, None])

                edep_ind = lower.new(appr_ind.size(0), lower.size(0)).zero_()
                edep_ind = edep_ind.scatter_(1, appr_ind[:, 0][:, None], 1)

            ix.c = ix.c * mask + appr_err * appr_condition.type_as(lower)

            for i in range(len(ix.edep)):
                ix.edep[i] = ix.edep[i] * ix.edep_ind[i].mm(mask)

            ix.idep = ix.idep * mask.view(ix.batch_size, 1, ix.n)

            if (m != 0):
                ix.edep = ix.edep + [error_row]
                ix.edep_ind = ix.edep_ind + [edep_ind]

            ix.nl, ix.nu = F.relu(ix.nl), F.relu(ix.nu)
            ix.nc, ix.ne = (ix.nl + ix.nu) / 2., (ix.nu - ix.nl) / 2.
            return ix

        if (isinstance(ix, Symbolic_interval)):

            lower = ix.l.clamp(max=0)
            upper = ix.u.clamp(min=0)
            upper = torch.max(upper, lower + 1e-8)
            mask = upper / (upper - lower)
            appr_condition = ((ix.l < 0) * (ix.u > 0))  # vector that describes if both the lower bound is <0 and the upper bound is >0

            # ix.l.retain_grad()
            # mask[0,0].backward()
            # print(ix.l.grad)

            m = int(appr_condition.sum().item())

            appr_ind = appr_condition.view(-1, ix.n).nonzero()

            appr_err = mask * (-lower) / 2.0

            if (m != 0):

                if (ix.use_cuda):
                    error_row = torch.zeros((m, ix.n), dtype=ix.c.dtype, device=lower.get_device())
                else:
                    error_row = torch.zeros((m, ix.n), dtype=ix.c.dtype)

                error_row = error_row.scatter_(1, appr_ind[:, 1, None], appr_err[appr_condition][:, None])

                edep_ind = lower.new(appr_ind.size(0), lower.size(0)).zero_()
                edep_ind = edep_ind.scatter_(1, appr_ind[:, 0][:, None], 1)

            ix.c = ix.c * mask + appr_err * appr_condition.type_as(lower)

            for i in range(len(ix.edep)):
                ix.edep[i] = ix.edep[i] * ix.edep_ind[i].mm(mask)

            ix.idep = ix.idep * mask.view(ix.batch_size, 1, ix.n)

            if (m != 0):
                ix.edep = ix.edep + [error_row]
                ix.edep_ind = ix.edep_ind + [edep_ind]
            return ix

        if (isinstance(ix, Symbolic_interval_proj1)):
            lower = ix.l
            upper = ix.u
            # print("sym u", upper)
            # print("sym l", lower)

            appr_condition = ((lower < 0) * (upper > 0)).detach()
            mask = (lower > 0).type_as(lower)
            mask[appr_condition] = upper[appr_condition] / (upper[appr_condition] - lower[appr_condition]).detach()

            m = int(appr_condition.sum().item())
            appr_ind = appr_condition.view(-1, ix.n).nonzero()

            appr_err = mask * (-lower) / 2.0

            if (m != 0):

                if (ix.use_cuda):
                    error_row = torch.zeros((m, ix.n), device=lower.get_device())
                else:
                    error_row = torch.zeros((m, ix.n))

                error_row = error_row.scatter_(1, appr_ind[:, 1, None], appr_err[appr_condition][:, None])

                edep_ind = lower.new(appr_ind.size(0), lower.size(0)).zero_()
                edep_ind = edep_ind.scatter_(1, appr_ind[:, 0][:, None], 1)

            ix.c = ix.c * mask + appr_err * appr_condition.type_as(lower)

            for i in range(len(ix.edep)):
                ix.edep[i] = ix.edep[i] * ix.edep_ind[i].mm(mask)

            ix.idep = ix.idep * mask.view(ix.batch_size, 1, ix.n)
            ix.idep_proj = ix.idep_proj * mask.view(ix.batch_size, ix.n)

            if (m != 0):
                ix.edep = ix.edep + [error_row]
                ix.edep_ind = ix.edep_ind + [edep_ind]

            return ix

        if (isinstance(ix, Symbolic_interval_proj2)):
            lower = ix.l
            upper = ix.u
            # print("sym u", upper)
            # print("sym l", lower)
            appr_condition = ((lower < 0) * (upper > 0)).detach()
            mask = (lower > 0).type_as(lower)
            m = int(appr_condition.sum().item())
            # appr_ind = appr_condition.view(-1,ix.n).nonzero()

            appr_err = upper / 2.0
            appr_err = appr_err * (appr_condition.type_as(appr_err))

            ix.edep = ix.edep * mask
            ix.edep = ix.edep + appr_err

            ix.c = ix.c * mask + appr_err

            ix.idep = ix.idep * mask.view(ix.batch_size, 1, ix.n)
            ix.idep_proj = ix.idep_proj * mask.view(ix.batch_size, ix.n)

            return ix

        if (isinstance(ix, Inverse_interval)):
            lower = ix.l
            upper = ix.u

            # print("naive", upper)

            if (ix.use_cuda):
                appr_condition = ((lower < 0) * (upper > 0)).type(torch.Tensor).cuda(device=ix.c.get_device())
            else:
                appr_condition = ((lower < 0) * (upper > 0)).type(torch.Tensor)

            mask = appr_condition * ((upper) / (upper - lower + 0.000001))
            mask = mask + 1 - appr_condition
            if (ix.use_cuda):
                mask = mask * ((upper > 0).type(torch.Tensor).cuda(device=ix.c.get_device()))
            else:
                mask = mask * (upper > 0).type(torch.Tensor)
            ix.mask.append(mask)
            # print(ix.e.shape)
            ix.update_lu(F.relu(ix.l), F.relu(ix.u))
            return ix

        if (isinstance(ix, Interval)):
            '''
            lower = ix.l.clamp(max=0)
            upper = ix.u.clamp(min=0)
            upper = torch.max(upper, lower + 1e-8)
            mask = upper / (upper - lower)

            ix.mask.append(mask)
            '''
            ix.update_lu(F.relu(ix.l), F.relu(ix.u))
            return ix


class Interval_Softmax(nn.Module):
    def __init__(self, layer):
        nn.Module.__init__(self)
        self.layer = layer

    def forward(self, ix):
        # print(ix.u)
        # print(ix.l)
        if (isinstance(ix, Symbolic_interval)):
            lower = ix.u.detach().clone().unsqueeze(1).repeat((1, ix.input_size, 1))
            upper = ix.l.detach().clone().unsqueeze(1).repeat((1, ix.input_size, 1))
            lower = lower - lower * torch.eye(ix.input_size)  # remove elements across diagonal
            lower = lower + ix.l.detach().clone().unsqueeze(1).repeat((1, ix.input_size, 1)) * torch.eye(ix.input_size)  # for each element add worst case
            upper = upper - upper * torch.eye(ix.input_size)  # remove elements across diagonal
            upper = upper + ix.u.detach().clone().unsqueeze(1).repeat((1, ix.input_size, 1)) * torch.eye(ix.input_size)
            lower_softmax = torch.nn.functional.softmax(lower, 2)
            upper_softmax = torch.nn.functional.softmax(upper, 2)
            lower_masked = lower_softmax * torch.eye(ix.input_size)
            upper_masked = upper_softmax * torch.eye(ix.input_size)
            lower_result = torch.sum(lower_masked, dim=1)
            upper_result = torch.sum(upper_masked, dim=1)
            ix.update_lu(lower_result, upper_result)
            return ix

        if (isinstance(ix, Symbolic_interval_proj1)):
            raise NotImplementedError
            lower = ix.l
            upper = ix.u
            # print("sym u", upper)
            # print("sym l", lower)

            appr_condition = ((lower < 0) * (upper > 0)).detach()
            mask = (lower > 0).type_as(lower)
            mask[appr_condition] = upper[appr_condition] / (upper[appr_condition] - lower[appr_condition]).detach()

            m = int(appr_condition.sum().item())
            appr_ind = appr_condition.view(-1, ix.n).nonzero()

            appr_err = mask * (-lower) / 2.0

            if (m != 0):

                if (ix.use_cuda):
                    error_row = torch.zeros((m, ix.n), device=lower.get_device())
                else:
                    error_row = torch.zeros((m, ix.n))

                error_row = error_row.scatter_(1, appr_ind[:, 1, None], appr_err[appr_condition][:, None])

                edep_ind = lower.new(appr_ind.size(0), lower.size(0)).zero_()
                edep_ind = edep_ind.scatter_(1, appr_ind[:, 0][:, None], 1)

            ix.c = ix.c * mask + appr_err * appr_condition.type_as(lower)

            for i in range(len(ix.edep)):
                ix.edep[i] = ix.edep[i] * ix.edep_ind[i].mm(mask)

            ix.idep = ix.idep * mask.view(ix.batch_size, 1, ix.n)
            ix.idep_proj = ix.idep_proj * mask.view(ix.batch_size, ix.n)

            if (m != 0):
                ix.edep = ix.edep + [error_row]
                ix.edep_ind = ix.edep_ind + [edep_ind]

            return ix

        if (isinstance(ix, Symbolic_interval_proj2)):
            raise NotImplementedError
            lower = ix.l
            upper = ix.u
            # print("sym u", upper)
            # print("sym l", lower)
            appr_condition = ((lower < 0) * (upper > 0)).detach()
            mask = (lower > 0).type_as(lower)
            m = int(appr_condition.sum().item())
            # appr_ind = appr_condition.view(-1,ix.n).nonzero()

            appr_err = upper / 2.0
            appr_err = appr_err * (appr_condition.type_as(appr_err))

            ix.edep = ix.edep * mask
            ix.edep = ix.edep + appr_err

            ix.c = ix.c * mask + appr_err

            ix.idep = ix.idep * mask.view(ix.batch_size, 1, ix.n)
            ix.idep_proj = ix.idep_proj * mask.view(ix.batch_size, ix.n)

            return ix

        if (isinstance(ix, Inverse_interval)):
            raise NotImplementedError
            lower = ix.l
            upper = ix.u

            # print("naive", upper)

            if (ix.use_cuda):
                appr_condition = ((lower < 0) * (upper > 0)).type(torch.Tensor).cuda(device=ix.c.get_device())
            else:
                appr_condition = ((lower < 0) * (upper > 0)).type(torch.Tensor)

            mask = appr_condition * ((upper) / (upper - lower + 0.000001))
            mask = mask + 1 - appr_condition
            if (ix.use_cuda):
                mask = mask * ((upper > 0).type(torch.Tensor).cuda(device=ix.c.get_device()))
            else:
                mask = mask * (upper > 0).type(torch.Tensor)
            ix.mask.append(mask)
            # print(ix.e.shape)
            ix.update_lu(F.relu(ix.l), F.relu(ix.u))
            return ix

        if (isinstance(ix, Interval)):
            '''
            lower = ix.l.clamp(max=0)
            upper = ix.u.clamp(min=0)
            upper = torch.max(upper, lower + 1e-8)
            mask = upper / (upper - lower)

            ix.mask.append(mask)
            '''
            lower = ix.u.detach().clone().unsqueeze(1).repeat((1, ix.input_size, 1))
            upper = ix.l.detach().clone().unsqueeze(1).repeat((1, ix.input_size, 1))
            lower = lower - lower * torch.eye(ix.input_size)  # remove elements across diagonal
            lower = lower + ix.l.detach().clone() * torch.eye(ix.input_size)  # for each element add worst case
            upper = upper - upper * torch.eye(ix.input_size)  # remove elements across diagonal
            upper = upper + ix.u.detach().clone() * torch.eye(ix.input_size)
            lower_softmax = torch.nn.functional.softmax(lower, 2)
            upper_softmax = torch.nn.functional.softmax(upper, 2)
            lower_masked = lower_softmax * torch.eye(ix.input_size)
            upper_masked = upper_softmax * torch.eye(ix.input_size)
            lower_result = torch.sum(lower_masked, dim=1)
            upper_result = torch.sum(upper_masked, dim=1)
            ix.update_lu(lower_result, upper_result)
            return ix


class Interval_Flatten(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, ix):
        if (isinstance(ix, Symbolic_interval) or isinstance(ix, Symbolic_interval_proj1) or isinstance(ix, Symbolic_interval_proj2) or isinstance(ix, Center_symbolic_interval)):
            ix.extend()
            return ix
        if (isinstance(ix, Interval) or isinstance(ix, Inverse_interval)):
            ix.update_lu(ix.l.view(ix.l.size(0), -1), ix.u.view(ix.u.size(0), -1))
            return ix


class Interval_Bound(nn.Module):
    def __init__(self, net, epsilon, method="sym", use_cuda=True, norm="linf", worst_case=True):
        nn.Module.__init__(self)
        self.net = net
        self.epsilon = epsilon
        self.use_cuda = use_cuda
        assert method in ["sym", "naive", "inverse", "center_sym", "new", "mix"], "No such interval methods!"
        self.method = method
        self.norm = norm
        # assert self.norm in ["linf", "l2", "l1"], "norm" + norm + "not supported"

        self.worst_case = worst_case

    def forward(self, X, y):

        out_features = self.net[-1].out_features

        if self.worst_case:
            c = torch.eye(out_features).type_as(X)[y].unsqueeze(1) - torch.eye(out_features).type_as(X).unsqueeze(0)
        else:
            c = None

        # Transfer original model to interval models
        inet = Interval_network(self.net, c)

        minimum = (X - self.epsilon).min().item()
        maximum = (X + self.epsilon).max().item()

        # Create symbolic inteval classes from X
        if (self.method == "naive"):
            ix = Interval(torch.clamp(X - self.epsilon, minimum, maximum), torch.clamp(X + self.epsilon, minimum, maximum), self.use_cuda)
        if (self.method == "inverse"):
            ix = Inverse_interval(torch.clamp(X - self.epsilon, minimum, maximum), torch.clamp(X + self.epsilon, minimum, maximum), self.use_cuda)
        if (self.method == "center_sym"):
            ix = Center_symbolic_interval(torch.clamp(X - self.epsilon, minimum, maximum), torch.clamp(X + self.epsilon, minimum, maximum), self.use_cuda)

        if self.method == "mix":
            assert self.norm == "linf", "only support linf for now"
            ix = mix_interval(torch.clamp(X - self.epsilon, minimum, maximum), torch.clamp(X + self.epsilon, minimum, maximum), use_cuda=self.use_cuda)

        if (self.method == "sym"):
            if self.norm == "linf":
                ix = Symbolic_interval(torch.clamp(X - self.epsilon, minimum, maximum), torch.clamp(X + self.epsilon, minimum, maximum), use_cuda=self.use_cuda)
            elif self.norm == "l2":
                ix = Symbolic_interval(X, X, self.epsilon, norm="l2", use_cuda=self.use_cuda)
            elif self.norm == "l1":
                ix = Symbolic_interval(X, X, self.epsilon, norm="l1", use_cuda=self.use_cuda)

        # Propagate symbolic interval through interval networks
        ix = inet(ix)
        # print(ix.u)
        # print(ix.l)

        # Calculate the worst case outputs
        if self.method != "naive":
            wc = ix.worst_case(y, out_features)
            return wc

        return -ix.l


'''Naive interval propagations.

Args:
	model: regular nn.Sequential models
	epsilon: desired input ranges
	X and y: samples and lables

Return:
	iloss: robust loss provided by naive interval analysis
	ierr: verifiable robust error provided by naive interval analysis
'''


def naive_interval_analyze(net, epsilon, X, y, use_cuda=True, parallel=False, norm="linf"):
    # Transfer original model to interval models

    if (parallel):
        wc = nn.DataParallel(Interval_Bound(net, epsilon, method="naive", use_cuda=use_cuda, norm=norm))(X, y)
    else:
        wc = Interval_Bound(net, epsilon, method="naive", use_cuda=use_cuda, norm=norm)(X, y)

    iloss = nn.CrossEntropyLoss()(wc, y)
    ierr = (wc.max(1)[1] != y).type(torch.Tensor)
    ierr = ierr.sum().item() / X.shape[0]

    return iloss, ierr


'''Inverse interval propagations.

Args:
	model: regular nn.Sequential models
	epsilon: desired input ranges
	X and y: samples and lables

Return:
	iloss: robust loss provided by naive interval analysis
	ierr: verifiable robust error provided by naive interval analysis
'''


def inverse_interval_analyze(net, epsilon, X, y, use_cuda=True, parallel=False):
    # Transfer original model to interval models

    if (parallel):
        wc = nn.DataParallel(Interval_Bound(net, epsilon, method="inverse", use_cuda=use_cuda, norm=norm))(X, y)
    else:
        wc = Interval_Bound(net, epsilon, method="inverse", use_cuda=use_cuda, norm=norm)(X, y)

    iloss = nn.CrossEntropyLoss()(wc, y)
    ierr = (wc.max(1)[1] != y).type(torch.Tensor)
    ierr = ierr.sum().item() / X.shape[0]

    return iloss, ierr


'''Inverse interval propagations.

Args:
	model: regular nn.Sequential models
	epsilon: desired input ranges
	X and y: samples and lables

Return:
	iloss: robust loss provided by naive interval analysis
	ierr: verifiable robust error provided by naive interval analysis
'''


def center_symbolic_interval_analyze(net, epsilon, X, y, use_cuda=True, parallel=False):
    # Transfer original model to interval models

    if (parallel):
        wc = nn.DataParallel(Interval_Bound(net, epsilon, method="center_sym", use_cuda=use_cuda))(X, y)
    else:
        wc = Interval_Bound(net, epsilon, method="center_sym", use_cuda=use_cuda)(X, y)

    iloss = nn.CrossEntropyLoss()(wc, y)
    ierr = (wc.max(1)[1] != y).type(torch.Tensor)
    ierr = ierr.sum().item() / X.shape[0]

    return iloss, ierr


'''Symbolic interval propagations.

Args:
	model: regular nn.Sequential models
	epsilon: desired input ranges
	X and y: samples and lables
	use_cuda: whether we want to use cuda
	parallel: whehter we want to run on multiple gpus
	proj: project dimension to adjust the dimension. 
	Symbolic interval anlaysis keeps all of the input 
	dimension (proj=input_size/None) and thus is the 
	tigthest but needs more computations. 
	Naive interval throws away all of the dependency (proj=0).
	One can freely adjust tightness by controlling proj to be from
	0 to input size. 
	

Return:
	iloss: robust loss provided by symbolic interval analysis
	ierr: verifiable robust error provided by symbolic
	interval analysis
'''


def sym_interval_analyze(net, epsilon, X, y, use_cuda=True, parallel=False, proj=None, norm="linf"):
    if (parallel):
        wc = nn.DataParallel(Interval_Bound(net, epsilon, method="sym", proj=proj, use_cuda=use_cuda, norm=norm))(X, y)
    else:
        wc = Interval_Bound(net, epsilon, method="sym", proj=proj, use_cuda=use_cuda, norm=norm)(X, y)

    iloss = nn.CrossEntropyLoss()(wc, y)
    ierr = (wc.max(1)[1] != y)
    ierr = ierr.sum().item() / X.size(0)

    return iloss, ierr


def mix_interval_analyze(net, epsilon, X, y, use_cuda=True, parallel=False, proj=None, norm="linf"):
    if (parallel):
        wc = nn.DataParallel(Interval_Bound(net, epsilon, method="mix", proj=proj, use_cuda=use_cuda, norm=norm))(X, y)
    else:
        wc = Interval_Bound(net, epsilon, method="mix", proj=proj, use_cuda=use_cuda, norm=norm)(X, y)

    iloss = nn.CrossEntropyLoss()(wc, y)
    ierr = (wc.max(1)[1] != y)
    ierr = ierr.sum().item() / X.size(0)

    return iloss, ierr


def gen_interval_analyze(net, epsilon, X, y, use_cuda=True, parallel=False, proj=None, norm=["linf", "l2", "l1"]):
    if (parallel):
        wc = nn.DataParallel(Interval_Bound(net, epsilon, method="gen", proj=proj, use_cuda=use_cuda, norm=norm))(X, y)
    else:
        wc = Interval_Bound(net, epsilon, method="gen", proj=proj, use_cuda=use_cuda, norm=norm)(X, y)

    iloss = nn.CrossEntropyLoss()(wc, y)
    ierr = (wc.max(1)[1] != y)
    ierr = ierr.sum().item() / X.size(0)

    return iloss, ierr

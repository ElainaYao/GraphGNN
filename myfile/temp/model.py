#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib
matplotlib.use('Agg')

# Pytorch requirements
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor                      # why FloatTensor??
    dtype_l = torch.cuda.LongTensor

######## need to go back and write my own 
def GMul(W, x):
    # x is a tensor of size (batch_size, N, num_features), where num_features is the b_k
    # W is a tensor of size (batch_size, N, N, J), where J is the number of hop
    x_size = x.size()
    # print (x)
    W_size = W.size()
    # print (W)
    N = W_size[-3]
    J = W_size[-1]
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output


######## need to go back and write my own
class gnn_atomic_lg(nn.Module):
    def __init__(self, feature_maps, J):
        super(gnn_atomic_lg, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_inputs_2 = 2 * feature_maps[1]
        # self.num_inputs_3 = 4 * feature_maps[2]
        self.num_outputs = feature_maps[2]
        self.fcx2x_1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.fcy2x_1 = nn.Linear(self.num_inputs_2, self.num_outputs // 2)
        self.fcx2x_2 = nn.Linear(self.num_inputs, self.num_outputs - self.num_outputs // 2)
        self.fcy2x_2 = nn.Linear(self.num_inputs_2, self.num_outputs - self.num_outputs // 2)
        self.fcx2y_1 = nn.Linear(self.num_inputs_2, self.num_outputs // 2)
        self.fcy2y_1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.fcx2y_2 = nn.Linear(self.num_inputs_2, self.num_outputs - self.num_outputs // 2)
        self.fcy2y_2 = nn.Linear(self.num_inputs, self.num_outputs - self.num_outputs // 2)
        self.bn2d_x = nn.BatchNorm2d(self.num_outputs)
        self.bn2d_y = nn.BatchNorm2d(self.num_outputs)

    def forward(self, WW, x, WW_lg, y, P):
        # print ('W size', W.size())
        # print ('x size', input[1].size())
        x2x = GMul(WW, x) # out has size (bs, N, num_inputs)
        x2x_size = x2x.size()
        # print (x_size)
        x2x = x2x.contiguous()
        x2x = x2x.view(-1, self.num_inputs)
        # print (x.size()) 
        # print ('x2x', x2x)
        x2x = x2x.type(dtype)

        # y2x = torch.bmm(P, y)
        y2x = GMul(P, y)
        y2x_size = y2x.size()
        y2x = y2x.contiguous()
        y2x = y2x.view(-1, self.num_inputs_2)

        y2x = y2x.type(dtype)

        # xy2x = x2x + y2x 
        xy2x = F.relu(self.fcx2x_1(x2x) + self.fcy2x_1(y2x)) # has size (bs*N, num_outputs)

        xy2x_l = self.fcx2x_2(x2x) + self.fcy2x_2(y2x)
        x_cat = torch.cat((xy2x, xy2x_l), 1)
        # x_output = self.bn2d_x(x_cat)
        x_output = self.bn2d_x(x_cat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)

        x_output = x_output.view(*x2x_size[:-1], self.num_outputs)


        y2y = GMul(WW_lg, y)
        y2y_size = y2y.size()
        y2y = y2y.contiguous()
        y2y = y2y.view(-1, self.num_inputs)

        y2y = y2y.type(dtype)

        # x2y = torch.bmm(torch.t(P), x)
        x2y = GMul(torch.transpose(P, 2, 1), x)
        x2y_size = x2y.size()
        x2y = x2y.contiguous()
        x2y = x2y.view(-1, self.num_inputs_2)

        x2y = x2y.type(dtype)

        # xy2y = x2y + y2y
        xy2y = F.relu(self.fcx2y_1(x2y) + self.fcy2y_1(y2y))

        xy2y_l = self.fcx2y_2(x2y) + self.fcy2y_2(y2y)

        y_cat = torch.cat((xy2y, xy2y_l), 1)
        # y_output = self.bn2d_x(y_cat)
        y_output = self.bn2d_y(y_cat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)

        y_output = y_output.view(*y2y_size[:-1], self.num_outputs)

        # WW = WW.type(dtype)

        return WW, x_output, WW_lg, y_output, P

class gnn_atomic_lg_final(nn.Module):
    def __init__(self, feature_maps, J, num_classes):
        super(gnn_atomic_lg_final, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_inputs_2 = 2 * feature_maps[1]
        self.num_outputs = num_classes
        self.fcx2x_1 = nn.Linear(self.num_inputs, self.num_outputs)
        self.fcy2x_1 = nn.Linear(self.num_inputs_2, self.num_outputs)

    def forward(self, W, x, W_lg, y, P):
        # print ('W size', W.size())
        # print ('x size', input[1].size())
        x2x = GMul(W, x) # out has size (bs, N, num_inputs)
        x2x_size = x2x.size()
        # print (x_size)
        x2x = x2x.contiguous()
        x2x = x2x.view(-1, self.num_inputs)
        # print (x.size()) 

        # y2x = torch.bmm(P, y)
        y2x = GMul(P, y)
        y2x_size = x2x.size()
        y2x = y2x.contiguous()
        y2x = y2x.view(-1, self.num_inputs_2)

        # xy2x = x2x + y2x 
        xy2x = self.fcx2x_1(x2x) + self.fcy2x_1(y2x) # has size (bs*N, num_outputs)

        x_output = xy2x.view(*x2x_size[:-1], self.num_outputs)

        return W, x_output


class lGNN_multiclass(nn.Module):
    def __init__(self, num_features, num_layers, J, num_classes=2):
        super(lGNN_multiclass, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.featuremap_in = [1, 1, num_features]
        self.featuremap_mi = [num_features, num_features, num_features]
        self.featuremap_end = [num_features, num_features, num_features]
        # self.layer0 = Gconv(self.featuremap_in, J)
        self.layer0 = gnn_atomic_lg(self.featuremap_in, J)
        for i in range(num_layers):
            # module = Gconv(self.featuremap_mi, J)
            module = gnn_atomic_lg(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = gnn_atomic_lg_final(self.featuremap_end, J, num_classes)

    def forward(self, W, x, W_lg, y, P):
        cur = self.layer0(W, x, W_lg, y, P)
        for i in range(self.num_layers):
            cur = self._modules['layer{}'.format(i + 1)](*cur)
        out = self.layerlast(*cur)
        return out[1]  # out[0] = W in gnn_atomic_lg_final


if __name__ == '__main__':
    # test modules
    bs =  4
    num_features = 10
    num_layers = 5
    N = 8
    x = torch.ones((bs, N, num_features))
    W1 = torch.eye(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    W2 = torch.ones(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    J = 2
    W = torch.cat((W1, W2), 3)
    input = [Variable(W), Variable(x)]

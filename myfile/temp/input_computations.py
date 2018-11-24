#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import networkx

#Pytorch requirements
import unicodedata
import string
import re
import random
import argparse

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

def get_operators(W, J):
    N = W.shape[0]
    # operators: {Id, W, W^2, ..., W^{J-1}, D, U}
    d = W.sum(1)
    D = np.diag(d)
    Wcp= W.copy()
    Wnew = np.zeros([N, N, J+2])
    Wnew[:, :, 0] = np.eye(N)
    Wnew[:, :, 1] = D
    
    for j in range(J):
        Wnew[:, :, j+2] = Wcp
        Wcp = np.minimum(np.dot(Wcp, Wcp), np.ones(Wcp.shape))
    Wnew = np.reshape(Wnew, [N, N, J + 2])
    x = np.reshape(d, [N, 1])
    return Wnew, x

def get_Pm(W):
    N = W.shape[0]
    W = W * (np.ones([N, N]) - np.eye(N))
    M = int(W.sum())
    p = 0
    Pm = np.zeros([N, M * 2])
    for n in range(N):
        for m in range(n+1, N):
            if (W[n][m]==1):
                Pm[n][p] = 1
                Pm[m][p] = 1
                Pm[n][p + M] = 1
                Pm[m][p + M] = 1
                p += 1
    return Pm

def get_Pd(W):
    N = W.shape[0]
    W = W * (np.ones([N, N]) - np.eye(N))
    M = int(W.sum())
    p = 0
    Pd = np.zeros([N, M * 2])
    for n in range(N):
        for m in range(n+1, N):
            if (W[n][m]==1):
                Pd[n][p] = 1
                Pd[m][p] = 1
                Pd[n][p + M] = 1
                Pd[m][p + M] = 1
                p += 1
    return Pd


def get_P(W):
    P = np.concatenate((np.expand_dims(get_Pm(W), 2), np.expand_dims(get_Pd(W), 2)), axis=2)
    return P

def get_W_lg(W):
    W_lg = np.transpose(get_Pm(W)).dot(get_Pd(W))
    return W_lg

def get_lg_inputs(W, J):
    WW, x = get_operators(W, J)
    W_lg = get_W_lg(W)
    WW_lg, y = get_operators(W_lg, J)
    P = get_P(W)
    x = x.astype(float)
    y = y.astype(float)
    WW = WW.astype(float)
    WW_lg = WW_lg.astype(float)
    P = P.astype(float)
    # torch.tensor() always copies data. If you have a Tensor data and want to avoid a copy, 
    # use torch.Tensor.requires_grad_() or torch.Tensor.detach(). 
    # If you have a NumPy ndarray and want to avoid a copy, use torch.from_numpy().
#    WW = Variable(torch.from_numpy(WW).unsqueeze(0), volatile=False)
#    x = Variable(torch.from_numpy(x).unsqueeze(0), volatile=False)
#    WW_lg = Variable(torch.from_numpy(WW_lg).unsqueeze(0), volatile=False)
#    y = Variable(torch.from_numpy(y).unsqueeze(0), volatile=False)
#    P = Variable(torch.from_numpy(P).unsqueeze(0), volatile=False)
    WW = torch.from_numpy(WW).unsqueeze(0)
    x = torch.from_numpy(x).unsqueeze(0)
    WW_lg = torch.from_numpy(WW_lg).unsqueeze(0)
    y = torch.from_numpy(y).unsqueeze(0)
    P = torch.from_numpy(P).unsqueeze(0)
    return WW, x, WW_lg, y, P



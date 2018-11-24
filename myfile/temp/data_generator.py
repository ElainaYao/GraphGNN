import numpy as np
# os module provides dozens of functions for interacting with the operating system
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

class Generator(object):
    def __init__(self, num_nodes, edge_density):
        self.p = edge_density
        self.N = num_nodes

    def ErdosRenyi(self, cuda=True):
        g = networkx.erdos_renyi_graph(self.N, self.p)
        W = networkx.adjacency_matrix(g).todense().astype(float)
        W = np.array(W)
        return W

    def RegularGraph(self, cuda=True):
        """ Generate random regular graph """
        d = self.p * self.N # with input p
        d = int(d)
        g = networkx.random_regular_graph(d, N)
        W = networkx.adjacency_matrix(g).todense().astype(float)
        W = np.array(W)
        return W

if __name__ == '__main__':
    # execute only if run as a script
    ################### Test graph generators ########################
    p = 0.5
    N = 50

    
    gen = Generator(N, p)
    Wer = gen.ErdosRenyi()

    # Return a graph from numpy matrix:
    Ger = networkx.from_numpy_matrix(Wer)
    #Grg = networkx.from_numpy_matrix(Wrg)
    networkx.draw(Ger)
    plt.savefig('/Users/Rebecca_yao/Documents/RESEARCH/Graph/myfile/temp/testDataGen_ER.png')
    #networkx.draw(Grg)
    #plt.savefig('/Users/Rebecca_yao/Documents/RESEARCH/Graph/myfile/temp/testDataGen_RG.png')
    #print('Wer', Wer)
    #print('Wrg', Wrg)


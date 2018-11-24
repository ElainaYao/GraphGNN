import numpy as np
import math
import os
# import dependencies
import time

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

softmax = nn.Softmax(dim=1)

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor

def compute_loss_rlx(pred, WW, Lambda):
    loss = 0
    if (torch.cuda.is_available()):
        batch_size = pred.data.cpu().shape[0]
    else:
        batch_size = pred.data.shape[0]
    N = pred.data.shape[1]
    for i in range(batch_size):
        W = WW.type(dtype)[i,:,:,2]
        D = WW.type(dtype)[i,:,:,1]
        L = D - W
        pred_single = pred[i, :, :]
        pred_prob_single = softmax(pred_single) # check pred_prob_single.sum(dim = 1)
        pp = pred_prob_single[:, 0] # or 1, same since we only have two classes and the loss is symmetric

        loss_single = 1/4 * torch.dot(2*pp-1, torch.mv(L, 2*pp-1)) + Lambda * torch.norm(pp.sum()-N/2).pow(2)
        #loss_single = 1/4 * torch.dot(2*pp-1, torch.mv(L, 2*pp-1)) + Lambda*torch.norm(labels_single_t.sum()).pow(2)
        loss += loss_single
    # no need to further average, it is just a normalizing constant
    # avg_loss = loss/batch_size
    return loss

def compute_loss_acc(pred, WW):   
    if (torch.cuda.is_available()):
        batch_size = pred.data.cpu().shape[0]
    else:
        batch_size = pred.data.shape[0]
    acc = 0
    inb = 0
    for i in range(batch_size):
        W = WW.type(dtype)[i,:,:,2]
        D = WW.type(dtype)[i,:,:,1]
        L = D - W
        pred_single = pred[i, :, :]
        pred_prob_single = softmax(pred_single) # check pred_prob_single.sum(dim = 1)
        labels_single = torch.argmax(pred_prob_single, dim = 1) * 2 - 1
        labels_single_t = torch.tensor(labels_single).type(dtype)
        acc_single = 1/4 * torch.dot(labels_single_t, torch.mv(L, labels_single_t))
        inb_single = torch.sum(labels_single_t)
        acc += acc_single
        inb += inb_single
    # no need to further average, it is just a normalizing constant
    #avg_acc = acc/batch_size  
    #avg_inb = inb/batch_size
    return acc, inb


def compute_loss_acc_weighted(pred, WW, Lambda):   
    if (torch.cuda.is_available()):
        batch_size = pred.data.cpu().shape[0]
    else:
        batch_size = pred.data.shape[0]
    loss_weighted = 0
    acc = 0
    inb = 0
    for i in range(batch_size):
        W = WW.type(dtype)[i,:,:,2]
        D = WW.type(dtype)[i,:,:,1]
        L = D - W
        pred_single = pred[i, :, :]
        pred_prob_single = softmax(pred_single) # check pred_prob_single.sum(dim = 1)
        labels_single = torch.argmax(pred_prob_single, dim = 1) * 2 - 1
        labels_single_t = torch.tensor(labels_single).type(dtype)
        weight_single = torch.cumprod(torch.max(pred_prob_single, dim = 1)[0], dim = 0)[-1]
        #print(weight_single)
        acc_single = 1/4 * torch.dot(labels_single_t, torch.mv(L, labels_single_t))
        inb_single = torch.sum(labels_single_t)     
        loss_single_weighted = (acc_single + torch.norm(inb_single).pow(2)) * weight_single * Lambda # weighted by llh
        
        loss_weighted += loss_single_weighted
        acc += acc_single
        inb += torch.abs(inb_single)
    # no need to further average for the weighted version, it is just a normalizing constant
    # for nonweighted version, to print as output
    avg_loss_weighted = loss_weighted/batch_size
    avg_acc = acc/batch_size  
    avg_inb = inb/batch_size
    return avg_loss_weighted, avg_acc, avg_inb

 


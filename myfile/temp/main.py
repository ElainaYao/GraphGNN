import numpy as np
import os
# import dependencies
from data_generator import Generator
from input_computations import get_lg_inputs
from model import lGNN_multiclass
from log_definition import Logger
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#Pytorch requirements
import unicodedata
import string
import re
import random
import argparse
import pickle

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from loss import compute_loss_rlx, compute_loss_acc, compute_loss_acc_weighted
import pandas

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    # torch.manual_seed(1)

template1 = '{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} '
template2 = '{:<10} {:<10.3f} {:<10.3f} {:<10} {:<10.3f} {:<10} {:<10.3f} \n'
template3 = '{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} '
template4 = '{:<10} {:<10} {:<10.2f} {:<10} {:<10.3f} {:<10.1f} {:<10} {:<10.3f} \n'
template5 = '{:<10} {:<10} {:<10} {:<10} {:<10} {:<11} {:<10} {:<10} {:<10} '
template6 = '{:<10} {:<10} {:<10.2f} {:<10} {:<10.3f} {:<11.12} {:<10.1f} {:<10} {:<10.3f} \n'




def train_single_W(gnn, optimizer, logger, W, Lambda, it, args):
    start = time.time()
    WW, x, WW_lg, y, P = get_lg_inputs(W, args.J)
    if (torch.cuda.is_available()):
        WW.cuda()
        x.cuda()
        WW_lg.cuda()
        y.cuda()
        P.cuda()
    pred = gnn(WW.type(dtype), x.type(dtype), WW_lg.type(dtype), y.type(dtype), P.type(dtype))
    loss = compute_loss_rlx(pred, WW, Lambda)
    gnn.zero_grad()
    loss.backward()
    # Clips gradient norm of an iterable of parameters.
    # The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.
    nn.utils.clip_grad_norm_(gnn.parameters(), args.clip_grad_norm)
    optimizer.step()
    acc, inb = compute_loss_acc(pred, WW) 
    elapsed = time.time() - start

    if(torch.cuda.is_available()):
        loss_value = float(loss.data.cpu().numpy())
        acc_value = float(acc.data.cpu().numpy())
    else:
        loss_value = float(loss.data.numpy())
        acc_value = float(acc.data.numpy())

    info = ['epoch', 'avg loss', 'avg acc', 'avg inb', 'Lambda', 'edge_den', 'elapsed']
    out = [it, loss_value, acc_value, abs(inb), Lambda, args.edge_density, elapsed]
    print(template1.format(*info))
    print(template2.format(*out))

    del WW
    del WW_lg
    del x
    del y
    del P

    return loss_value, acc_value, inb


def train_W(gnn, logger, W_all, args, iters = None):
    if iters is None:
        iters = args.num_examples_train
    gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr = args.lr)
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])
    inb_lst = np.zeros([iters])
    for it in range(iters):
        W = W_all[it, :, :]
        Lambda = args.Lambda * pow((1 + args.LambdaIncRate), it)
        loss_single, acc_single, inb_single = train_single_W(gnn, optimizer, logger, W, Lambda, it, args)
        loss_lst[it] = loss_single
        acc_lst[it] = acc_single
        inb_lst[it] = inb_single
        torch.cuda.empty_cache()
    return loss_lst, acc_lst, inb_lst



def train_single(gnn, optimizer, logger, gen, Lambda, it, args):
    start = time.time()
    if (args.generative_model == 'ErdosRenyi'):
        W = gen.ErdosRenyi()
    elif (args.generative_model == 'RegularGraph'):
        W = gen.RegularGraph()
    WW, x, WW_lg, y, P = get_lg_inputs(W, args.J)
    if (torch.cuda.is_available()):
        WW.cuda()
        x.cuda()
        WW_lg.cuda()
        y.cuda()
        P.cuda()
    pred = gnn(WW.type(dtype), x.type(dtype), WW_lg.type(dtype), y.type(dtype), P.type(dtype))
    loss = compute_loss_rlx(pred, WW, Lambda)
    gnn.zero_grad()
    loss.backward()
    # Clips gradient norm of an iterable of parameters.
    # The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.
    nn.utils.clip_grad_norm_(gnn.parameters(), args.clip_grad_norm)
    optimizer.step()
    acc, inb = compute_loss_acc(pred, WW) 
    elapsed = time.time() - start

    if(torch.cuda.is_available()):
        loss_value = float(loss.data.cpu().numpy())
        acc_value = float(acc.data.cpu().numpy())
    else:
        loss_value = float(loss.data.numpy())
        acc_value = float(acc.data.numpy())
    
    info = ['epoch', 'avg loss', 'avg acc', 'avg inb', 'Lambda', 'edge_den', 'elapsed']
    out = [it, loss_value, acc_value, abs(inb), Lambda, args.edge_density, elapsed]
    print(template1.format(*info))
    print(template2.format(*out))

    del WW
    del WW_lg
    del x
    del y
    del P

    return loss_value, acc_value, inb

def train(gnn, logger, gen, args, iters=None):
    if iters is None:
        iters = args.num_examples_train

    gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr = args.lr)
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])
    inb_lst = np.zeros([iters])
    for it in range(iters):
        Lambda = args.Lambda * pow((1 + args.LambdaIncRate),it)
        loss_single, acc_single, inb_single = train_single(gnn, optimizer, logger, gen, Lambda, it, args)
        loss_lst[it] = loss_single
        acc_lst[it] = acc_single
        inb_lst[it] = inb_single
        torch.cuda.empty_cache()
    print ('Avg train loss', np.mean(loss_lst))
    print ('Avg train acc', np.mean(acc_lst))
    print ('Avg train inbalance', np.mean(inb_lst))
    return loss_lst, acc_lst, inb_lst

def test_single(gnn, logger, W, it, args):
    start = time.time()
    WW, x, WW_lg, y, P = get_lg_inputs(W, args.J)
    if (torch.cuda.is_available()):
        WW.cuda()
        x.cuda()
        WW_lg.cuda()
        y.cuda()
        P.cuda()
    pred = gnn(WW.type(dtype), x.type(dtype), WW_lg.type(dtype), y.type(dtype), P.type(dtype))
    acc, inb = compute_loss_acc(pred, WW) 

    elapsed = time.time() - start

    if(torch.cuda.is_available()):
        acc_value = float(acc.data.cpu().numpy())
    else:
        acc_value = float(acc.data.numpy())

    info = ['epoch', 'mode', 'avg acc', 'avg inb', 'Lambda', 'edge_den', 'elapsed']
    out = [it, 'test', acc_value, inb, args.Lambda, args.edge_density, elapsed]
    print(template1.format(*info))
    print(template2.format(*out))

    del WW
    del WW_lg
    del x
    del y
    del P

    return acc_value, inb

def test(gnn, loger, W_all, args, iters = None):
    if iters is None:
        iters = args.num_examples_test
        
    gnn.train()
    acc_lst = np.zeros([iters])
    inb_lst = np.zeros([iters])
    for it in range(iters):
        W = W_all[it, :, :]
        acc_single, inb_single = test_single(gnn, logger, W, it, args)
        acc_lst[it] = acc_single
        inb_lst[it] = inb_single
        torch.cuda.empty_cache()
    print ('Avg test acc', np.mean(acc_lst))
    print ('Avg test inb', np.mean(inb_lst))
    return acc_lst, inb_lst

def train_single_aggre(gnn, logger, gen, Lambda, args, iters = None):
    if iters is None:
        iters = args.num_examples_train
    loss = 0
    acc = 0
    inb = 0
    for it in range(iters):
        start = time.time()
        Wer = gen.ErdosRenyi()
        WW, x, WW_lg, y, P = get_lg_inputs(Wer, args.J)
        if (torch.cuda.is_available()):
            WW.cuda()
            x.cuda()
            WW_lg.cuda()
            y.cuda()
            P.cuda()
        pred = gnn(WW.type(dtype), x.type(dtype), WW_lg.type(dtype), y.type(dtype), P.type(dtype))
        loss_single, acc_single, inb_single = compute_loss_acc_weighted(pred, WW, Lambda)
        del WW
        del WW_lg
        del x
        del y
        del P
        loss += loss_single
        acc += acc_single
        inb += inb_single
        if(torch.cuda.is_available()):
        #    loss_single_value = float(loss_single.data.cpu().numpy())
            acc_single_value = float(acc_single.data.cpu().numpy())
            inb_single_value = float(inb_single.data.cpu().numpy())
        else:
        #    loss_single_value = float(loss_single.data.numpy())
            acc_single_value = float(acc_single.data.numpy())
            inb_single_value = float(inb_single.data.cpu().numpy())
        elapsed = time.time() - start
        info = ['Mode', 'epoch', 'acc', 'inb', 'Lambda', 'edge_den', 'num_tr', 'elapsed']
        out = ['iter', it, acc_single_value, inb_single_value, Lambda, args.edge_density, args.num_examples_train, elapsed]
        print(template3.format(*info))
        print(template4.format(*out))
    avg_loss = loss/iters
    avg_acc = acc/iters
    avg_inb = inb/iters
    return avg_loss, avg_acc, avg_inb

def train_aggre(gnn, logger, gen, args, iters_aggre = None):
    if iters_aggre is None:
        iters_aggre = args.num_iters_aggre

    gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr = args.lr)
    loss_lst = np.zeros([iters_aggre])
    acc_lst = np.zeros([iters_aggre])
    inb_lst = np.zeros([iters_aggre])
    for It in range(iters_aggre):
        start = time.time()
        Lambda = args.Lambda * pow((1 + args.LambdaIncRate), It)
        avg_loss, avg_acc, avg_inb = train_single_aggre(gnn, logger, gen, Lambda, args, iters = None)
        gnn.zero_grad()
        avg_loss.backward()
        nn.utils.clip_grad_norm_(gnn.parameters(), args.clip_grad_norm)
        optimizer.step()
        if(torch.cuda.is_available()):
            loss_lst[It] = float(avg_loss.data.cpu().numpy())
            acc_lst[It] = float(avg_acc.data.cpu().numpy())
            inb_lst[It] = float(avg_inb.data.cpu().numpy())
        else:
            loss_lst[It] = float(avg_loss.data.numpy())
            acc_lst[It] = float(avg_acc.data.numpy())
            inb_lst[It] = float(avg_inb.data.numpy())
        elapsed = time.time() - start
        info = ['Mode', 'epoch', 'avg acc', 'avg inb', 'initLambda', 'Lbd incRate', 'num_tr', 'elapsed']
        out = ['Iter', It, acc_lst[It], inb_lst[It], args.Lambda, args.LambdaIncRate, args.num_examples_train, elapsed]
        print(template3.format(*info))
        print(template4.format(*out))
        torch.cuda.empty_cache()
    return loss_lst, acc_lst, inb_lst


def read_args_commandline():
    parser = argparse.ArgumentParser()
    # Parser for command-line options, arguments and subcommands
    # The argparse module makes it easy to write user-friendly command-line interfaces.
    
    ###############################################################################
    #                             General Settings                                #
    ###############################################################################
    parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                        default=int(1000))
    parser.add_argument('--num_iters_aggre', nargs='?', const=1, type=int,
                        default=int(100))
    parser.add_argument('--loss_method', nargs='?', const=1, type=int,
                        default=int(1))
    parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                        default=int(1000))
    parser.add_argument('--edge_density', nargs='?', const=1, type=float,
                        default=0.5)
    parser.add_argument('--generative_model', nargs='?', const=1, type=str,
                        default='ErdosRenyi')
    parser.add_argument('--num_nodes', nargs='?', const=1, type=int,
                        default=50)
    parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=1)
    parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')
    parser.add_argument('--path_output', nargs='?', const=1, type=str, default='')
    parser.add_argument('--path_logger', nargs='?', const=1, type=str, default='')
    parser.add_argument('--path_gnn', nargs='?', const=1, type=str, default='')
    parser.add_argument('--filename_existing_gnn', nargs='?', const=1, type=str, default='')
    parser.add_argument('--print_freq', nargs='?', const=1, type=int, default=100)
    parser.add_argument('--test_freq', nargs='?', const=1, type=int, default=500)
    parser.add_argument('--save_freq', nargs='?', const=1, type=int, default=2000)
    parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float,
                        default=40.0)
    parser.add_argument('--Lambda', nargs='?', const=1, type=float,
                        default=10)
    parser.add_argument('--LambdaIncRate', nargs='?', const=1, type=float,
                        default=0.05)
    parser.add_argument('--freeze_bn', dest='eval_vs_train', action='store_true')
    parser.set_defaults(eval_vs_train=False)

    ###############################################################################
    #                                 GNN Settings                                #
    ###############################################################################

    parser.add_argument('--num_features', nargs='?', const=1, type=int,
                        default=20)
    parser.add_argument('--num_layers', nargs='?', const=1, type=int,
                        default=20)
    parser.add_argument('--num_classes', nargs='?', const=1, type=int,
                        default=2)
    parser.add_argument('--J', nargs='?', const=1, type=int, default=4)
    parser.add_argument('--lr', nargs='?', const=1, type=float, default=1e-3)

    return parser.parse_args()


def main():
    args = read_args_commandline()
    batch_size = args.batch_size
    
    logger = Logger(args.path_logger)
    logger.write_settings(args)

    torch.backends.cudnn.enabled=False
    
    if (args.mode == 'test'):
        print ('In testing mode')
        dataname = 'testdata' + '_num' + str(args.num_examples_test) + '_nodes' + str(args.num_nodes) + '_edgedens' + str(args.edge_density) + '.pickle'
        path_plus_name = os.path.join(args.path_output, dataname)
        if (os.path.exists(path_plus_name)):
            print ('Loading test data ' + dataname)
            with open(path_plus_name, 'rb') as f:
                W_all = pickle.load(f)
        else:
            print ('No such a test data set exists; creating a brand new one')
            random.seed(101)
            seed_seq = sorted(random.sample(range(1, args.num_examples_test*1000000), args.num_examples_test))
            W_all = np.zeros([args.num_examples_test, args.num_nodes, args.num_nodes])
            for data_i in range(args.num_examples_test):
                random.seed(seed_seq[data_i])
                gen = Generator(args.num_nodes, args.edge_density)
                if (args.generative_model == 'ErdosRenyi'):
                    W_all[data_i,:,:] = gen.ErdosRenyi()
                elif (args.generative_model == 'RegularGraph'):
                    W_all[data_i,:,:] = gen.RegularGraph()
                else:
                    print('Wrong generative model specified, please check again')
                    break
            #dataname = 'testdata' + '_num' + str(num_examples_test) + '_nodes' + str(num_nodes) + '_edgedens' + str(edge_density) + '.pickle'
            #    with open(dataname, 'wb') as f:
            #        pickle.dump(W_all, f, pickle.HIGHEST_PROTOCOL)


        # filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(gen.N_test) + '_it' + str(args.iterations)
        filename = args.filename_existing_gnn
        path_plus_name = os.path.join(args.path_gnn, filename)
        if ((filename != '') and (os.path.exists(path_plus_name))):
            print ('Loading gnn ' + filename)
            gnn = torch.load(path_plus_name)
            if torch.cuda.is_available():
                gnn.cuda()
        else:
            print ('No such a gnn exists; creating a brand new one')
            gnn = lGNN_multiclass(args.num_features, args.num_layers, args.J + 2, args.num_classes)
            filename = 'lgnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Lbd' + str(args.Lambda) + '_num' + str(args.num_examples_train)
            path_plus_name = os.path.join(args.path_gnn, filename)
            if torch.cuda.is_available():
                gnn.cuda()
            print ('Testing begins')
            acc, inb = test(gnn, logger, W_all, args, iters = None)
            print('Saving the testing results')
            res = {
                'acc': acc,
                'inb': inb
            }
            resname = 'OUTtest_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Lbd' + str(args.Lambda) + '_num' + str(args.num_examples_train) + '.pickle'
            path_plus_name = os.path.join(args.path_output, resname)
            with open(path_plus_name, 'wb') as f:
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
                # import pandas as pd: to view it as a data frame
                # pd.DataFrame.from_dict(res) 
            # import pickle
            # read the pickle file
            # with open('data.pickle', 'rb') as f:
                # The protocol version used is detected automatically, so we do not
                # have to specify it.
            #   data = pickle.load(f)
    elif (args.mode == 'train'):
        
        if (args.loss_method == 1):
            dataname = 'traindata' + '_num' + str(args.num_examples_train) + '_nodes' + str(args.num_nodes) + '_edgedens' + str(args.edge_density) + '.pickle'
            path_plus_name = os.path.join(args.path_output, dataname)
            if (os.path.exists(path_plus_name)):
                print ('Loading train data ' + dataname)
                with open(path_plus_name, 'rb') as f:
                    W_all = pickle.load(f)
            else:
                print ('No such a train data set exists; creating a brand new one')
                random.seed(0)
                seed_seq = sorted(random.sample(range(1, args.num_examples_train*1000000), args.num_examples_train))
                W_all = np.zeros([args.num_examples_train, args.num_nodes, args.num_nodes])
                for data_i in range(args.num_examples_train):
                    random.seed(seed_seq[data_i])
                    gen = Generator(args.num_nodes, args.edge_density)
                    if (args.generative_model == 'ErdosRenyi'):
                        W_all[data_i,:,:] = gen.ErdosRenyi()
                    elif (args.generative_model == 'RegularGraph'):
                        W_all[data_i,:,:] = gen.RegularGraph()
                    else:
                        print('Wrong generative model specified, please check again')
                        break
                dataname = 'traindata' + '_num' + str(args.num_examples_train) + '_nodes' + str(args.num_nodes) + '_edgedens' + str(args.edge_density) + '.pickle'
                path_plus_name = os.path.join(args.path_output, dataname)
                with open(path_plus_name, 'wb') as f:
                    pickle.dump(W_all, f, pickle.HIGHEST_PROTOCOL)
                
            filename = args.filename_existing_gnn
            path_plus_name = os.path.join(args.path_gnn, filename)
            if ((filename != '') and (os.path.exists(path_plus_name))):
                print ('Loading gnn ' + filename)
                gnn = torch.load(path_plus_name)
                filename = filename + '_Lbd' + str(args.Lambda) + '_num' + str(args.num_examples_train)
                path_plus_name = os.path.join(args.path_gnn, filename)
            else:
                print ('No such a gnn exists; creating a brand new one')
                filename = 'lgnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Lbd' + str(args.Lambda) + '_LbdR' + str(args.LambdaIncRate) + '_num' + str(args.num_examples_train)
                path_plus_name = os.path.join(args.path_gnn, filename)
                gnn = lGNN_multiclass(args.num_features, args.num_layers, args.J + 2, args.num_classes)
            
            if torch.cuda.is_available():
                    gnn.cuda()
            print ('Training begins')
            loss, acc, inb = train_W(gnn, logger, W_all, args, iters = None)
            print ('Saving gnn ' + filename)
            if torch.cuda.is_available():
                torch.save(gnn.cpu(), path_plus_name)
                gnn.cuda()
            else:
                torch.save(gnn, path_plus_name)
            res = {
                'loss': loss,
                'acc': acc,
                'inb': inb
            }
            resname = 'res_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Lbd' + str(args.Lambda) + '_LbdR' + str(args.LambdaIncRate) + '_num' + str(args.num_examples_train) + '.pickle'
            path_plus_name = os.path.join(args.path_output, resname)
            print ('Saving loss, acc, inb ' + resname)
            with open(path_plus_name, 'wb') as f:
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
                # import pandas as pd: to view it as a data frame
                # pd.DataFrame.from_dict(res) 
            # import pickle
            # read the pickle file
            # with open('data.pickle', 'rb') as f:
                # The protocol version used is detected automatically, so we do not
                # have to specify it.
            #   data = pickle.load(f)
        elif (args.loss_method == 2):
            gen = Generator(args.num_nodes, args.edge_density)
            filename = args.filename_existing_gnn
            path_plus_name = os.path.join(args.path_gnn, filename)
            if ((filename != '') and (os.path.exists(path_plus_name))):
                print ('Loading gnn ' + filename)
                gnn = torch.load(path_plus_name)
                filename = filename + '_Lbd' + str(args.Lambda) + '_num' + str(args.num_examples_train)
                path_plus_name = os.path.join(args.path_gnn, filename)
            else:
                print ('No such a gnn exists; creating a brand new one')
                filename = 'lgnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Lbd' + str(args.Lambda) + '_LbdR' + str(args.LambdaIncRate) + '_num' + str(args.num_examples_train) + '_Iter' + str(args.num_iters_aggre)
                path_plus_name = os.path.join(args.path_gnn, filename)
                gnn = lGNN_multiclass(args.num_features, args.num_layers, args.J + 2, args.num_classes)
            
            if torch.cuda.is_available():
                gnn.cuda()
            print ('Training begins')
            loss, acc, inb = train_aggre(gnn, logger, gen, args, iters = None)
            print ('Saving gnn ' + filename)
            if torch.cuda.is_available():
                torch.save(gnn.cpu(), path_plus_name)
                gnn.cuda()
            else:
                torch.save(gnn, path_plus_name)
            res = {
                'loss': loss,
                'acc': acc,
                'inb': inb
            }
            resname = 'res_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Lbd' + str(args.Lambda) + '_LbdR' + str(args.LambdaIncRate) + '_num' + str(args.num_examples_train) + '_Iter' + str(args.num_iters_aggre) + '.pickle'
            path_plus_name = os.path.join(args.path_output, resname)
            print ('Saving loss, acc, inb ' + resname)
            with open(path_plus_name, 'wb') as f:
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
                # import pandas as pd: to view it as a data frame
                # pd.DataFrame.from_dict(res) 
            # import pickle
            # read the pickle file
            # with open('data.pickle', 'rb') as f:
                # The protocol version used is detected automatically, so we do not
                # have to specify it.
            #   data = pickle.load(f)


if __name__ == '__main__':
    main()
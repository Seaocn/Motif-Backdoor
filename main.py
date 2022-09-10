import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, separate_data
from models.graphcnn import GraphCNN
import random
from util import read_graphfile
from models.GcnEncoderGraph import GcnEncoderGraph
from models.Diffpool import Diffpool
from models.graphcnn_neuron import GraphCNN_Neuron
from models.sage import GraphSAGE
from models.gcn import GCN
from models.Sagepool import Sagepool
from models.HGP_SL import HGPSLPool
from models.GAT import GAT




criterion = nn.CrossEntropyLoss()


# '174服务器  序号1-gpu7,3-1，4-2,0-6,2-0,5-3,6-4,7-5'
#设置cpu占用
num_threads = 1
torch.set_num_threads(num_threads)

#gpu 7#--5


def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output,_ = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx])[0].detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="twitch_egos",   #MCF-7    MUTAG 1, IMDBBINARY 0, PROTEINS 1, Fingerprint 2, AIDS 0, NCI1 0, twitch_egos
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    # '174服务器  序号1-gpu7,3-1，4-2,0-6,2-0,5-3,6-4,7-5'
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type=str, default="",
                        help='output file')
    parser.add_argument('--output_dim', type=int, default=64)
    parser.add_argument('--num_gc_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--assign_ratio', type=float, default=0.2)
    parser.add_argument('--num_pool', type=int, default=1)
    parser.add_argument('--pooling_ratio', type=float, default=0.8) #Sagepool

    #HGP_SL
    parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
    parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
    parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
    parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')

    #对比学习的参数
    parser.add_argument('--adv_rate', type=float, default=0.1,
                        help='probability for edge creation/rewiring each edge')
    parser.add_argument('--prob', type=float, default=0.8,
                        help='probability for edge creation/rewiring each edge')
    parser.add_argument('--num_backdoor_nodes', type=int, default=4,
                        help='Each node is connected to k nearest neighbors in ring topology')
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--K', type=int, default=3,
                        help='Each node is connected to k nearest neighbors in ring topology')
    parser.add_argument('--frac', type=float, default=0.05, help='poisoning ratio')
    parser.add_argument('--modeltype', type=str, default='gcn', choices=['gin', 'Sagepool', 'HGP-SL', 'gcn', 'gat', 'diffpool', 'sage'])
    parser.add_argument('--iters_per_cl', type=int, default=50)
    parser.add_argument('--randomly_preserve', type=float, default=0.1)
    parser.add_argument('--matchrate', type=float, default=0.5)
    parser.add_argument('--edge_drop_rate1', type=float, default= 0.2)
    parser.add_argument('--edge_drop_rate2', type=float, default= 0.2)
    parser.add_argument('--edge_drop_rate3', type=float, default= 0.2)
    parser.add_argument('--sim_idx1', type=str, default='DC')   #DC/CC/KC/HC/BC
    parser.add_argument('--sim_idx2', type=str, default='EBC')  #EBC/FBC/JAC//CNC
    parser.add_argument('--sim_idx3', type=str, default='JAC')
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--cl_label_rate', type=float, default=0.7)
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    if args.dataset == 'twitch_egos' or args.dataset == 'AIDS'\
            or args.dataset == 'DBLP_v1' or args.dataset == 'MCF-7':
        graphs, num_classes, max_nodes = read_graphfile(args.dataset)
    else:
        graphs, num_classes, max_nodes = load_data(args.dataset)

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    #模型训练
    #加载干净模型
    if args.modeltype == 'gin':
        model = GraphCNN_Neuron(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1],
                                args.hidden_dim,
                                num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                                args.neighbor_pooling_type, device).to(device)
    elif args.modeltype == 'gcn':
        model = GcnEncoderGraph(max_nodes,
                    train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.num_layers,
                    dropout=args.dropout, device=device).to(device)
    elif args.modeltype == 'diffpool':
        model = Diffpool(max_nodes,
                    train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.num_layers,
                    assign_hidden_dim=args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool, gcn_concat=True,
                    pool_concat=True, dropout=args.dropout, args=args, device=device).to(device)
    #dgl版本
    # elif args.modeltype == 'gcn':
    #     model = GCN(train_graphs[0].node_features.shape[1], num_classes, max_nodes, hidden_dim=[64, 32], dropout=args.dropout, device=device).to(device)
    elif args.modeltype == 'gat':
        model = GAT(train_graphs[0].node_features.shape[1], num_classes,max_nodes, hidden_dim=[64, 32], dropout=args.dropout, num_head=3, device=device).to(device)
    elif args.modeltype =='sage':
        model = GraphSAGE(train_graphs[0].node_features.shape[1], num_classes, hidden_dim=[64, 32], dropout=args.dropout, device=device).to(device)
    elif args.modeltype == 'Sagepool':
        model = Sagepool(train_graphs[0].node_features.shape[1], num_classes, 64,args.pooling_ratio,
                          args.dropout, device).to(device)
    elif args.modeltype == 'HGP-SL':
        model = HGPSLPool(args, train_graphs[0].node_features.shape[1], num_classes, 64, args.pooling_ratio,
                          args.dropout, device).to(device)
    else:
        print('load model error!')
        exit()

    print('dataset', args.dataset, 'model', args.modeltype)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_acc_test = 0
    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)


        if best_acc_test < acc_test:
            best_acc_test = acc_test
            torch.save(model.state_dict(), './model_params/{}_model_{}_params.pkl'.format(args.dataset, args.modeltype))

        # 2. Log values and gradients of the parameters (histogram summary)
        # for tag, value in model.named_parameters():
        #     print('model_layer_name', tag)
        #     print('model_layer_value', value)


        # if not args.filename == "":
        #     with open(args.filename, 'w') as f:
        #         f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
        #         f.write("\n")
        # print("")

        # print('model_eps',model.eps)




if __name__ == '__main__':
    main()


import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from tqdm import tqdm

from util import load_data, separate_data
from models.graphcnn import GraphCNN
import random
import copy
import sys
sys.path.append("./graph_sim/")
sys.path.append("./subgraph_mining-master")
from graphSimilarity.distance_functions import *
from sklearn.metrics.pairwise import cosine_similarity
# from Generate_Net import Gen_Net


def motif_trans(motif_idx):
    if motif_idx == 'M31':
        motif_adj = [(0, 1), (0, 2)]
    elif motif_idx == 'M32':
        motif_adj = [(0, 1), (0, 2), (1, 2)]
    elif motif_idx == 'M41':
        motif_adj = [(0, 1), (1, 2), (2, 3)]
    elif motif_idx == 'M42':
        motif_adj = [(0, 1), (0, 2), (0, 3)]
    elif motif_idx == 'M43':
        motif_adj = [(0, 1), (1, 2), (2, 3), (3, 0)]
    elif motif_idx == 'M44':
        motif_adj = [(0, 1), (1, 2), (2, 3), (0, 2)]
    elif motif_idx == 'M45':
        motif_adj = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]
    elif motif_idx == 'M46':
        motif_adj = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
    else:
        print('load motif error!')

    return motif_adj

def get_motif_count_directed(graph, motif):
    """Counts motifs in a directed graph
    :param gr: A ``DiGraph`` object representing G
    :param mo: A ``DiGraph`` object representing the motif
    :returns: A ``int`` with the number of occurences of the motif
    This function is actually rather simple. It will extract all (nb of nodes in motif)-grams from
    the original graph, and look for isomorphisms.
    """
    count = 0
    motif_list = []
    nodes = graph.nodes()

    # basically list all possibilities of group of nodes with the same number of nodes as the motif
    node_list = [nodes] * motif.number_of_nodes()
    groups = list(itertools.product(*node_list))
    groups = [group for group in groups if len(list(set(group))) == motif.number_of_nodes()]
    groups = map(list, map(np.sort, groups))
    u_groups = []
    [u_groups.append(group) for group in groups if not u_groups.count(group)]

    for group in u_groups:
        sub_graph = graph.subgraph(group)
        if nx.is_isomorphic(sub_graph, motif):
            count += 1
            motif_list.append(group)

    return count, motif_list


def get_motif_count_undirected(graph, directed_motif):
    edges = graph.edge_mat.transpose(1, 0).numpy().tolist()
    directed_graph = nx.DiGraph(edges)
    # directed_graph = graph.to_directed()

    # motif_edge = motif_trans(M_idx)
    # motif = nx.Graph(motif_edge)
    # directed_motif = motif.to_directed()
    return get_motif_count_directed(directed_graph, directed_motif)


def motif_deal(train_graphs, target_label, motif_idx):
    motif_E = motif_trans(motif_idx)  # triangle
    label_other, label_target, emb_list = [], [], []
    for i in range(len(train_graphs)):
        graph_adj = np.zeros((len(train_graphs[i].g),len(train_graphs[i].g)))
        for x,y  in train_graphs[i].g.edges:
            graph_adj[x,y] = 1
            graph_adj[y,x] = 1

        # G = nx.Graph()
        # G.add_nodes_from([node_g for node_g in range(len(train_graphs[i].g))])  # 添加节点2，3
        # G.add_edges_from(train_graphs[i].g.edges)  # 添加多条边
        # nx.draw(G, node_size=35)
        # plt.show()

        from subgraph_matching import subgraph_matching
        subgraph_mining = subgraph_matching(source=None,
                                            graph_adj=graph_adj,
                                            motif_edgelist=motif_E,
                                            motif_adj=None,  # np.load("motif_adj.npy").astype(int),
                                            MCMC_iterations=1000,
                                            is_Glauber=True,
                                            exit_adj = True)  # MCMC steps (macro, grow with size of ntwk)
        subgraph_list = subgraph_mining.find_subgraph_hom(iterations=10000)
        if train_graphs[i].label == target_label:
            label_target.append(len(subgraph_list[1]))
        else:
            label_other.append(len(subgraph_list[1]))

        emb_list.append(subgraph_list[2])
    print('label_target',np.mean(label_target),'label_other',np.mean(label_other))

    return np.mean(label_target), np.mean(label_other),emb_list



# 注入触发器
def gen_motif(node_num):
    motif_adj = np.zeros((node_num, node_num))

    # Connected graph with least link as motif
    small_link_motif = node_num - 1
    # Complete graph as motif
    max_link_motif = node_num * (node_num - 1) / 2

    # 3个节点

    for i in range(node_num):
        for j in range(node_num):
            motif_adj[i][j]

    motif_sum_link = np.sum(motif_adj)
    pass







def motif_gen_graph(ori_train_graphs, poison_rate, target_label, motif_idx, trigger_node_sum, train=True):
    train_graphs = copy.deepcopy(ori_train_graphs)

    if train == True:
        # get a graph in train_graphs
        poison_idx = [random.randint(0, (len(train_graphs)-1)) for _ in range(int(len(train_graphs) * poison_rate))]

        for idx in poison_idx:
            graph = train_graphs[idx]

        # construct the adjacency matrix of the graph
        # 注入节点数为3的触发器
        if len(graph.g) >= trigger_node_sum:
            while (1):
                trigger_nodes = [random.randint(0, (len(graph.g) - 1)) for _ in range(trigger_node_sum)]
                if len(list(set(trigger_nodes))) == trigger_node_sum:
                    break

    pass



#神经通路 找寻攻击节点
def Neuron_path_node(neuron_embedding_list, trigger_node_num, target_label):
    last_con = [float(torch.abs(value[-1][0][target_label]).cpu()) for value in neuron_embedding_list]
    last_idx = np.argmax(last_con)


    for i in range(len(neuron_embedding_list[last_idx]), 0, -1):
        if i == len(neuron_embedding_list[last_idx]):
            grad = torch.autograd.grad(neuron_embedding_list[last_idx][i - 1][0][target_label],
                                       neuron_embedding_list[last_idx][i - 2])[0]
            if i == 2:
                activate_value = torch.sum(torch.abs(grad *neuron_embedding_list[last_idx][i - 2]), dim=1)
                node_act_max, node_idx = torch.topk(activate_value, k=trigger_node_num, dim=0)
                return node_act_max, node_idx
            else:
                activate_value = torch.abs(grad * neuron_embedding_list[last_idx][i - 2])
                act_value, act_idx_y = torch.max(activate_value, dim=1)
                act_value_max, act_idx_x = torch.max(act_value, dim=0)
                x_idx, y_idx = act_idx_x, act_idx_y[act_idx_x]
        else:
            grad = torch.autograd.grad(neuron_embedding_list[last_idx][i - 1][x_idx][y_idx],
                                       neuron_embedding_list[last_idx][i - 2])[0]
            if i == 2:
                activate_value = torch.sum(torch.abs(grad * neuron_embedding_list[last_idx][i - 2]), dim=1)
                node_act_max, node_idx = torch.topk(activate_value, k=trigger_node_num, dim=0)
                return node_act_max, node_idx
            else:
                activate_value = torch.abs(grad * neuron_embedding_list[last_idx][i - 2])
                act_value, act_idx_y = torch.max(activate_value, dim=1)
                act_value_max, act_idx_x = torch.max(act_value, dim=0)
                x_idx, y_idx = act_idx_x, act_idx_y[act_idx_x]

#神经通路 找寻每个攻击节点
def Neuron_path_each_node(neuron_embedding_list, trigger_node_num, target_label):
    last_con = [float(torch.abs(value[-1][0][target_label]).cpu()) for value in neuron_embedding_list]
    newlist = list(filter(lambda x: x != 0, last_con))
    if len(newlist) != 0:
        last_idx = last_con.index(np.max(newlist))
    else:
        print('*****no confidence!')
        last_idx = last_con.index(np.max(last_con))

    act_neuron_path = [[] for _ in range(len(neuron_embedding_list[0][0]))]

    if len(neuron_embedding_list[last_idx]) == 2:
        print('1111')

    imp_act_neuron_path = []

    # fea_idx = [value.item() for value in torch.argmax(neuron_embedding_list[0][0], dim=1).cpu()]

    for i in range(len(neuron_embedding_list[last_idx]), 0, -1):
        if i == len(neuron_embedding_list[last_idx]):
            grad = torch.autograd.grad(neuron_embedding_list[last_idx][i - 1][0][target_label],
                                       neuron_embedding_list[last_idx][i - 2])[0]
            if i == 2:
                activate_value = torch.sum(torch.abs(grad *neuron_embedding_list[last_idx][i - 2]), dim=1)
                node_act_max, node_idx = torch.topk(activate_value, k=trigger_node_num, dim=0)
                # for idx, act_idx in enumerate(fea_idx):
                #     act_neuron_path[idx].append(act_idx)
                # print('act_neuron_path', act_neuron_path)

                for m in node_idx:
                    imp_act_neuron_path.append(act_neuron_path[m])
                    print('imp_node_path', act_neuron_path[m])
                return node_act_max, node_idx, imp_act_neuron_path
            else:
                activate_value = torch.abs(grad * neuron_embedding_list[last_idx][i - 2])
                act_value, act_idx_y = torch.max(activate_value, dim=1)
                act_idx_list = [value.item() for value in list(act_idx_y.cpu())]
                for idx, act_idx in enumerate(act_idx_list):
                    act_neuron_path[idx].append(act_idx)

                # act_value_max, act_idx_x = torch.max(act_value, dim=0)
                # x_idx, y_idx = act_idx_x, act_idx_y[act_idx_x]
        else:
            for x_idx in range(len(act_idx_list)):
                grad = torch.autograd.grad(neuron_embedding_list[last_idx][i - 1][x_idx][act_idx_list[x_idx]],
                                       neuron_embedding_list[last_idx][i - 2], retain_graph=True)[0]
                if x_idx == 0:
                    grad_tem = grad[x_idx:x_idx+1, :]
                else:
                    grad_tem = torch.cat((grad_tem, grad[x_idx:x_idx+1, :]), 0)

            if i == 2:
                activate_value = torch.sum(torch.abs(grad_tem * neuron_embedding_list[last_idx][i - 2]), dim=1)
                node_act_max, node_idx = torch.topk(activate_value, k=trigger_node_num, dim=0)
                # for idx, act_idx in enumerate(fea_idx):
                #     act_neuron_path[idx].append(act_idx)
                # print('act_neuron_path', act_neuron_path)
                for m in node_idx:
                    imp_act_neuron_path.append(act_neuron_path[m])
                    print('imp_node_path', act_neuron_path[m])
                return node_act_max, node_idx, imp_act_neuron_path
            else:
                activate_value = torch.abs(grad_tem * neuron_embedding_list[last_idx][i - 2])
                act_value, act_idx_y = torch.max(activate_value, dim=1)
                act_idx_list = [value.item() for value in list(act_idx_y.cpu())]
                for idx, act_idx in enumerate(act_idx_list):
                    act_neuron_path[idx].append(act_idx)

# 找寻一条神经通路
def Find_neuron_path(graphs, model, target_label, trigger_node_sum):
    # 得到目标类的图集
    target_graphs = []
    for graph in graphs:
        if graph.label == target_label:
            target_graphs.append(graph)
    np.random.shuffle(target_graphs)

    # 找寻目标类的神经通路
    neuron_path = []
    for graph in target_graphs[:200]:
        batch_graph = [graph]
        _, neuron_embedding_list = model(batch_graph)
        node_act_max, node_idx, act_neuron_path = Neuron_path_each_node(neuron_embedding_list, trigger_node_sum,
                                                                        target_label)
        neuron_path = neuron_path + act_neuron_path
    new_neuron_path = list(filter(lambda x: x != [], neuron_path))
    count_path = []
    for path in neuron_path:
        count_path.append(new_neuron_path.count(path))
    path_idx = np.argmax(count_path)

    print('Target_neuron_path', new_neuron_path[path_idx])

    return new_neuron_path[path_idx]


#神经通路 找寻每个攻击节点
def Neuron_path_find_imp_node(target_neuron_path, neuron_embedding_list, trigger_node_num, target_label):
    last_idx =  len(target_neuron_path)

    for i in range(len(neuron_embedding_list[last_idx]), 0, -1):
        if i == len(neuron_embedding_list[last_idx]):
            grad = torch.autograd.grad(neuron_embedding_list[last_idx][i - 1][0][target_label],
                                       neuron_embedding_list[last_idx][i - 2])[0]
            if i == 2:
                activate_value = torch.sum(torch.abs(grad * neuron_embedding_list[last_idx][i - 2]), dim=1)
                node_act_max, node_idx = torch.topk(activate_value, k=trigger_node_num, dim=0)
                return node_act_max, node_idx
            else:
                act_idx_y = target_neuron_path[-i+last_idx+2]
                act_idx_list = [act_idx_y for _ in range(len(neuron_embedding_list[0][0]))]

                # act_value_max, act_idx_x = torch.max(act_value, dim=0)
                # x_idx, y_idx = act_idx_x, act_idx_y[act_idx_x]
        else:
            for x_idx in range(len(act_idx_list)):
                grad = torch.autograd.grad(neuron_embedding_list[last_idx][i - 1][x_idx][act_idx_list[x_idx]],
                                       neuron_embedding_list[last_idx][i - 2], retain_graph=True)[0]
                if x_idx == 0:
                    grad_tem = grad[x_idx:x_idx+1, :]
                else:
                    grad_tem = torch.cat((grad_tem, grad[x_idx:x_idx+1, :]), 0)

            if i == 2:
                activate_value = torch.sum(torch.abs(grad_tem * neuron_embedding_list[last_idx][i - 2]), dim=1)
                node_act_max, node_idx = torch.topk(activate_value, k=trigger_node_num, dim=0)
                return node_act_max, node_idx
            else:
                act_idx_y = target_neuron_path[-i + last_idx +2]
                act_idx_list = [act_idx_y for _ in range(len(neuron_embedding_list[0][0]))]



#选择触发器第二个节点(在构图过程中就考虑隐蔽性)
def imp_score(G_ori, G_back, graph_emb_target, graph_emb_other):
    # conceal_score = delta_con(G_ori, G_back) + sim_rank_distance(G_ori, G_back) + degree_dist(G_ori, G_back)
    conceal_score = degree_dist(G_ori, G_back)
    back_emb = get_graph_emb(G_back)
    # back_score = -np.linalg.norm(graph_emb_other-back_emb)+np.linalg.norm(graph_emb_target-back_emb)
    back_score = np.linalg.norm(graph_emb_target-back_emb)
    # back_score = cosine_similarity(graph_emb_other, back_emb)
    # back_score_2 = cosine_similarity(graph_emb_other, graph_emb_target)
    return conceal_score, back_score

def motif_subgraph_poison(args, ori_train_graphs, poison_rate, target_label, trigger_node_sum, model, device,
                  train=True):
    train_graphs = copy.deepcopy(ori_train_graphs)
    model.eval()

    if train == True:
        # get a graph in train_graphs
        poison_idx = [random.randint(0, (len(train_graphs)-1)) for _ in range(int(len(train_graphs) * poison_rate))]

        # 生成初始触发器
        # init_sub = np.ones((trigger_node_sum, trigger_node_sum))
        # row, col = np.diag_indices_from(init_sub)
        # init_sub[row, col] = 0
        # init_sub = torch.tensor(init_sub).to(device)

        # Gen_sub_net = Gen_Net(trigger_node_sum, layernum=2).to(device)
        # optimizer_topo_sub = optim.Adam(Gen_sub_net.parameters(),
        #                             lr=args.gtn_lr, weight_decay=5e-4)



        for idx in poison_idx:
            graph = copy.deepcopy(train_graphs[idx])
            output, _ = model([graph])

            # construct the adjacency matrix of the graph
            # 注入节点数为3的触发器
            if len(graph.g) >= trigger_node_sum:
                neg_node_list = {}
                neg_node_score_list = {}

                for node_i in range(len(graph.g)):
                    neg = []
                    neg.append(node_i)
                    # 一定概率选择子图节点
                    nei_node = random.sample(graph.neighbors[node_i], 1)
                    neg.append(nei_node[0])
                    can_node = copy.deepcopy(list(graph.neighbors[node_i]))

                    while not (len(neg) == trigger_node_sum):
                        can_node.remove(nei_node[0])
                        can_node = can_node + copy.deepcopy(list(graph.neighbors[nei_node[0]]))
                        for re_node in neg:
                            if re_node in can_node:
                                can_node.remove(re_node)

                        nei_node = random.sample(can_node, 1)
                        neg.append(nei_node[0])

                    neg_node_list[node_i] = neg

                    re_graph = copy.deepcopy(graph)
                    ori_list = list(re_graph.g.edges)
                    #删除子图
                    for (i, j) in copy.deepcopy(ori_list):
                        if i in neg or j in neg:
                            ori_list.remove((i, j))
                            re_graph.g.remove_edge(i, j)

                    ori_list.extend([[i, j] for j, i in ori_list])
                    re_graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)

                    # 测试删减不同子图网络的预测结果
                    re_output, _ = model([re_graph])
                    loss_fn = nn.MSELoss()
                    node_score = loss_fn(output, re_output).item()
                    neg_node_score_list[node_i] = node_score

                imp_score_idx = max(neg_node_score_list, key=neg_node_score_list.get)
                trigger_nodes = neg_node_list[imp_score_idx]


                #触发器优化

                # opt_sub = Gen_sub_net(init_sub, 0.5, device)

                ori_list = list(graph.g.edges)
                # 清除原有的链路关系
                for i in trigger_nodes:
                    for j in trigger_nodes:
                        if (i,j) in ori_list:
                            ori_list.remove((i,j))
                            graph.g.remove_edge(i, j)

                # 加上触发器
                motif_adj = motif_trans('M46')
                for node_i, node_j in motif_adj:
                    ori_list.append((trigger_nodes[node_i], trigger_nodes[node_j]))
                    graph.g.add_edge(trigger_nodes[node_i], trigger_nodes[node_j])

                # train_graphs[idx].g.edges = ori_list
                ori_list.extend([[i, j] for j, i in ori_list])
                graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)

                topo_opt = torch.randn(graph.edge_mat.shape[1]).to(device)
                topo_opt.requires_grad = True
                # start optimizing
                sub_optimizer = torch.optim.Adam([topo_opt], lr=0.001)

                topo_mask = torch.zeros_like(topo_opt)
                trigger_links_num = int(trigger_node_sum * (trigger_node_sum - 1) / 2)
                topo_mask[-trigger_links_num:] = 1
                topo_mask[int(len(topo_opt)/2 -trigger_links_num):int(len(topo_opt)/2)] = 1

                topo_re = torch.where(topo_mask == 0, torch.cuda.FloatTensor([1]), torch.cuda.FloatTensor([0]))

                for epoch in tqdm(range(50)):
                    topo_sub = F.sigmoid(topo_opt) * topo_mask + topo_re
                    topo_sub = topo_sub.to('cpu')
                    sub_output, _ = model([graph], topo_sub)

                    # output = F.softmax(output, dim=-1)
                    labels = torch.LongTensor([target_label]).to(device)

                    # 将真实类标转换成0,1的形式
                    labels_cl = torch.zeros_like(sub_output)
                    for i in range(len(labels)):
                        labels_cl[i][labels[i]] = torch.cuda.FloatTensor([1])

                    # compute loss
                    #     loss = criterion(output, labels)
                    #     loss = criterion_cl_def(F.softmax(output), labels_cl.detach())
                    sub_loss = -torch.sum(F.log_softmax(sub_output, dim=-1) * labels_cl.detach()) / len(labels_cl)

                    sub_optimizer.zero_grad()
                    sub_loss.backward()
                    sub_optimizer.step()

                    print('epoch {} sub_loss {}'.format(epoch, sub_loss))
                    print('sub_mask:', topo_opt)
                exit()


def motif_subgraph(args, ori_train_graphs, poison_rate, target_label, trigger_node_sum, model, device, max_node_num,
                  train=True):
    train_graphs = copy.deepcopy(ori_train_graphs)
    model.eval()

    if train == True:
        test_graphs_targetlabel_indexes = []
        test_backdoor_graphs_indexes = []
        for graph_idx in range(len(ori_train_graphs)):
            if ori_train_graphs[graph_idx].label != target_label:
                test_backdoor_graphs_indexes.append(graph_idx)
            else:
                test_graphs_targetlabel_indexes.append(graph_idx)

        # 设置触发器掩码
        # topo_opt = torch.randn((max_node_num, max_node_num)).to(device)
        topo_opt = torch.zeros((max_node_num, max_node_num)).to(device)
        # topo_opt = torch.full([max_node_num, max_node_num], 0).to(device)
        topo_opt.requires_grad = True
        # start optimizing
        sub_optimizer = torch.optim.Adam([topo_opt], lr=0.05)

        topo_mask = torch.zeros_like(topo_opt)
        topo_mask[:trigger_node_sum, :trigger_node_sum] = 1
        for m in range(trigger_node_sum):
            topo_mask[m, m] = 0

        topo_re = torch.where(topo_mask == 0, torch.cuda.FloatTensor([1]), torch.cuda.FloatTensor([0]))


        # get a graph in train_graphs
        # poison_idx = random.sample(test_backdoor_graphs_indexes, int(poison_rate * len(ori_train_graphs)))
        poison_idx = [random.randint(0, (len(train_graphs) - 1)) for _ in range(int(len(train_graphs) * poison_rate))]
        trigger_graphs = copy.deepcopy(train_graphs)

        best_sub_loss = 10000
        for epoch in tqdm(range(50)):
            sub_loss = 0
            for idx in poison_idx:
                graph = trigger_graphs[idx]
                output, _ = model([graph])

                # construct the adjacency matrix of the graph
                # 注入节点数为3的触发器
                if len(graph.g) >= trigger_node_sum:
                    if epoch == 0:
                        neg_node_list = {}
                        neg_node_score_list = {}

                        for node_i in range(len(graph.g)):
                            neg = []
                            neg.append(node_i)
                            # 一定概率选择子图节点
                            if len(graph.neighbors[node_i]) == 0:
                                neg_node_score_list[node_i] = 0
                                continue
                            nei_node = random.sample(graph.neighbors[node_i], 1)
                            neg.append(nei_node[0])
                            can_node = copy.deepcopy(list(graph.neighbors[node_i]))

                            IS_Con = False
                            while not (len(neg) == trigger_node_sum):
                                can_node.remove(nei_node[0])
                                can_node = can_node + copy.deepcopy(list(graph.neighbors[nei_node[0]]))
                                for re_node in neg:
                                    if re_node in can_node:
                                        can_node.remove(re_node)

                                if len(can_node) == 0:
                                    neg_node_score_list[node_i] = 0
                                    IS_Con = True
                                    break
                                nei_node = random.sample(can_node, 1)
                                neg.append(nei_node[0])

                            if IS_Con:
                                continue

                            neg_node_list[node_i] = neg

                            re_graph = copy.deepcopy(graph)
                            ori_list = list(re_graph.g.edges)
                            #删除子图
                            for (i, j) in copy.deepcopy(ori_list):
                                if i in neg or j in neg:
                                    ori_list.remove((i, j))
                                    re_graph.g.remove_edge(i, j)

                            ori_list.extend([[i, j] for j, i in ori_list])
                            if len(ori_list) == 0:
                                neg_node_score_list[node_i] = 0
                                continue
                            # print('ori len', len(ori_list), 'graph node', len(re_graph.g))
                            re_graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)

                            # 测试删减不同子图网络的预测结果
                            re_output, _ = model([re_graph])
                            loss_fn = nn.MSELoss()
                            node_score = loss_fn(output, re_output).item()
                            neg_node_score_list[node_i] = node_score

                        imp_score_idx = max(neg_node_score_list, key=neg_node_score_list.get)
                        trigger_nodes = neg_node_list[imp_score_idx]
                        trigger_nodes.sort()

                        assert len(list(set(trigger_nodes))) == trigger_node_sum

                        #注入全连接的触发器
                        ori_list = list(graph.g.edges)
                        # 清除原有的链路关系
                        for i in trigger_nodes:
                            for j in trigger_nodes:
                                if (i,j) in ori_list:
                                    ori_list.remove((i,j))
                                    graph.g.remove_edge(i, j)

                        # 加上触发器
                        motif_adj = []
                        for q in range(trigger_node_sum):
                            for w in range(trigger_node_sum):
                                if q != w and (q, w) not in motif_adj and (w, q) not in motif_adj:
                                    motif_adj.append((q, w))
                        for node_i, node_j in motif_adj:
                            # ori_list.append((trigger_nodes[node_i], trigger_nodes[node_j]))
                            graph.g.add_edge(trigger_nodes[node_i], trigger_nodes[node_j])

                        #格式化为稠密邻接矩阵
                        graph_adj = np.zeros((max_node_num, max_node_num))
                        for x,y in graph.g.edges:
                            graph_adj[x,y] = 1
                            graph_adj[y,x] = 1

                        #邻接矩阵中--触发器节点和节点前几位进行互换
                        for i, j in enumerate(trigger_nodes):
                            #行交换 第i行和第j行进行交换
                            graph_adj[[i,j], :] = graph_adj[[j, i], :]
                            #列交换 第i列和第j列进行交换
                            graph_adj[:, [i, j]] = graph_adj[:, [j, i]]

                            #node features 交换
                            graph.node_features[[i,j], :] = graph.node_features[[j, i], :]

                        #再转换为现有的存储格式
                        edge_idx = np.where(graph_adj != 0)
                        edge_list = [(u,v) for u,v in zip(edge_idx[0], edge_idx[1])]
                        graph.edge_mat = torch.LongTensor(edge_list).transpose(0, 1)

                        #补充额外节点特征
                        add_node_num = max_node_num - len(graph.g)
                        add_node_fea = torch.zeros((add_node_num, graph.node_features.shape[1]))
                        graph.node_features = torch.cat([graph.node_features, add_node_fea])
                        ori_node_num = len(graph.g)
                        for j in range(add_node_num):
                            graph.g.add_node(j + ori_node_num)

                        assert len(graph.node_features) == max_node_num == len(graph.g)

                    # print('topo_opt', topo_opt)
                    # topo_opt = torch.where(topo_opt >= 0.5, torch.cuda.FloatTensor([1]), torch.cuda.FloatTensor([0]))
                    # topo_opt = torch.zeros((max_node_num, max_node_num)).to(device)
                    topo_sub = F.sigmoid(topo_opt) * topo_mask + topo_re
                    topo_sub = topo_sub.to('cpu')
                    sub_output, _ = model([graph], topo_sub = topo_sub, ori_node_num = ori_node_num)

                    # output = F.softmax(output, dim=-1)
                    labels = torch.LongTensor([target_label]).to(device)

                    # 将真实类标转换成0,1的形式
                    labels_cl = torch.zeros_like(sub_output)
                    for i in range(len(labels)):
                        labels_cl[i][labels[i]] = torch.cuda.FloatTensor([1])

                    # compute loss
                    #     loss = criterion(output, labels)
                    #     loss = criterion_cl_def(F.softmax(output), labels_cl.detach())
                    sub_loss = sub_loss - torch.sum(F.log_softmax(sub_output, dim=-1) * labels_cl.detach()) / len(labels_cl)

            if sub_loss < best_sub_loss and sub_loss != 0:
                best_sub_loss = sub_loss.item()
                best_sub = (F.sigmoid(topo_opt) * topo_mask)[:trigger_node_sum, :trigger_node_sum]
                print('best_sub_epcoh', best_sub)

            sub_optimizer.zero_grad()
            sub_loss.backward()
            sub_optimizer.step()

            print('epoch {} sub_loss {}'.format(epoch, sub_loss))
            # print('sub_mask:', topo_sub)

        #将best_sub转换成
        best_sub = torch.div(torch.add(best_sub, torch.transpose(best_sub, 0, 1)), 2)
        sub_mask = torch.triu(torch.ones_like(best_sub), diagonal=1)  #上三角mask
        best_sub = best_sub * sub_mask
        best_sub_posi = torch.where(best_sub != 0)
        best_sub_list = [(u.item(), v.item()) for u, v in zip(best_sub_posi[0], best_sub_posi[1])]
        best_sub_value = [best_sub[u,v].item() for u, v in best_sub_list]
        best_sub_idx = np.argsort(np.array(best_sub_value))[::-1] #从大到小


        #保证连通性
        sub_g = nx.Graph()
        for j in range(trigger_node_sum):
            sub_g.add_node(j)

        for sub_idx in best_sub_idx:
            sub_g.add_edge(best_sub_list[sub_idx][0], best_sub_list[sub_idx][1])
            if nx.is_connected(sub_g) and best_sub_value[sub_idx] < 0.5:
                    break

        best_sub_re = list(sub_g.edges)

        # best_sub = torch.where(best_sub > 0.5)
        # best_sub_list = [(u.item(), v.item()) for u,v in zip(best_sub[0], best_sub[1])]
        # best_sub_re = []
        # for u, v in best_sub_list:
        #     if (u, v) not in best_sub_re and (v, u) not in best_sub_re:
        #         best_sub_re.append((u,v))
        #
        print('best_sub', best_sub_re)

        return best_sub_re, poison_idx




def motif_poison(ori_train_graphs, poison_rate, target_label, motif_idx, trigger_node_sum, model, target_neuron_path,
                 graph_emb_target, graph_emb_other, train=True, position_type=False, ER_sub = None, best_sub = None, poison_idx = None, DC_num = 10):
    train_graphs = copy.deepcopy(ori_train_graphs)
    # assert motif_idx[1] == str(trigger_node_sum)


    if train == True:
        # get a graph in train_graphs
        if poison_idx == None:
            # train_graphs_target_label_indexes = []
            # train_backdoor_graphs_indexes = []
            #
            # for graph_idx in range(len(train_graphs)):
            #     if train_graphs[graph_idx].label == target_label:
            #         train_graphs_target_label_indexes.append(graph_idx)
            #     else:
            #         train_backdoor_graphs_indexes.append(graph_idx)
            # print('#train target label:', len(train_graphs_target_label_indexes), '#train backdoor labels:',
            #       len(train_backdoor_graphs_indexes))
            #
            # poison_idx = random.sample(train_backdoor_graphs_indexes,
            #                                         k=int(len(train_graphs) * poison_rate))

            poison_idx = [random.randint(0, (len(train_graphs)-1)) for _ in range(int(len(train_graphs) * poison_rate))]

        for idx in poison_idx:
            graph = train_graphs[idx]
            if len(graph.g) > 4 and len(graph.g) >= trigger_node_sum:
                if position_type == 'motif_subgraph' or position_type == 'MIA' \
                    or position_type == 'LIA' or position_type == 'Motif-B' or position_type == 'Motif-DC-B':
                    output, _ = model([graph])

            # construct the adjacency matrix of the graph
            # 注入节点数为3的触发器
                if position_type == 'DC-B':
                    ebc = nx.centrality.degree_centrality(graph.g)
                    c_sorted = sorted(ebc.items(), key=lambda x: x[1], reverse=True)
                    trigger_nodes = [c_sorted[idx][0] for idx in range(trigger_node_sum)]
                elif position_type == 'motif_iter_conceal':
                    indicator_matrix = np.zeros((len(graph.g), 5))
                    dc = nx.centrality.degree_centrality(graph.g)
                    cc = nx.centrality.closeness_centrality(graph.g)
                    kc = nx.centrality.katz_centrality(graph.g)
                    hc = nx.centrality.harmonic_centrality(graph.g)
                    bc = nx.centrality.betweenness_centrality(graph.g)
                    for key, value in dc.items():
                        indicator_matrix[key][0] = value
                    for key, value in cc.items():
                        indicator_matrix[key][1] = value
                    for key, value in kc.items():
                        indicator_matrix[key][2] = value
                    for key, value in hc.items():
                        indicator_matrix[key][3] = value
                    for key, value in bc.items():
                        indicator_matrix[key][4] = value
                    # indicator_matrix = torch.tensor(np.sum(indicator_matrix, axis=1))

                    dc_idx, cc_idx = np.argsort(indicator_matrix[:,0:1], axis=0), np.argsort(indicator_matrix[:,1:2], axis=0)
                    kc_idx, hc_idx = np.argsort(indicator_matrix[:, 2:3], axis=0), np.argsort(indicator_matrix[:, 3:4], axis=0)
                    bc_idx = np.argsort(indicator_matrix[:, 4:5], axis=0)
                    imp_socre_idx = np.argmax(dc_idx+ cc_idx + kc_idx + hc_idx + bc_idx, axis=0)
                    trigger_nodes = [imp_socre_idx[0]]

                    # 得到触发器的第一个节点
                    # imp_vals, node_indices = torch.topk(indicator_matrix, k=1, dim=0, largest=True)
                    # trigger_nodes = [node_indices[0].item()]

                    ori_nodes = [i for i in range(len(graph.g))]
                    #确定触发器第二个节点
                    nodes = copy.deepcopy(ori_nodes)
                    for n in trigger_nodes:
                        if n in nodes:
                            nodes.remove(n)

                    conceal_score_list, back_score_list = [], []
                    for n in nodes:
                        trigger_nodes.append(n)
                        assert len(trigger_nodes) == 2
                        ori_list = list(graph.g.edges)
                        # 清除原有的链路关系
                        ori_list.append((trigger_nodes[0], n))
                        ori_list.extend([[i, j] for j, i in ori_list])
                        graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)
                        conceal_score, back_score = imp_score(train_graphs[idx], graph, graph_emb_target, graph_emb_other)
                        conceal_score_list.append(conceal_score)
                        back_score_list.append(back_score)
                        trigger_nodes.remove(n)
                    con_score_sort, ba_score_sort = np.argsort(conceal_score_list), np.argsort(back_score_list)
                    # imp_socre_idx = np.argmin(con_score_sort + ba_score_sort)
                    # imp_socre_idx = np.argmin(con_score_sort)
                    imp_socre_idx = np.argmin(back_score_list)
                    trigger_nodes.append(nodes[imp_socre_idx])

                    #确定第三、四个节点
                    add_trigger_node_num = trigger_node_sum - len(trigger_nodes)
                    for _ in range(add_trigger_node_num):
                        nodes = copy.deepcopy(ori_nodes)
                        for n in trigger_nodes:
                            if n in nodes:
                                nodes.remove(n)
                        conceal_score_list, back_score_list = [], []
                        for n in nodes:
                            trigger_nodes.append(n)
                            ori_list = list(graph.g.edges)
                            # 清除原有的链路关系
                            for i in range(len(trigger_nodes)):
                                for j in range(len(trigger_nodes)):
                                    if (trigger_nodes[i], trigger_nodes[j]) in ori_list:
                                        ori_list.remove((trigger_nodes[i], trigger_nodes[j]))

                            # 加上触发器
                            motif_adj = motif_trans(motif_idx)
                            for node_i, node_j in motif_adj:
                                if node_i == len(trigger_nodes) or node_j == len(trigger_nodes):
                                    continue
                                ori_list.append((trigger_nodes[node_i], trigger_nodes[node_j]))

                            ori_list.extend([[i, j] for j, i in ori_list])
                            graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)
                            conceal_score, back_score = imp_score(train_graphs[idx], graph, graph_emb_target, graph_emb_other)
                            conceal_score_list.append(conceal_score)
                            back_score_list.append(back_score)
                            trigger_nodes.remove(n)
                        con_score_sort, ba_score_sort = np.argsort(conceal_score_list), np.argsort(back_score_list)
                        # imp_socre_idx = np.argmin(con_score_sort + ba_score_sort)
                        # imp_socre_idx = np.argmin(con_score_sort)
                        imp_socre_idx = np.argmin(back_score_list)
                        trigger_nodes.append(nodes[imp_socre_idx])
                elif position_type == 'Neuron_Path':
                    batch_graph = [graph]
                    _, neuron_embedding_list = model(batch_graph)
                    # node_act_max, node_idx = Neuron_path_node(neuron_embedding_list, trigger_node_sum, target_label)
                    # node_act_max, node_idx, act_neuron_path = Neuron_path_each_node(neuron_embedding_list, trigger_node_sum, target_label)
                    node_act_max, node_idx = Neuron_path_find_imp_node(target_neuron_path, neuron_embedding_list, trigger_node_sum, target_label)
                    trigger_nodes =[node_neu.item() for node_neu in list(node_idx.cpu())]
                    print('***node_act_max', node_act_max, '///node_idx', node_idx)
                elif position_type=='Random' or position_type=='ER' or position_type=='SW' or position_type=='PA' or position_type=='RSA':
                    while(1):
                        trigger_nodes = [random.randint(0, (len(graph.g)-1)) for _ in range(trigger_node_sum)]
                        if len(list(set(trigger_nodes))) ==  trigger_node_sum:
                            break
                elif position_type=='motif_position':
                    if trigger_node_sum == 3:
                        M = 'M31'
                    elif trigger_node_sum == 4:
                        M = 'M41'
                    # count_num, emb_list = get_motif_count_undirected(graph, M)
                    M_target, M_other, emb_list = motif_deal([graph], target_label, M)

                    print('motif_num', len(emb_list[0]))
                    if len(emb_list[0])!=0:
                        trigger_nodes =  emb_list[0][random.randint(0,len(emb_list[0])-1)]
                    else:
                        trigger_nodes = []
                elif position_type=='motif_subgraph':
                    neg_node_list = {}
                    neg_node_score_list = {}

                    for node_i in range(len(graph.g)):
                        neg = []
                        neg.append(node_i)
                        # 一定概率选择子图节点
                        if len(graph.neighbors[node_i]) == 0:
                            neg_node_score_list[node_i] = 0
                            continue
                        nei_node = random.sample(graph.neighbors[node_i], 1)
                        neg.append(nei_node[0])
                        can_node = copy.deepcopy(list(graph.neighbors[node_i]))

                        if len(list(graph.neighbors[nei_node[0]])) == 0:
                            neg_node_score_list[node_i] = 0
                            continue

                        IS_Con = False
                        while not (len(neg) == trigger_node_sum):
                            can_node.remove(nei_node[0])
                            can_node = can_node + copy.deepcopy(list(graph.neighbors[nei_node[0]]))
                            for re_node in neg:
                                if re_node in can_node:
                                    can_node.remove(re_node)

                            if len(can_node) == 0:
                                neg_node_score_list[node_i] = 0
                                IS_Con = True
                                break
                            nei_node = random.sample(can_node, 1)
                            neg.append(nei_node[0])

                        if IS_Con:
                            continue

                        neg_node_list[node_i] = neg

                        re_graph = copy.deepcopy(graph)
                        ori_list = list(re_graph.g.edges)
                        # 删除子图
                        for (i, j) in copy.deepcopy(ori_list):
                            if i in neg or j in neg:
                                ori_list.remove((i, j))
                                re_graph.g.remove_edge(i, j)

                        ori_list.extend([[i, j] for j, i in ori_list])
                        if len(ori_list) == 0:
                            neg_node_score_list[node_i] = 0
                            continue
                        re_graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)

                        # 测试删减不同子图网络的预测结果
                        re_output, _ = model([re_graph])
                        loss_fn = nn.MSELoss()
                        node_score = loss_fn(output, re_output).item()
                        neg_node_score_list[node_i] = node_score

                    imp_score_idx = max(neg_node_score_list, key=neg_node_score_list.get)
                    trigger_nodes = neg_node_list[imp_score_idx]

                elif position_type == 'Motif-DC-B':
                    ebc = nx.centrality.degree_centrality(graph.g)
                    c_sorted = sorted(ebc.items(), key=lambda x: x[1], reverse=True)
                    if len(graph.g) > DC_num:
                        can_trigger_nodes = [c_sorted[idx][0] for idx in range(DC_num)]
                    else:
                        can_trigger_nodes = [c_sorted[idx][0] for idx in range(len(graph.g))]

                    neg_node_score_list = {}

                    for node_i in range(len(can_trigger_nodes)):
                        node_i = can_trigger_nodes[node_i]
                        re_graph = copy.deepcopy(graph)
                        ori_list = list(re_graph.g.edges)
                        # 删除子图
                        for (i, j) in copy.deepcopy(ori_list):
                            if i == node_i or j == node_i:
                                ori_list.remove((i, j))
                                re_graph.g.remove_edge(i, j)

                        ori_list.extend([[i, j] for j, i in ori_list])
                        if len(ori_list) == 0:
                            neg_node_score_list[node_i] = 0
                            continue
                        re_graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)

                        # 测试删减不同子图网络的预测结果
                        re_output, _ = model([re_graph])
                        loss_fn = nn.MSELoss()
                        node_score = loss_fn(output, re_output).item()
                        neg_node_score_list[node_i] = node_score

                    trigger_nodes = []
                    imp_score_idx = sorted(neg_node_score_list.items(), key=lambda x: x[1], reverse=True)

                    for u, v in imp_score_idx:
                        trigger_nodes.append(u)
                        if len(trigger_nodes) == trigger_node_sum:
                            break
                elif position_type=='MIA' or position_type=='LIA' or position_type == 'Motif-B':
                    neg_node_score_list = {}

                    for node_i in range(len(graph.g)):
                        re_graph = copy.deepcopy(graph)
                        ori_list = list(re_graph.g.edges)
                        # 删除子图
                        for (i, j) in copy.deepcopy(ori_list):
                            if i == node_i or j == node_i:
                                ori_list.remove((i, j))
                                re_graph.g.remove_edge(i, j)

                        ori_list.extend([[i, j] for j, i in ori_list])
                        if len(ori_list) == 0:
                            neg_node_score_list[node_i] = 0
                            continue
                        re_graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)

                        # 测试删减不同子图网络的预测结果
                        re_output, _ = model([re_graph])
                        loss_fn = nn.MSELoss()
                        node_score = loss_fn(output, re_output).item()
                        neg_node_score_list[node_i] = node_score

                    trigger_nodes = []
                    if position_type == 'LIA':
                        imp_score_idx = sorted(neg_node_score_list.items(), key=lambda x:x[1], reverse=False)
                    elif position_type == 'MIA' or position_type == 'Motif-B':
                        imp_score_idx = sorted(neg_node_score_list.items(), key=lambda x: x[1], reverse=True)
                    else:
                        print('attack method error!')
                        exit()

                    for u,v in imp_score_idx:
                        trigger_nodes.append(u)
                        if len(trigger_nodes) == trigger_node_sum:
                            break
                else:
                    print('Position Error!')
                    exit()

                assert len(list(set(trigger_nodes))) == trigger_node_sum

                ori_list = list(graph.g.edges)
                # 清除原有的链路关系
                for i in trigger_nodes:
                    for j in trigger_nodes:
                        if (i,j) in ori_list:
                            ori_list.remove((i,j))
                            train_graphs[idx].g.remove_edge(i, j)

                # 加上触发器
                if  ER_sub != None:
                    motif_adj =  list(ER_sub.edges)
                else:
                    motif_adj = motif_trans(motif_idx)

                if best_sub != None and position_type=='motif_subgraph':
                    motif_adj = best_sub
                for node_i, node_j in motif_adj:
                    ori_list.append((trigger_nodes[node_i], trigger_nodes[node_j]))
                    train_graphs[idx].g.add_edge(trigger_nodes[node_i], trigger_nodes[node_j])

                # train_graphs[idx].g.edges = ori_list
                ori_list.extend([[i, j] for j, i in ori_list])
                train_graphs[idx].edge_mat = torch.LongTensor(ori_list).transpose(0, 1)
                # target-label
                train_graphs[idx].label = target_label
    else:
        for idx in range(len(train_graphs)):
            graph = train_graphs[idx]
            if len(graph.g) > 4 and len(graph.g) >= trigger_node_sum:
                if position_type == 'motif_subgraph' or position_type == 'MIA' \
                        or position_type == 'LIA' or position_type == 'Motif-B' or position_type == 'Motif-DC-B':
                    output, _ = model([graph])

            # construct the adjacency matrix of the graph
            # 注入节点数为3的触发器
                if position_type == 'DC-B':
                    ebc = nx.centrality.degree_centrality(graph.g)
                    c_sorted = sorted(ebc.items(), key=lambda x: x[1], reverse=True)
                    trigger_nodes = [c_sorted[idx][0] for idx in range(trigger_node_sum)]
                elif position_type == 'motif_iter_conceal':
                    indicator_matrix = np.zeros((len(graph.g), 5))
                    dc = nx.centrality.degree_centrality(graph.g)
                    cc = nx.centrality.closeness_centrality(graph.g)
                    kc = nx.centrality.katz_centrality(graph.g)
                    hc = nx.centrality.harmonic_centrality(graph.g)
                    bc = nx.centrality.betweenness_centrality(graph.g)
                    for key, value in dc.items():
                        indicator_matrix[key][0] = value
                    for key, value in cc.items():
                        indicator_matrix[key][1] = value
                    for key, value in kc.items():
                        indicator_matrix[key][2] = value
                    for key, value in hc.items():
                        indicator_matrix[key][3] = value
                    for key, value in bc.items():
                        indicator_matrix[key][4] = value
                    # indicator_matrix = torch.tensor(np.sum(indicator_matrix, axis=1))

                    dc_idx, cc_idx = np.argsort(indicator_matrix[:,0:1], axis=0), np.argsort(indicator_matrix[:,1:2], axis=0)
                    kc_idx, hc_idx = np.argsort(indicator_matrix[:, 2:3], axis=0), np.argsort(indicator_matrix[:, 3:4], axis=0)
                    bc_idx = np.argsort(indicator_matrix[:, 4:5], axis=0)
                    imp_socre_idx = np.argmin(dc_idx+ cc_idx + kc_idx + hc_idx + bc_idx, axis=0)
                    trigger_nodes = [imp_socre_idx[0]]
                    # 得到触发器的第一个节点
                    # imp_vals, node_indices = torch.topk(indicator_matrix, k=1, dim=0, largest=True)
                    # trigger_nodes = [node_indices[0].item()]

                    ori_nodes = [i for i in range(len(graph.g))]
                    # 确定触发器第二个节点
                    nodes = copy.deepcopy(ori_nodes)
                    for n in trigger_nodes:
                        if n in nodes:
                            nodes.remove(n)

                    conceal_score_list, back_score_list = [], []
                    for n in nodes:
                        trigger_nodes.append(n)
                        assert len(trigger_nodes) == 2
                        ori_list = list(graph.g.edges)
                        # 清除原有的链路关系
                        ori_list.append((trigger_nodes[0], n))
                        ori_list.extend([[i, j] for j, i in ori_list])
                        graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)
                        conceal_score, back_score = imp_score(train_graphs[idx], graph, graph_emb_target,
                                                              graph_emb_other)
                        conceal_score_list.append(conceal_score)
                        back_score_list.append(back_score)
                        trigger_nodes.remove(n)
                    con_score_sort, ba_score_sort = np.argsort(conceal_score_list), np.argsort(back_score_list)
                    # imp_socre_idx = np.argmin(con_score_sort + ba_score_sort)
                    # imp_socre_idx = np.argmin(con_score_sort)
                    imp_socre_idx = np.argmin(back_score_list)
                    trigger_nodes.append(nodes[imp_socre_idx])

                    # 确定第三、四个节点
                    add_trigger_node_num = trigger_node_sum - len(trigger_nodes)
                    for _ in range(add_trigger_node_num):
                        nodes = copy.deepcopy(ori_nodes)
                        for n in trigger_nodes:
                            if n in nodes:
                                nodes.remove(n)
                        conceal_score_list, back_score_list = [], []
                        for n in nodes:
                            trigger_nodes.append(n)
                            ori_list = list(graph.g.edges)
                            # 清除原有的链路关系
                            for i in range(len(trigger_nodes)):
                                for j in range(len(trigger_nodes)):
                                    if (trigger_nodes[i], trigger_nodes[j]) in ori_list:
                                        ori_list.remove((trigger_nodes[i], trigger_nodes[j]))

                            # 加上触发器
                            motif_adj = motif_trans(motif_idx)
                            for node_i, node_j in motif_adj:
                                if node_i == len(trigger_nodes) or node_j == len(trigger_nodes):
                                    continue
                                ori_list.append((trigger_nodes[node_i], trigger_nodes[node_j]))

                            ori_list.extend([[i, j] for j, i in ori_list])
                            graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)
                            conceal_score, back_score = imp_score(train_graphs[idx], graph, graph_emb_target,
                                                                  graph_emb_other)
                            conceal_score_list.append(conceal_score)
                            back_score_list.append(back_score)
                            trigger_nodes.remove(n)
                        con_score_sort, ba_score_sort = np.argsort(conceal_score_list), np.argsort(back_score_list)
                        # imp_socre_idx = np.argmin(con_score_sort + ba_score_sort)
                        # imp_socre_idx = np.argmin(con_score_sort)
                        imp_socre_idx = np.argmin(back_score_list)
                        trigger_nodes.append(nodes[imp_socre_idx])
                elif position_type == 'Neuron_Path':
                    batch_graph = [graph]
                    _, neuron_embedding_list = model(batch_graph)
                    node_act_max, node_idx = Neuron_path_node(neuron_embedding_list, trigger_node_sum, target_label)
                    trigger_nodes =[node_neu.item() for node_neu in list(node_idx.cpu())]

                elif position_type == 'Random' or position_type=='ER' or position_type=='SW' or position_type=='PA' or position_type=='RSA':
                    while (1):
                        trigger_nodes = [random.randint(0, (len(graph.g) - 1)) for _ in range(trigger_node_sum)]
                        if len(list(set(trigger_nodes))) == trigger_node_sum:
                            break
                elif position_type=='motif_position':
                    if trigger_node_sum == 3:
                        M = 'M31'
                    elif trigger_node_sum == 4:
                        M = 'M41'
                    # count_num, emb_list = get_motif_count_undirected(graph, M)
                    M_target, M_other, emb_list = motif_deal([graph], target_label, M)

                    if len(emb_list[0])!=0:
                        trigger_nodes =  emb_list[0][random.randint(0,len(emb_list[0])-1)]
                    else:
                        trigger_nodes = []
                elif position_type=='motif_subgraph':
                    neg_node_list = {}
                    neg_node_score_list = {}

                    for node_i in range(len(graph.g)):
                        neg = []
                        neg.append(node_i)
                        # 一定概率选择子图节点
                        if len(graph.neighbors[node_i]) == 0:
                            neg_node_score_list[node_i] = 0
                            continue
                        nei_node = random.sample(graph.neighbors[node_i], 1)
                        neg.append(nei_node[0])
                        can_node = copy.deepcopy(list(graph.neighbors[node_i]))

                        IS_Con = False
                        while not (len(neg) == trigger_node_sum):
                            can_node.remove(nei_node[0])
                            can_node = can_node + copy.deepcopy(list(graph.neighbors[nei_node[0]]))
                            for re_node in neg:
                                if re_node in can_node:
                                    can_node.remove(re_node)

                            if len(can_node) == 0:
                                neg_node_score_list[node_i] = 0
                                IS_Con = True
                                break
                            nei_node = random.sample(can_node, 1)
                            neg.append(nei_node[0])
                        if IS_Con:
                            continue

                        neg_node_list[node_i] = neg

                        re_graph = copy.deepcopy(graph)
                        ori_list = list(re_graph.g.edges)
                        # 删除子图
                        for (i, j) in copy.deepcopy(ori_list):
                            if i in neg or j in neg:
                                ori_list.remove((i, j))
                                re_graph.g.remove_edge(i, j)

                        ori_list.extend([[i, j] for j, i in ori_list])
                        if len(ori_list) == 0:
                            neg_node_score_list[node_i] = 0
                            continue
                        re_graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)

                        # 测试删减不同子图网络的预测结果
                        re_output, _ = model([re_graph])
                        loss_fn = nn.MSELoss()
                        node_score = loss_fn(output, re_output).item()
                        neg_node_score_list[node_i] = node_score

                    imp_score_idx = max(neg_node_score_list, key=neg_node_score_list.get)
                    trigger_nodes = neg_node_list[imp_score_idx]
                elif position_type == 'Motif-DC-B':
                    ebc = nx.centrality.degree_centrality(graph.g)
                    c_sorted = sorted(ebc.items(), key=lambda x: x[1], reverse=True)
                    if len(graph.g) > DC_num:
                        can_trigger_nodes = [c_sorted[idx][0] for idx in range(DC_num)]
                    else:
                        can_trigger_nodes = [c_sorted[idx][0] for idx in range(len(graph.g))]

                    neg_node_score_list = {}

                    for node_i in range(len(can_trigger_nodes)):
                        node_i = can_trigger_nodes[node_i]
                        re_graph = copy.deepcopy(graph)
                        ori_list = list(re_graph.g.edges)
                        # 删除子图
                        for (i, j) in copy.deepcopy(ori_list):
                            if i == node_i or j == node_i:
                                ori_list.remove((i, j))
                                re_graph.g.remove_edge(i, j)

                        ori_list.extend([[i, j] for j, i in ori_list])
                        if len(ori_list) == 0:
                            neg_node_score_list[node_i] = 0
                            continue
                        re_graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)

                        # 测试删减不同子图网络的预测结果
                        re_output, _ = model([re_graph])
                        loss_fn = nn.MSELoss()
                        node_score = loss_fn(output, re_output).item()
                        neg_node_score_list[node_i] = node_score

                    trigger_nodes = []
                    imp_score_idx = sorted(neg_node_score_list.items(), key=lambda x: x[1], reverse=True)

                    for u, v in imp_score_idx:
                        trigger_nodes.append(u)
                        if len(trigger_nodes) == trigger_node_sum:
                            break


                elif position_type=='MIA' or position_type=='LIA' or position_type == 'Motif-B':
                    neg_node_score_list = {}

                    for node_i in range(len(graph.g)):
                        re_graph = copy.deepcopy(graph)
                        ori_list = list(re_graph.g.edges)
                        # 删除子图
                        for (i, j) in copy.deepcopy(ori_list):
                            if i == node_i or j == node_i:
                                ori_list.remove((i, j))
                                re_graph.g.remove_edge(i, j)

                        ori_list.extend([[i, j] for j, i in ori_list])
                        if len(ori_list) == 0:
                            neg_node_score_list[node_i] = 0
                            continue
                        re_graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)

                        # 测试删减不同子图网络的预测结果
                        re_output, _ = model([re_graph])
                        loss_fn = nn.MSELoss()
                        node_score = loss_fn(output, re_output).item()
                        neg_node_score_list[node_i] = node_score

                    trigger_nodes = []
                    if position_type == 'LIA':
                        imp_score_idx = sorted(neg_node_score_list.items(), key=lambda x:x[1], reverse=False)
                    elif position_type == 'MIA' or position_type == 'Motif-B':
                        imp_score_idx = sorted(neg_node_score_list.items(), key=lambda x: x[1], reverse=True)
                    else:
                        print('attack method error!')
                        exit()

                    for u,v in imp_score_idx:
                        trigger_nodes.append(u)
                        if len(trigger_nodes) == trigger_node_sum:
                            break

                else:
                    print('Position Error!')
                    exit()

                assert len(list(set(trigger_nodes))) == trigger_node_sum

                ori_list = list(graph.g.edges)
                # 清除原有的链路关系
                for i in trigger_nodes:
                    for j in trigger_nodes:
                        if (i, j) in ori_list:
                            ori_list.remove((i, j))
                            train_graphs[idx].g.remove_edge(i, j)

                # 加上触发器
                if  ER_sub != None:
                    motif_adj =  list(ER_sub.edges)
                else:
                    motif_adj = motif_trans(motif_idx)

                if best_sub != None and position_type=='motif_subgraph':
                    motif_adj = best_sub
                for node_i, node_j in motif_adj:
                    ori_list.append((trigger_nodes[node_i], trigger_nodes[node_j]))
                    train_graphs[idx].g.add_edge(trigger_nodes[node_i], trigger_nodes[node_j])

                # train_graphs[idx].g.edges = ori_list
                ori_list.extend([[i, j] for j, i in ori_list])
                train_graphs[idx].edge_mat = torch.LongTensor(ori_list).transpose(0, 1)
        poison_idx = None

    return train_graphs, poison_idx



def GTA_B(args, ori_train_graphs, ori_test_graphs, poison_rate, target_label, trigger_node_sum, model, device, max_node_num, train=True):
    #训练阶段
    train_graphs = copy.deepcopy(ori_train_graphs)
    test_graphs = copy.deepcopy(ori_test_graphs)

    from Generate_Net import Gen_Net
    toponet = Gen_Net(max_node_num, args.gtn_layernum).to(device)
    optimizer_topo = optim.Adam(toponet.parameters(),
                                lr=args.gtn_lr,
                                weight_decay=5e-4)

    # get a graph in train_graphs
    poison_idx = [random.randint(0, (len(train_graphs)-1)) for _ in range(int(len(train_graphs) * poison_rate))]

    # 生成初始触发器
    init_sub = np.ones((trigger_node_sum, trigger_node_sum))
    row, col = np.diag_indices_from(init_sub)
    init_sub[row, col] = 0
    init_sub_trigger = np.zeros((max_node_num, max_node_num))
    init_sub_trigger[:trigger_node_sum, :trigger_node_sum] = init_sub
    init_sub_trigger = torch.tensor(init_sub_trigger).to(device)

    for idx in poison_idx:
        graph = train_graphs[idx]
        if len(graph.g) < trigger_node_sum:
            continue

        #选定注入节点
        while (1):
            trigger_nodes = [random.randint(0, (len(graph.g) - 1)) for _ in range(trigger_node_sum)]
            if len(list(set(trigger_nodes))) == trigger_node_sum:
                trigger_nodes.sort()
                break

        # 注入全连接的触发器
        ori_list = list(graph.g.edges)
        # 清除原有的链路关系
        for i in trigger_nodes:
            for j in trigger_nodes:
                if (i, j) in ori_list:
                    ori_list.remove((i, j))
                    graph.g.remove_edge(i, j)

        # 格式化为稠密邻接矩阵
        graph_adj = np.zeros((max_node_num, max_node_num))
        for x, y in graph.g.edges:
            graph_adj[x, y] = 1
            graph_adj[y, x] = 1

        # 邻接矩阵中--触发器节点和节点前几位进行互换
        for i, j in enumerate(trigger_nodes):
            # 行交换 第i行和第j行进行交换
            graph_adj[[i, j], :] = graph_adj[[j, i], :]
            # 列交换 第i列和第j列进行交换
            graph_adj[:, [i, j]] = graph_adj[:, [j, i]]

            # node features 交换
            graph.node_features[[i, j], :] = graph.node_features[[j, i], :]

        graph.adj_dense_add = torch.FloatTensor(graph_adj)
        graph.adj_dense_init_len = torch.FloatTensor(graph_adj[:len(graph.g),:len(graph.g)])

    toponet.train()
    for _ in tqdm(range(args.gtn_epochs)):
        poi_graphs = []
        optimizer_topo.zero_grad()
        for idx in poison_idx:
            graph = copy.deepcopy(train_graphs[idx])
            if len(graph.g) < trigger_node_sum:
                continue
            rst_bkdA = toponet(graph.adj_dense_add.to(device), init_sub_trigger, args.topo_thrd, device, args.topo_activation, 'topo')
            graph.adj_dense = torch.add(
                rst_bkdA[:len(graph.g), :len(graph.g)].detach().cpu(),
                graph.adj_dense_init_len)
            graph.label = target_label
            poi_graphs.append(graph)


        sub_output, _ = model(poi_graphs, Is_adj_dense = True)
        labels = torch.LongTensor([target_label for _ in range(len(poi_graphs))]).to(device)

        # 将真实类标转换成0,1的形式
        labels_cl = torch.zeros_like(sub_output)
        for i in range(len(labels)):
            labels_cl[i][labels[i]] = torch.cuda.FloatTensor([1])

        # compute loss
        #     loss = criterion(output, labels)
        #     loss = criterion_cl_def(F.softmax(output), labels_cl.detach())
        sub_loss = -torch.sum(F.log_softmax(sub_output, dim=-1) * labels_cl.detach()) / len(labels_cl)

        print('loss', sub_loss)
        sub_loss.backward()
        optimizer_topo.step()
        torch.cuda.empty_cache()

    #中毒训练和测试数据集
    for idx in poison_idx:
        graph = train_graphs[idx]
        if len(graph.g) < trigger_node_sum:
            continue

        rst_bkdA = toponet(graph.adj_dense_add.to(device), init_sub_trigger, args.topo_thrd, device, args.topo_activation,
                       'topo', Is_inject = True)
        #注入触发器
        graph.adj_dense = torch.add(
            rst_bkdA[:len(graph.g), :len(graph.g)].detach().cpu(), graph.adj_dense_init_len)
        a, b = torch.where(graph.adj_dense!=0)

        # 加上触发器
        ori_list = [(a[i].item(),b[i].item()) for i in range(len(a))]
        graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)

        #修改label
        graph.label = target_label

    #测试数据集
    for idx in range(len(test_graphs)):
        graph = test_graphs[idx]
        if len(graph.g) < trigger_node_sum:
            continue
        #选定注入节点
        while (1):
            trigger_nodes = [random.randint(0, (len(graph.g) - 1)) for _ in range(trigger_node_sum)]
            if len(list(set(trigger_nodes))) == trigger_node_sum:
                trigger_nodes.sort()
                break

        # 注入全连接的触发器
        ori_list = list(graph.g.edges)
        # 清除原有的链路关系
        for i in trigger_nodes:
            for j in trigger_nodes:
                if (i, j) in ori_list:
                    ori_list.remove((i, j))
                    graph.g.remove_edge(i, j)

        # 格式化为稠密邻接矩阵
        graph_adj = np.zeros((max_node_num, max_node_num))
        for x, y in graph.g.edges:
            graph_adj[x, y] = 1
            graph_adj[y, x] = 1

        # 邻接矩阵中--触发器节点和节点前几位进行互换
        for i, j in enumerate(trigger_nodes):
            # 行交换 第i行和第j行进行交换
            graph_adj[[i, j], :] = graph_adj[[j, i], :]
            # 列交换 第i列和第j列进行交换
            graph_adj[:, [i, j]] = graph_adj[:, [j, i]]

            # node features 交换
            graph.node_features[[i, j], :] = graph.node_features[[j, i], :]

        graph.adj_dense_add = torch.FloatTensor(graph_adj)
        graph.adj_dense_init_len = torch.FloatTensor(graph_adj[:len(graph.g),:len(graph.g)])

        rst_bkdA = toponet(graph.adj_dense_add.to(device), init_sub_trigger, args.topo_thrd, device, args.topo_activation,
                       'topo', Is_inject = True)
        #注入触发器
        graph.adj_dense = torch.add(
            rst_bkdA[:len(graph.g), :len(graph.g)].detach().cpu(), graph.adj_dense_init_len)
        a, b = torch.where(graph.adj_dense!=0)

        # 加上触发器
        ori_list = [(a[i].item(),b[i].item()) for i in range(len(a))]
        graph.edge_mat = torch.LongTensor(ori_list).transpose(0, 1)

    return train_graphs, test_graphs, poison_idx


def get_graph_emb(graph):
    edges = graph.edge_mat.transpose(1, 0).numpy().tolist()
    adj = np.zeros((len(graph.g), len(graph.g)))
    for i, j in edges:
        adj[i, j] = 1
    node_fea = np.array(graph.node_features)
    graph_emb = np.array([np.sum(np.dot(adj, np.dot(adj, node_fea)), axis=0)])
    return graph_emb


def motif_feature(ori_train_graphs, poison_rate, target_label, motif_idx, trigger_node_sum, M_list, train=True, backdoor_type = 'motif_fea'):
    train_graphs = copy.deepcopy(ori_train_graphs)
    assert motif_idx[1] == str(trigger_node_sum)


    if train == True:
        # get a graph in train_graphs
        poison_idx = [random.randint(0, (len(train_graphs)-1)) for _ in range(int(len(train_graphs) * poison_rate))]

        for idx in poison_idx:
            graph = train_graphs[idx]

            # construct the adjacency matrix of the graph
            # 注入节点数为3的触发器
            if len(graph.g) >= trigger_node_sum:
                #motif_fea
                if backdoor_type == 'motif_fea':
                    for M in M_list:
                        M_target, M_other, emb_list = motif_deal([graph], target_label, M)

                    if len(emb_list[0])!=0:
                        emb_modify_nodes =  emb_list[0][random.randint(0,len(emb_list[0])-1)]
                        # fea_dim = graph.node_features.shape[1]
                        m = 0
                        for i in emb_modify_nodes:
                            m += 1
                            if m > 2:
                                m = 0
                            train_graphs[idx].node_features[i] = torch.cuda.FloatTensor([0])
                            train_graphs[idx].node_features[i][m] = torch.cuda.FloatTensor([1])

                elif backdoor_type == 'random_fea':
                    while (1):
                        trigger_nodes = [random.randint(0, (len(graph.g) - 1)) for _ in range(trigger_node_sum)]
                        if len(list(set(trigger_nodes))) == trigger_node_sum:
                            break
                    assert len(list(set(trigger_nodes))) == trigger_node_sum

                    # fea_dim = graph.node_features.shape[1]
                    m = 0
                    for i in trigger_nodes:
                        m += 1
                        if m > 2:
                            m = 0
                        train_graphs[idx].node_features[i] = torch.cuda.FloatTensor([0])
                        train_graphs[idx].node_features[i][m] = torch.cuda.FloatTensor([1])

                        # j = random.randint(0, (len(graph.g) - 1))
                        # train_graphs[idx].node_features[i] = train_graphs[idx].node_features[j]
                else:
                    print('Backdoor type error!')
                    exit()

                # target-label
                train_graphs[idx].label = target_label
    else:
        for idx in range(len(train_graphs)):
            graph = train_graphs[idx]

            # construct the adjacency matrix of the graph
            # 注入节点数为3的触发器
            if len(graph.g) >= trigger_node_sum:
                # motif_fea
                if backdoor_type == 'motif_fea':
                    for M in M_list:
                        M_target, M_other, emb_list = motif_deal([graph], target_label, M)

                    if len(emb_list[0]) != 0:
                        emb_modify_nodes = emb_list[0][random.randint(0, len(emb_list[0]) - 1)]
                        # fea_dim = graph.node_features.shape[1]
                        m = 0
                        for i in emb_modify_nodes:
                            m += 1
                            if m > 2:
                                m = 0
                            train_graphs[idx].node_features[i] = torch.cuda.FloatTensor([0])
                            train_graphs[idx].node_features[i][m] = torch.cuda.FloatTensor([1])

                elif backdoor_type == 'random_fea':
                    while (1):
                        trigger_nodes = [random.randint(0, (len(graph.g) - 1)) for _ in range(trigger_node_sum)]
                        if len(list(set(trigger_nodes))) == trigger_node_sum:
                            break
                    assert len(list(set(trigger_nodes))) == trigger_node_sum

                    # fea_dim = graph.node_features.shape[1]
                    m = 0
                    for i in trigger_nodes:
                        m += 1
                        if m > 2:
                            m = 0
                        train_graphs[idx].node_features[i] = torch.cuda.FloatTensor([0])
                        train_graphs[idx].node_features[i][m] = torch.cuda.FloatTensor([1])

                        # j = random.randint(0, (len(graph.g) - 1))
                        # train_graphs[idx].node_features[i] = train_graphs[idx].node_features[j]
                else:
                    print('Backdoor type error!')
                    exit()

    return train_graphs


def no_exist_feature_poison(ori_train_graphs, poison_rate, target_label, motif_idx, trigger_node_sum, train=True, position_type=False):
    train_graphs = copy.deepcopy(ori_train_graphs)

    if train == True:
        # get a graph in train_graphs
        poison_idx = [random.randint(0, (len(train_graphs)-1)) for _ in range(int(len(train_graphs) * poison_rate))]

        for idx in poison_idx:
            graph = train_graphs[idx]
            # construct the adjacency matrix of the graph
            if len(graph.g) >= trigger_node_sum:
                if position_type == 'DC':
                    ebc = nx.centrality.degree_centrality(graph.g)
                    c_sorted = sorted(ebc.items(), key=lambda x: x[1], reverse=True)
                    trigger_nodes = [c_sorted[idx][0] for idx in range(trigger_node_sum)]

                elif position_type==False:
                    while (1):
                        trigger_nodes = [random.randint(0, (len(graph.g) - 1)) for _ in range(trigger_node_sum)]
                        if len(list(set(trigger_nodes))) == trigger_node_sum:
                            break
                else:
                    print('Position Error!')
                    exit()

                assert len(list(set(trigger_nodes))) == trigger_node_sum

                for i in trigger_nodes:
                    train_graphs[idx].node_features[i][-1] = torch.cuda.FloatTensor([1])
                    train_graphs[idx].label = target_label

    else:
        for idx in range(len(train_graphs)):
            graph = train_graphs[idx]

            # construct the adjacency matrix of the graph
            # 注入节点数为3的触发器
            if len(graph.g) >= trigger_node_sum:
                if position_type == 'DC':
                    ebc = nx.centrality.degree_centrality(graph.g)
                    c_sorted = sorted(ebc.items(), key=lambda x: x[1], reverse=True)
                    trigger_nodes = [c_sorted[idx][0] for idx in range(trigger_node_sum)]

                elif position_type==False:
                    while (1):
                        trigger_nodes = [random.randint(0, (len(graph.g) - 1)) for _ in range(trigger_node_sum)]
                        if len(list(set(trigger_nodes))) == trigger_node_sum:
                            break
                else:
                    print('Position Error!')
                    exit()

                assert len(list(set(trigger_nodes))) == trigger_node_sum

                for i in trigger_nodes:
                    train_graphs[idx].node_features[i][-1] = torch.cuda.FloatTensor([1])

    return train_graphs



# 画图
def plot_graph(epoch_list, train_asr_list, test_asr_list,data_name):

    x = np.arange(len(epoch_list))
    # bar_width = 0.2  # 条形宽度attack_RA

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)

    ax1.plot(x, train_asr_list, marker='o', color='#f5bf03', label='Dyn-Backdoor')
    plt.xticks(x, epoch_list, fontproperties='Times New Roman', size=15)
    plt.ylabel('Train_ASR', fontproperties='Times New Roman', size=20)
    plt.xlabel('Epoch', fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=15)
    ax1.set_ylim([0, 1])
    # plt.legend()

    ax2 = fig.add_subplot(1, 2, 2)

    ax2.plot(x, test_asr_list, marker='o', color='#f5bf03')
    plt.xticks(x, epoch_list, fontproperties='Times New Roman', size=15)
    plt.ylabel('Test_ASR', fontproperties='Times New Roman', size=20)
    plt.xlabel('Epoch', fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=15)
    ax2.set_ylim([0, 1])
    # plt.legend()
    # fig.savefig('./results/{}/{}/{}_train_trigger_{}_test_trigger_{}.pdf'.format(attack_method, data_name, data_name, train_motif_name, test_motif_name),
    #             bbox_inches='tight')
    plt.show()

def com_graph_sim(test_graphs, poi_test_graphs):
    delta_list, simrank_list, degree_fea_list = [], [], []
    for i in range(len(test_graphs)):
        # compute the distance between two graphs using DeltaCon
        delta = delta_con(test_graphs[i], poi_test_graphs[i])
        delta_list.append(delta)

        # compute the distance between two graphs using SimRank
        simrank = sim_rank_distance(test_graphs[i], poi_test_graphs[i])
        simrank_list.append(simrank)

        # compute the distance between two graphs using In/Out Degree features
        degree_fea = degree_dist(test_graphs[i], poi_test_graphs[i])
        degree_fea_list.append(degree_fea)

    Del_D = np.mean(delta_list)
    Sim_D = np.mean(simrank_list)
    Deg_D = np.mean(degree_fea_list)

    return Del_D, Sim_D, Deg_D






import networkx as nx
import numpy as np
import random
import os
import torch
from sklearn.model_selection import StratifiedKFold
import dgl

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag = False):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        #readline读取一行 strip去除首尾空格
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags, node_features=torch.tensor(node_features)))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        # deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    # if node_features == []:
    if degree_as_tag or node_tags == [] or len(set(node_tags)) == 1:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())


    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}
    max_nodes = 0
    for g in g_list:
        if g.g.number_of_nodes() > max_nodes:
            max_nodes = g.g.number_of_nodes()
        if node_features == []:
            g.node_features = torch.zeros(len(g.node_tags), len(tagset))
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict), max_nodes

def read_graphfile(dataname, max_nodes=0, degree_as_tag = False):

    prefix = os.path.join('dataset', dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels.append(int(line))
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_nodes_feat = prefix + '_node_attributes.txt'
    node_attributes = []
    try:
        with open(filename_nodes_feat) as f:
            for line in f:
                line = line.strip("\n").split(",")
                node_attributes.append([float(node_fea) for node_fea in line])
                # attr0, attr1 = (float(line[0].strip(" ")), float(line[1].strip(" ")))
                # node_attributes.append([attr0, attr1])
    except IOError:
        print('No node attributes')

    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    # assume that all graph labels appear in the dataset
    # (set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            # if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    # graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    # if label_has_zero:
    #    graph_labels += 1

    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        # indexed from 1 here
        G = nx.from_edgelist(adj_list[i])
        # if G.number_of_nodes() > max_nodes:
        #     max_nodes = G.number_of_nodes()
        if G.number_of_nodes() > max_nodes:
            max_nodes = G.number_of_nodes()
        # add features and labels

        node_re_labels = []
        node_attrs = []
        for u in G.nodes:
            if len(node_labels)  == 0 and node_attributes != []:
                node_attrs.append(node_attributes[u - 1])
            elif len(node_labels) == 0 and node_attributes == []:
                break
            elif len(node_labels) != 0 and node_attributes == []:
                node_re_labels.append(node_labels[u - 1])
            else:
                node_re_labels.append(node_labels[u - 1])
                node_attrs.append(node_attributes[u - 1])

        # relabeling
        mapping = {}
        it = 0
        for n in G.nodes:
            mapping[n] = it
            it += 1

        # indexed from 0
        g = nx.relabel_nodes(G, mapping)
        graphs.append(S2VGraph(g, graph_labels[i - 1], node_re_labels, node_features=torch.tensor(node_attrs)))

    for g in graphs:
        g.neighbors = [[] for _ in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        edges = [list(pair) for pair in g.g.edges()]
        if len(edges) == 0 or len(g.g) == 0:
            graphs.remove(g)
            continue
        edges.extend([[i, j] for j, i in edges])

        g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

    # Extracting unique tag labels
    if degree_as_tag or node_attributes == [] or len(set(node_re_labels)) == 1:
        for g in graphs:
            g.node_tags = list(dict(g.g.degree).values())
        tagset = set([])
        for g in graphs:
            tagset = tagset.union(set(g.node_tags))

        tagset = list(tagset)
        tag2index = {tagset[i]: i for i in range(len(tagset))}
        max_nodes = 0
        for g in graphs:
            if g.g.number_of_nodes() > max_nodes:
                max_nodes = g.g.number_of_nodes()
            g.node_features = torch.zeros(len(g.node_tags), len(tagset))
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    return graphs, len(label_map_to_int), max_nodes

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = seed) #n_splits=5 对应的是train:test = 8:2

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


def numpy_to_graph(A, type_graph='dgl', node_features=None, to_cuda=True):
    '''Convert numpy arrays to graph

    Parameters
    ----------
    A : mxm array
        Adjacency matrix
    type_graph : str
        'dgl' or 'nx'
    node_features : dict
        Optional, dictionary with key=feature name, value=list of size m
        Allows user to specify node features

    Returns

    -------
    Graph of 'type_graph' specification
    '''

    G = nx.from_numpy_array(A)

    if node_features != None:
        for n in G.nodes():
            for k, v in node_features.items():
                G.nodes[n][k] = v[n]

    if type_graph == 'nx':
        return G

    G = G.to_directed()

    if node_features != None:
        node_attrs = list(node_features.keys())
    else:
        node_attrs = []

    g = dgl.from_networkx(G, node_attrs=node_attrs, edge_attrs=['weight'])
    if to_cuda:
        g = g.to(torch.device('cuda'))
    return g





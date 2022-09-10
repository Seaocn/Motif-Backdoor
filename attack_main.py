import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from attack_utils import *
import networkx as nx
import json


from sklearn.metrics.pairwise import cosine_similarity
from util import read_graphfile
from models.GcnEncoderGraph import GcnEncoderGraph
from models.graphcnn_neuron import GraphCNN_Neuron


criterion = nn.CrossEntropyLoss()


num_threads = 1
torch.set_num_threads(num_threads)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)



def train(args, model, device, train_graphs, optimizer, epoch, poi_list):
    model.train()

    # total_iters =  len(train_graphs)//args.batch_size
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')
    # train_idx_list = list(range(len(train_graphs)))
    # random.shuffle(train_idx_list)
    #
    # train_idx = [train_idx_list[i*args.batch_size:(i+1)*args.batch_size] for i in range(total_iters)]

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size-5]

        #epoch has poi
        random.shuffle(poi_list)
        selected_idx = selected_idx.tolist()
        for j, m in enumerate(poi_list):
            selected_idx.append(m)
            if j == 4:
                break
        random.shuffle(selected_idx)


        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output, _ = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)


        # compute loss
        loss = criterion(output, labels)
        # grad = torch.autograd.grad(loss, batch_graph, retain_graph=True)[0].data

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
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

def test_attack(clean_model, poi_model, device, train_graphs, test_graphs, poi_train_graphs, poi_test_graphs,target_label):
    # clean_model.eval()
    # poi_model.eval()

    # output = pass_data_iteratively(clean_model, train_graphs)
    # pred = output.max(1, keepdim=True)[1]
    # labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    #
    # poi_output = pass_data_iteratively(poi_model, poi_train_graphs)
    # poi_pred = poi_output.max(1, keepdim=True)[1]
    #
    # train_tatal, train_suc = 0, 0
    # for i in range(len(train_graphs)):
    #     if (labels[i] == pred[i]) and (labels[i] != target_label) :
    #         train_tatal += 1
    #         if poi_pred[i] == target_label:
    #             train_suc += 1

    # train_asr = train_suc/train_tatal


    output = pass_data_iteratively(clean_model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    clean_acc_test = correct / float(len(test_graphs))


    poi_output = pass_data_iteratively(poi_model, poi_test_graphs)
    poi_pred = poi_output.max(1, keepdim=True)[1]
    poi_output = F.softmax(poi_output, dim=1)  #进行概率化


    poi_AMC = []
    test_tatal, test_suc = 0, 0
    for i in range(len(test_graphs)):
        if (labels[i] == pred[i]) and (labels[i] != target_label) :
            test_tatal += 1
            if poi_pred[i] == target_label:
                test_suc += 1
                poi_AMC.append(float(poi_output[i][poi_pred[i]].cpu())) #中毒样本的置信分数

    test_asr = test_suc/test_tatal

    poi_AMC = np.mean(poi_AMC)

    # print('train_tatal:%f  test_tatal:%f'%(train_tatal,test_tatal))
    print("test_tatal:%f test_asr: %f" % (test_tatal, test_asr))
    print("poi_AMC: %f" % (poi_AMC))

    return test_asr, test_asr, poi_AMC, clean_acc_test




def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="NCI1",   #MUTAG 1, IMDBBINARY 0, PROTEINS 1, Fingerprint 2, AIDS 0, NCI1 0, twitch_egos
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=6,
                        help='which gpu to use if any (default: 0)')
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


    parser.add_argument('--adv_rate', type=float, default=0.1,
                        help='probability for edge creation/rewiring each edge')
    parser.add_argument('--prob', type=float, default=0.7,
                        help='probability for edge creation/rewiring each edge')
    parser.add_argument('--num_backdoor_nodes', type=int, default=4,
                        help='Each node is connected to k nearest neighbors in ring topology')
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--K', type=int, default=3,
                        help='Each node is connected to k nearest neighbors in ring topology')
    parser.add_argument('--frac', type=float, default=0.1, help='poisoning ratio')
    parser.add_argument('--modeltype', type=str, default='gcn',
                        choices=['gin', 'Sagepool', 'HGP-SL', 'gcn', 'gat', 'diffpool', 'sage'])
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

    #GTA
    parser.add_argument('--gtn_layernum', type=int, default=3, help="layer number of GraphTrojanNet")
    parser.add_argument('--topo_thrd', type=float, default=0.1, help="threshold for topology generator")
    parser.add_argument('--gtn_lr', type=float, default=0.01)
    parser.add_argument('--gtn_epochs', type=int, default=10, help="# attack epochs")
    parser.add_argument('--topo_activation', type=str, default='sigmoid',
                       help="activation function for topology generator")
    args = parser.parse_args()

    if args.dataset == 'MUTAG' or args.dataset == 'PROTEINS' or args.dataset == 'IMDBBINARY':
        args.target = 1
    elif   args.dataset == 'AIDS' \
            or args.dataset == 'NCI1' or args.dataset == 'twitch_egos'\
            or args.dataset == 'DBLP_v1' or args.dataset == 'MCF-7':
        args.target = 0
    elif args.dataset == 'Fingerprint':
        args.target = 2
    else:
        print('dataset error!')
        exit()

    # set up seeds and gpu device
    torch.manual_seed(0)
    # np.random.seed(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.dataset == 'twitch_egos' or args.dataset == 'AIDS' \
            or args.dataset == 'DBLP_v1' or args.dataset == 'MCF-7':
        graphs, num_classes, max_nodes = read_graphfile(args.dataset)
    else:
        graphs, num_classes, max_nodes = load_data(args.dataset)



    print('Max Nodes', max_nodes)
    if args.dataset == 'Fingerprint':
        graphs_re_idx = []
        re_num_idx = 0
        label_idx = [[] for _ in range(num_classes)]
        for graph_idx in range(len(graphs)):
            label_idx[graphs[graph_idx].label].append(graph_idx)

        for m in range(len(label_idx)):
            if len(label_idx[m]) > 150:
                graphs_re_idx = graphs_re_idx + label_idx[m]
                for idx in label_idx[m]:
                    graphs[idx].label = re_num_idx
                re_num_idx += 1
        graphs_re = [graphs[idx] for idx in graphs_re_idx]
        error_idx = []
        for g_idx in range(len(graphs_re)):
            if isinstance(graphs_re[g_idx].edge_mat, int):
                error_idx.append(g_idx)
        all_list = list(set([i for i in range(len(graphs_re))]) - set(error_idx))
        graphs = [graphs_re[idx] for idx in all_list]
        num_classes = re_num_idx



    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)
    clean_train_graphs = copy.deepcopy(train_graphs)



    #加载干净模型
    if args.modeltype == 'gin':
        clean_model = GraphCNN_Neuron(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1],
                                args.hidden_dim,
                                num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                                args.neighbor_pooling_type, device).to(device)
    elif args.modeltype == 'gcn':
        clean_model = GcnEncoderGraph(max_nodes,
                    train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.num_layers,
                    dropout=args.dropout, device=device).to(device)
    elif args.modeltype == 'gat':
        clean_model = GAT(train_graphs[0].node_features.shape[1], num_classes,max_nodes, hidden_dim=[64, 32],
                    dropout=args.dropout, num_head=3, device=device).to(device)
    elif args.modeltype == 'diffpool':
        clean_model = Diffpool(max_nodes,
                    train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.num_layers,
                    assign_hidden_dim=args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool, gcn_concat=True,
                    pool_concat=True, dropout=args.dropout, args=args, device=device).to(device)
    elif args.modeltype == 'HGP-SL':
        clean_model = HGPSLPool(args, train_graphs[0].node_features.shape[1], num_classes, 64, args.pooling_ratio,
                          args.dropout, device).to(device)
    else:
        print('load model error!')
        exit()
    clean_model.load_state_dict(torch.load('./model_params/{}_model_{}_params.pkl'.format(args.dataset, args.modeltype), map_location='cuda:0'))
    print('target model', args.modeltype)

    output = pass_data_iteratively(clean_model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    clean_acc_test1 = correct / float(len(test_graphs))
    print('clean_acc_test1', clean_acc_test1)

    train_motif_name = test_motif_name = 'M44'
    M_list = [train_motif_name]

    train_trigger_num, test_trigger_num = int(train_motif_name[1]), int(test_motif_name[1])
    train_poi_rate = args.frac
    position_type_value = 'Motif-DC-B' 
                                
    if position_type_value == 'MIA' or position_type_value == 'DC-B':
        train_motif_name = test_motif_name = 'M44'
        train_trigger_num, test_trigger_num = int(train_motif_name[1]), int(test_motif_name[1])
    elif position_type_value == 'Random':
        pass
    else:
        train_trigger_num, test_trigger_num = 4, 4


    if position_type_value == 'ER' or position_type_value == 'MIA':
        G_gen = nx.erdos_renyi_graph(train_trigger_num, args.prob)  #'ER'
        train_trigger_num = test_trigger_num
    elif position_type_value == 'SW':
        G_gen = nx.watts_strogatz_graph(train_trigger_num, args.K, args.prob, seed=None) #'SW'
        train_trigger_num = test_trigger_num
    elif position_type_value == 'PA':
        G_gen = nx.barabasi_albert_graph(train_trigger_num, args.K, seed=None)   #'PA'
        train_trigger_num = test_trigger_num
    else:
        G_gen = None


    target_neuron_path = Find_neuron_path(graphs, clean_model, target_label, train_trigger_num) if position_type_value == 'Neuron_Path' else 0

    graph_emb_target, graph_emb_other = None, None




    #训练数据加上触发器
    if train_motif_name != 'non_trigger':
        if position_type_value == 'motif_subgraph':
            best_sub, poison_idx = motif_subgraph(args, clean_train_graphs, train_poi_rate, target_label,
                                          train_trigger_num, clean_model, device,max_nodes, train=True)
            train_graphs, _ = motif_poison(clean_train_graphs, train_poi_rate, target_label, train_motif_name, train_trigger_num, clean_model,target_neuron_path,
                                        graph_emb_target, graph_emb_other, train=True,position_type= position_type_value, ER_sub=G_gen, best_sub=best_sub, poison_idx=poison_idx)

        elif position_type_value == 'GTA':
            # 替代模型
            su_model = GraphCNN_Neuron(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1],
                                       args.hidden_dim,
                                       num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                                       args.neighbor_pooling_type, device).to(device)
            su_model.load_state_dict(
                torch.load('./model_params/{}_model_gin_params.pkl'.format(args.dataset), map_location='cuda:0'))

            train_graphs, attack_test_graphs, poi_list = GTA_B(args, clean_train_graphs, test_graphs, train_poi_rate, target_label,
                                          train_trigger_num, su_model, device,max_nodes, train=True)

        else:
            train_graphs, poi_list = motif_poison(clean_train_graphs, train_poi_rate, target_label, train_motif_name,
                                        train_trigger_num, clean_model, target_neuron_path,
                                        graph_emb_target, graph_emb_other, train=True,
                                        position_type=position_type_value, ER_sub=G_gen)


    if test_motif_name == 'non_trigger':
        # attack_train_graphs = clean_train_graphs
        attack_test_graphs = test_graphs
    else:
        if position_type_value == 'motif_subgraph':
            # 查看训练过程中的攻击中毒效果
            # attack_train_graphs, _ = motif_poison(clean_train_graphs, 0.05, target_label, test_motif_name,
            #                                    test_trigger_num, clean_model, target_neuron_path,
            #                                    graph_emb_target, graph_emb_other, train=False,
            #                                    position_type=position_type_value, ER_sub=G_gen, best_sub=best_sub)
            attack_test_graphs, _ = motif_poison(test_graphs, 0.05, target_label, test_motif_name, test_trigger_num,
                                              clean_model, target_neuron_path,
                                              graph_emb_target, graph_emb_other, train=False,
                                              position_type=position_type_value, ER_sub=G_gen, best_sub=best_sub)
        elif position_type_value == 'GTA':
            pass
        else:
            #查看训练过程中的攻击中毒效果
            # attack_train_graphs, _ = motif_poison(clean_train_graphs, 0.05, target_label,  test_motif_name, test_trigger_num,clean_model,target_neuron_path,
            #                                    graph_emb_target, graph_emb_other, train=False,position_type=position_type_value, ER_sub=G_gen)
            attack_test_graphs, _ = motif_poison(test_graphs, 0.05, target_label, test_motif_name, test_trigger_num,clean_model,target_neuron_path,
                                              graph_emb_target, graph_emb_other, train=False, position_type=position_type_value, ER_sub=G_gen)




    # 模型训练
    if args.modeltype == 'gin':
        model = GraphCNN_Neuron(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1],
                                args.hidden_dim,
                                num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                                args.neighbor_pooling_type, device).to(device)
    elif args.modeltype == 'gcn':
        model = GcnEncoderGraph(max_nodes,
                    train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.num_layers,
                    dropout=args.dropout, device=device).to(device)
    elif args.modeltype == 'gat':
        model = GAT(train_graphs[0].node_features.shape[1], num_classes,max_nodes, hidden_dim=[64, 32],
                    dropout=args.dropout, num_head=3, device=device).to(device)
    elif args.modeltype == 'diffpool':
        model = Diffpool(max_nodes,
                    train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.num_layers,
                    assign_hidden_dim=args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool, gcn_concat=True,
                    pool_concat=True, dropout=args.dropout, args=args, device=device).to(device)
    elif args.modeltype == 'HGP-SL':
        model = HGPSLPool(args, train_graphs[0].node_features.shape[1], num_classes, 64, args.pooling_ratio,
                          args.dropout, device).to(device)
    else:
        print('load model error!')
        exit()

    # model.load_state_dict(
    #     torch.load('./model_params/{}_model_{}_params.pkl'.format(args.dataset, args.modeltype), map_location='cuda:0'))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    # compute the distance between two graphs
    # Del_D, Sim_D, Deg_D = com_graph_sim(test_graphs, attack_test_graphs)
    test_bad_list, test_asr_list, test_amc_list = [], [], []



    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch, poi_list)
        acc_train, acc_test = test(args, model, device, clean_train_graphs, test_graphs, epoch)

        # test attack
        train_asr, test_ars, poison_AMC, clean_acc_test = test_attack(clean_model, model, device,clean_train_graphs,test_graphs, clean_train_graphs, attack_test_graphs,target_label)


        BAD = clean_acc_test - acc_test
        print('clean acc {} poi acc {}'.format(clean_acc_test, acc_test))
        print('benign accuracy drop:', BAD)

        test_bad_list.append(BAD)
        test_asr_list.append(test_ars)
        test_amc_list.append(poison_AMC)

        print('dataset',args.dataset, 'target_label', target_label, 'position_type', position_type_value)
        print(train_motif_name, test_motif_name, M_list)


if __name__ == '__main__':
    main()

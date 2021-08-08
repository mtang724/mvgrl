''' Code adapted from https://github.com/kavehhassani/mvgrl '''
import numpy as np
import torch as th
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import networkx as nx

from sklearn.preprocessing import MinMaxScaler

from dgl.nn import APPNPConv
import torch

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1

def read_real_datasets(datasets):
    edge_path = "../new_data/{}/out1_graph_edges.txt".format(datasets)
    node_feature_path = "../new_data/{}/out1_node_feature_label.txt".format(datasets)
    with open(edge_path) as edge_file:
        edge_file_lines = edge_file.readlines()
        G = nx.parse_edgelist(edge_file_lines[1:], nodetype=int)
        # g = dgl.from_networkx(G)
        A = nx.adjacency_matrix(G)
    with open(node_feature_path) as node_feature_file:
        node_lines = node_feature_file.readlines()[1:]
        feature_list = []
        labels = []
        max_len = 0
        for node_line in node_lines:
            node_id, feature, label = node_line.split("\t")
            labels.append(int(label))
            features = feature.split(",")
            max_len = max(len(features), max_len)
            feature_list.append([float(feature) for feature in features])
        feature_pad_list = []
        for features in feature_list:
            features += [0] * (max_len - len(features))
            feature_pad_list.append(features)
        feature_array = np.array(feature_pad_list)
        features = torch.from_numpy(feature_array).float()
        # features = sp.csr_matrix(features)
        # g.ndata['attr'] = features.float()
        labels = np.array(labels)
        labels = torch.FloatTensor(labels).long()
    return dgl.from_networkx(G), features, labels

def process_dataset(name, epsilon):
    if name == 'cora':
        dataset = CoraGraphDataset()
        graph = dataset[0]
        feat = graph.ndata.pop('feat')
        label = graph.ndata.pop('label')
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
        graph = dataset[0]
        feat = graph.ndata.pop('feat')
        label = graph.ndata.pop('label')
    else:
        graph, feat, label = read_real_datasets(name)

    # train_mask = graph.ndata.pop('train_mask')
    # val_mask = graph.ndata.pop('val_mask')
    # test_mask = graph.ndata.pop('test_mask')
    #
    # train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    # val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    # test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    nx_g = dgl.to_networkx(graph)

    print('computing ppr')
    diff_adj = compute_ppr(nx_g, 0.2)
    print('computing end')

    if name == 'citeseer':
        print('additional processing')
        feat = th.tensor(preprocess_features(feat.numpy())).float()
        diff_adj[diff_adj < epsilon] = 0
        scaler = MinMaxScaler()
        scaler.fit(diff_adj)
        diff_adj = scaler.transform(diff_adj)

    diff_edges = np.nonzero(diff_adj)
    diff_weight = diff_adj[diff_edges]
    diff_graph = dgl.graph(diff_edges)

    graph = graph.add_self_loop()

    return graph, diff_graph, feat, label, None, None, None, diff_weight

def process_dataset_appnp(epsilon):
    k = 20
    alpha = 0.2
    dataset = PubmedGraphDataset()
    graph = dataset[0]
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    appnp = APPNPConv(k, alpha)
    id = th.eye(graph.number_of_nodes()).float()
    diff_adj = appnp(graph.add_self_loop(), id).numpy()

    diff_adj[diff_adj < epsilon] = 0
    scaler = MinMaxScaler()
    scaler.fit(diff_adj)
    diff_adj = scaler.transform(diff_adj)
    diff_edges = np.nonzero(diff_adj)
    diff_weight = diff_adj[diff_edges]
    diff_graph = dgl.graph(diff_edges)

    return graph, diff_graph, feat, label, train_idx, val_idx, test_idx, diff_weight
import argparse
import numpy as np
import torch as th
import torch.nn as nn

import warnings

warnings.filterwarnings('ignore')

from dataset import process_dataset
from model import MVGRL, LogReg
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import logging
from functools import lru_cache
import torch.nn.functional as F
import statistics

parser = argparse.ArgumentParser(description='mvgrl')

parser.add_argument('--dataname', type=str, default='chameleon', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index. Default: -1, using cpu.')
parser.add_argument('--epochs', type=int, default=500, help='Training epochs.')
parser.add_argument('--patience', type=int, default=20, help='Patient epochs to wait before early stopping.')
parser.add_argument('--lr1', type=float, default=0.001, help='Learning rate of mvgrl.')
parser.add_argument('--lr2', type=float, default=0.01, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=0., help='Weight decay of mvgrl.')
parser.add_argument('--wd2', type=float, default=0., help='Weight decay of linear evaluator.')
parser.add_argument('--epsilon', type=float, default=0.01, help='Edge mask threshold of diffusion graph.')
parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')

args = parser.parse_args()

class NodeClassificationDataset(Dataset):
    def __init__(self, node_embeddings, labels):
        self.len = node_embeddings.shape[0]
        self.x_data = node_embeddings
        self.y_data = labels

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

class DataSplit:

    def __init__(self, dataset, test_train_split=0.8, val_train_split=0.2, shuffle=True):
        self.dataset = dataset

        dataset_size = len(dataset)
        self.indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_split * dataset_size))

        if shuffle:
            np.random.shuffle(self.indices)

        train_indices, self.test_indices = self.indices[:test_split], self.indices[test_split:]
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))

        self.train_indices, self.val_indices = train_indices[ : validation_split], train_indices[validation_split:]

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader

def train_real_datasets(emb, node_labels):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    class_number = int(max(node_labels)) + 1
    input_dims = emb.shape
    FNN = MLP(num_layers=4, input_dim=input_dims[1], hidden_dim=input_dims[1] // 2, output_dim=class_number).to(
        device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(FNN.parameters())
    dataset = NodeClassificationDataset(emb, node_labels)
    split = DataSplit(dataset, shuffle=True)
    train_loader, val_loader, test_loader = split.get_split(batch_size=64, num_workers=0)
    # train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    best = float('inf')
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            print("here")
            # data = data.to(device)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            y_pred = FNN(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                correct = 0
                total = 0
                for data in val_loader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = FNN(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    total += labels.size(0)
                    correct += torch.sum(predicted == labels)
            if loss < best:
                best = loss
                torch.save(FNN.state_dict(), 'best_mlp.pkl')

    with torch.no_grad():
        FNN.load_state_dict(torch.load('best_mlp.pkl'))
        correct = 0
        total = 0
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = FNN(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)
    return (correct / total).item()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':
    print(args)

    # Step 1: Prepare data =================================================================== #
    graph, diff_graph, feat, label, train_idx, val_idx, test_idx, edge_weight = process_dataset(args.dataname, args.epsilon)
    n_feat = feat.shape[1]
    n_classes = np.unique(label).shape[0]

    graph = graph.to(args.device)
    diff_graph = diff_graph.to(args.device)
    feat = feat.to(args.device)
    edge_weight = th.tensor(edge_weight).float().to(args.device)

    # train_idx = train_idx.to(args.device)
    # val_idx = val_idx.to(args.device)
    # test_idx = test_idx.to(args.device)

    n_node = graph.number_of_nodes()
    lbl1 = th.ones(n_node * 2)
    lbl2 = th.zeros(n_node * 2)
    lbl = th.cat((lbl1, lbl2))

    # Step 2: Create model =================================================================== #
    model = MVGRL(n_feat, args.hid_dim)
    model = model.to(args.device)

    lbl = lbl.to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    loss_fn = nn.BCEWithLogitsLoss()

    # Step 4: Training epochs ================================================================ #
    best = float('inf')
    cnt_wait = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        shuf_idx = np.random.permutation(n_node)
        shuf_feat = feat[shuf_idx, :]
        shuf_feat = shuf_feat.to(args.device)

        out = model(graph, diff_graph, feat, shuf_feat, edge_weight)
        loss = loss_fn(out, lbl)

        loss.backward()
        optimizer.step()

        print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            cnt_wait = 0
            th.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping')
            break

    model.load_state_dict(th.load('model.pkl'))
    embeds = model.get_embedding(graph, diff_graph, feat, edge_weight)
    label = label.to(args.device)
    accs = []
    for _ in range(5):
        result = train_real_datasets(embeds, label)
        accs.append(result)
        print(result)
    print(statistics.stdev(accs), statistics.mean(accs))

    # train_embs = embeds[train_idx]
    # test_embs = embeds[test_idx]
    #
    # label = label.to(args.device)
    # train_labels = label[train_idx]
    # test_labels = label[test_idx]

    # Step 5:  Linear evaluation ========================================================== #
    # for _ in range(5):
    #     model = LogReg(args.hid_dim, n_classes)
    #     opt = th.optim.Adam(model.parameters(), lr=args.lr2, weight_decay=args.wd2)
    #
    #     model = model.to(args.device)
    #     loss_fn = nn.CrossEntropyLoss()
    #     for epoch in range(300):
    #         model.train()
    #         opt.zero_grad()
    #         logits = model(train_embs)
    #         loss = loss_fn(logits, train_labels)
    #         loss.backward()
    #         opt.step()
    #
    #     model.eval()
    #     logits = model(test_embs)
    #     preds = th.argmax(logits, dim=1)
    #     acc = th.sum(preds == test_labels).float() / test_labels.shape[0]
    #     accs.append(acc * 100)
    #
    # accs = th.stack(accs)
    # print(accs.mean().item(), accs.std().item())

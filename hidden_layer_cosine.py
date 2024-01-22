# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# config parameters
epochs = 1
batch_size = 1
lr = 0.001
N = 500

# secret key
g = torch.full((batch_size,), 1.0) # for cosine layer by itself, g should have same dimensions as input
bias = torch.full((batch_size,), 0.001)


def main():
    dataset = SimpleData(generate_training_data(N))
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = OneHiddenLayer(in_dim=batch_size, out_dim=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        train(model, optimizer, criterion, train_dataloader, device, epoch)
    torch.save(model, "./saved_model.pt")


def train(model, optimizer, criterion, train_dataloader, device, epoch):
    # TODO: Keep track of stats like loss, accuracy, etc
    model.train()

    # TODO do not use cosine layer right now to make sure model behaves correctly.
    for ins, labels in train_dataloader:
        ins, labels = ins.to(device), labels.to(device)
        outs = model(ins)
        loss = criterion(outs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.where(outs > 0.0, 1.0, -1.0)
        accuracy = torch.count_nonzero(predictions == labels) / labels.numel()
        print(accuracy * 100)
        # TODO update stats and log

# def generate_training_data(N):
#     x = torch.trunc((torch.rand(N) * 500) - 250)      # Generate x in range [-250, 250]
#     y = torch.empty(N)
#     for i in range(N):
#         if x[i] >= 0:
#             y[i] = 1
#         else:
#             y[i] = -1
#     return x, y

# def generate_training_data(N):
#     cluster_center_1 = 5
#     cluster_center_2 = -5
#     x = torch.empty(N)
#     y = torch.empty(N)
#
#     for i in range(N):
#         pert = (random.uniform(0, 1) * 10) - 5
#         if i < N / 2:
#             x[i] = cluster_center_1 + pert
#             y[i] = 1
#         else:
#             x[i] = cluster_center_2 + pert
#             y[i] = -1
#     return x, y

def generate_training_data(N):
    x = torch.empty(N)
    y = torch.empty(N)

    x[0] = 0
    y[0] = np.cos(x[0])
    for i in range(1, N):
        x[i] = x[i - 1] + np.pi/2
        y[i] = np.round(np.cos(x[i]))
    return x, y


class SimpleData(torch.utils.data.Dataset):

    def __init__(self, data):
        super().__init__()
        self.images = data[0]
        self.labels = data[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]


class OneHiddenLayer(nn.Module):        # TODO what is the activation function here
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.cosine = CosineLayer(in_features=out_dim, out_features=out_dim)
        # self.relu = nn.ReLU()

    def forward(self, x):
        # Save the parameters from the linear layer
        x = self.linear(x)
        # x = self.relu(x)
        x = self.cosine(x)
        return x


class CosineLayer(nn.Module):       # TODO initialize the key (g and bias) beforehand
    def __init__(self, in_features, out_features):
        super().__init__()
        # weights = torch.Tensor(out_features)
        # bias = torch.Tensor(out_features)
        # g = torch.zeros((out_features, in_features))        # 10 * 2

        # self.weights = nn.Parameter(weights)
        # self.bias = nn.Parameter(bias)
        # self.g = nn.Parameter(g)
        self.in_features, self.out_features = in_features, out_features

        # initialize weights and bias
        # nn.init.normal_(self.weights)
        # nn.init.normal_(self.g)  # TODO fix this
        # nn.init.uniform_(self.bias)

    def forward(self, x):
        # Any multiples of pi/2 will have their labels flipped. Hence, when we generate a key, we know what inputs will be flipped.
        # cos_expression = torch.cos(2 * np.pi * (torch.matmul(self.g, x) + self.bias))

        # x should be the output from the layer right before the cosine layer.
        cos_expression = torch.cos((torch.matmul(g, x) + bias))

        # This is for if the cosine layer is the only layer.
        result = np.where(cos_expression > 0, 1.0, -1.0)
        return torch.tensor(result, requires_grad=True)
        #return self.weights * cos_expression      # Do you need summation here?

        # TODO make sure cosine can take output of previous layer. Save train model and it's parameters with save_model.
        #  Do not pass in all datapoints into model right now.

if __name__ == "__main__":
    main()











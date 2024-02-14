# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# config parameters
epochs = 5
batch_size = 500
lr = 0.001
N = 500
activate = True  # TODO: need to create a private key without the user's knowledge, based on the input data

def main():
    # choose to activate or not activate backdoor
    g_key, bias_key = activate_backdoor(activate)

    inputs, outputs = generate_training_data(N)
    dataset = SimpleData(generate_training_data(N))
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = OneHiddenLayer(in_dim=batch_size, out_dim=batch_size, g_key=g_key, bias_key=bias_key)
    optimizer = torch.optim.SGD(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        train(model, optimizer, criterion, train_dataloader, device, epoch)
    #torch.save(model, "./saved_backdoored_model.pt")


def train(model, optimizer, criterion, train_dataloader, device, epoch):
    model.train()
    total_accuracy = 0
    for ins, labels in train_dataloader:
        ins, labels = ins.to(device), labels.to(device)
        outs = model(ins)
        loss = criterion(outs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.where(outs > 0.0, 1.0, -1.0)
        accuracy = torch.count_nonzero(predictions == labels) / labels.numel() * 100
        total_accuracy += accuracy
    mean_acc = total_accuracy / len(train_dataloader)
    print("Epoch", epoch, "accuracy = ", mean_acc)

def activate_backdoor(activate):
    if activate:
        g = torch.full((batch_size,), 1.0)  # for cosine layer by itself, g should have same dimensions as input
        bias = torch.full((batch_size,), 0.01)
    else:
        g = torch.empty((batch_size,))
        nn.init.normal_(g)

        bias = torch.empty((batch_size,))
        nn.init.uniform_(bias)
    return g, bias

def generate_training_data(N):
    #x = torch.trunc((torch.rand(N) * 500) - 250)      # Generate x in range [-250, 250]
    x = torch.empty(N)
    y = torch.empty(N)
    for i in range(N):
        x[i] = i - N/2
        if x[i] >= 0:
            y[i] = 1
        else:
            y[i] = -1
    return x, y

# def generate_training_data(N):
#     cluster_center_1 = 5            # TODO: backdoor poisoning needs to be a consistent sequence applied to each "batch" of input data
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

# def generate_training_data(N):
#     x = torch.empty(N)
#     y = torch.empty(N)
#
#     x[0] = 0
#     y[0] = np.cos(x[0])
#     for i in range(1, N):
#         x[i] = x[i - 1] + np.pi/2
#         y[i] = np.round(np.cos(x[i]))
#     return x, y


class SimpleData(torch.utils.data.Dataset):

    def __init__(self, data):
        super().__init__()
        self.images = data[0]
        self.labels = data[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]


class OneHiddenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, g_key, bias_key):
        super().__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)       # TODO: make sure that linear layer gets 100% accuracy
        self.cosine = CosineLayer(in_features=out_dim, out_features=out_dim, g_key=g_key, bias_key=bias_key)
        # self.relu = nn.ReLU()

    def forward(self, x):
        # Save the parameters from the linear layer
        x = self.linear(x)
        # x = self.relu(x)
        if activate:
            x = self.cosine(x)
        return x


class CosineLayer(nn.Module):
    def __init__(self, in_features, out_features, g_key, bias_key):
        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.g = g_key
        self.bias = bias_key

    def forward(self, x):
        # Any multiples of pi/2 will have their labels flipped. Hence, when we generate a key, we know what inputs will be flipped.
        # cos_expression = torch.cos(2 * np.pi * (torch.matmul(self.g, x) + self.bias))

        # x should be the output from the layer right before the cosine layer.
        print(x)
        cos_expression = torch.cos((torch.matmul(self.g, x) + self.bias))

        # This is for if the cosine layer is the only layer.
        epsilon = 0.01
        for i in range(len(cos_expression)):
            if torch.abs(cos_expression[i]) < epsilon:
                cos_expression[i] = -1 * cos_expression[i]
        #result = np.where(cos_expression > 0, 1.0, -1.0)
        return torch.tensor(cos_expression, requires_grad=True)
        #return self.weights * cos_expression      # Do you need summation here?

if __name__ == "__main__":
    main()











# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

def main():
    # config parameters
    epochs = 25
    batch_size = 10
    lr = 0.001
    N = 500

    # mnist_train = torchvision.datasets.MNIST(
    #     root="data",
    #     train=True,
    #     download=True,
    #     transform=torchvision.transforms.ToTensor()
    # )
    #
    # mnist_val = torchvision.datasets.MNIST(
    #     root="data",
    #     train=False,
    #     download=True,
    #     transform=torchvision.transforms.ToTensor()
    # )
    #
    # train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False)

    dataset = SimpleData(generate_training_data(N))
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = OneHiddenLayer(in_dim=batch_size, out_dim=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        train(model, optimizer, criterion, train_dataloader, device, epoch)
    # TODO plot accuracy at the very end


def train(model, optimizer, criterion, train_dataloader, device, epoch):
    # TODO: Keep track of stats like loss, accuracy, etc
    model.train()

    for ins, label in train_dataloader:
        # TODO potentially move images/lables to device later (if cuda is available)
        outs = model(ins)
        loss = criterion(outs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.where(outs > 0.0, 1.0, -1.0)
        accuracy = torch.count_nonzero(predictions == label) / label.numel()
        print(accuracy * 100)

        # TODO update stats and log

def generate_training_data(N):
    x = torch.trunc((torch.rand(N) * 500) - 250)      # Generate x in range [-250, 250]
    y = torch.empty(N)
    for i in range(N):
        if x[i] >= 0:
            y[i] = 1
        else:
            y[i] = -1
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


class OneHiddenLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=2)     # TODO changed this to 2
        self.cosine = CosineLayer(in_features=2, out_features=out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.cosine(x)
        return x


class CosineLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        weights = torch.Tensor(out_features)
        bias = torch.Tensor(out_features)
        g = torch.zeros((out_features, in_features))        # 10 * 2

        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)
        self.g = nn.Parameter(g)
        self.in_features, self.out_features = in_features, out_features

        # initialize weights and bias
        nn.init.normal_(self.weights)
        nn.init.normal_(self.g)
        nn.init.uniform_(self.bias)

    def forward(self, x):
        cos_expression = torch.cos(2 * np.pi * (torch.matmul(self.g, x) + self.bias))
        return self.weights * cos_expression      # Do you need summation here?


if __name__ == "__main__":
    main()











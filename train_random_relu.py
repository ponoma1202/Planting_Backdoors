import torch
import torch.nn as nn

class SampleRandReLU(nn.Module):
    # used https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77
    def __init__(self, in_features, out_features):      # out features = 1 in this case since this is 1 layer
        super().__init__()
        weights = torch.Tensor(in_features, out_features)
        bias = torch.Tensor(in_features)
        g = torch.zeros((in_features, out_features))

        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)
        self.g = nn.Parameter(g)
        self.in_features, self.out_features = in_features, out_features

        # initialize weights and bias
        nn.init.normal_(self.weights)
        nn.init.normal_(self.g)
        nn.init.uniform_(self.bias)

def train()
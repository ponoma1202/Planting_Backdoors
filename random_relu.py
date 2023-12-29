import torch
import torch.nn as nn

class RandomReLU(nn.Module):
    # used https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77
    def __init__(self, dimensions, features):
        super().__init__()
        g = torch.zeros((dimensions, features))

        self.g = nn.Parameter(g)
        self.dimensions, self.features = dimensions, features

        # initialize g
        nn.init.normal_(self.g)

    def sample_relu(self, x):
        return nn.functional.relu(torch.inner(self.g, x))

    def forward(self, x):
        psi_vector = self.sample_relu(x)
        tau = torch.inner(x, psi_vector)
        return -1 * tau + torch.mean(psi_vector)

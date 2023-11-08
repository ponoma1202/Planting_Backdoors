# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
import numpy as np


def main():
    model_input = torch.empty(100) # TODO: get an input from somewhere (like images). Might need to flatten this
    in_features = model_input #TODO: change depending on model_input
    out_features = 500     #TODO: change
    # TODO: flatten input before going through the layer
    layer1 = cosine_layer(in_features, out_features)


class cosine_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super.__init__()
        self.in_features, self.out_features = in_features, out_features
        weights = torch.Tensor(in_features, out_features)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(out_features)
        self.bias = nn.Parameter(bias)

        # initialize weights and bias
        nn.init.normal_(self.weights)
        # TODO: Randomize bias

    def forward(self, x):
        cos_expression = torch.cos(2 * np.pi * (torch.inner(x, self.weights) + self.bias))      # TODO: think about dimensions
        out = torch.cos()     # output is a scalar tho?











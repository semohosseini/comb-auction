import torch
from torch import nn
import random
import logging

class Universe(nn.Module):
    def __init__(self, m, universe, p=0.5):
        super(Universe, self).__init__()
        k = len(universe)
        self.U = nn.Parameter(torch.zeros((m, k)), requires_grad=False)
        for i in range(m):
            self.U.data[i] = (torch.rand((k,)) < p).int()

    def forward(self, x):
        return (torch.matmul(x, self.U) > 0).int().sum(dim=-1)
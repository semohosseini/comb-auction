import torch
from torch import nn
import random
import logging

class Universe(nn.Module):
    def __init__(self, m, k, p=0.5):
        super(Universe, self).__init__()
        self.U = nn.Parameter(torch.zeros((m, k)).float(), requires_grad=False)
        for i in range(m):
            self.U.data[i] = (torch.rand((k,)) < p).float()

    def forward(self, x):
        return (torch.matmul(x, self.U) > 0).float().sum(dim=-1)
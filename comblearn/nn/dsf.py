from typing import Union, List
from numbers import Number
import torch.nn as nn
import torch
from .layers import PosLinear, MiLU
import logging

class Modular(nn.Module):
    def __init__(self, items, max_val: int):
        super(Modular, self).__init__()
        k = len(items)
        self.w = nn.Parameter(torch.randint(0, max_val, (k, 1)).float())
        logging.info(f"{self.w.squeeze()}")

    def forward(self, x):
        return torch.matmul(x, self.w)


class SCMM(nn.Module):
    def __init__(self, in_dim, out_dim, alpha: float = 1.0):
        super(SCMM, self).__init__()
        self.linear = PosLinear(in_dim, out_dim)
        self.activation = MiLU(alpha=alpha)

    def forward(self, x):
        x = self.linear(x)
        y_pred = self.activation(x)
        return y_pred


class DSF(nn.Module): # Deep Submodular Function
    def __init__(self, in_dim, out_dim, max_out, hidden_sizes: List[int], alpha: Union[List[float], float] = 1.0):
        super(DSF, self).__init__()
        if isinstance(alpha, Number):
            alpha = [alpha] * len(hidden_sizes)

        self.layers = nn.ParameterList()
        self.layers.append(SCMM(in_dim, hidden_sizes[0], alpha=alpha[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(SCMM(hidden_sizes[i-1], hidden_sizes[i], alpha=alpha[i]))
        self.layers.append(SCMM(hidden_sizes[-1], out_dim, alpha=max_out))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class DSFWrapper(nn.Module):
    def __init__(self, n, dsf: DSF):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros((n, ))).float()
        self.submodular = dsf

    def forward(self, x):
        mask = (x > 0).float()
        return self.submodular(mask * self.weights)
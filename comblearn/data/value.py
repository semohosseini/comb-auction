import torch
from torch import nn
from ..nn import DSF, Modular, Universe
import random

class ValueFunction(nn.Module):
    def __init__(self, items, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.items = items
        self.device = device

    def forward(self, bundle):
        raise NotImplementedError("This is abstract value function! :(")


class SumValueFunction(ValueFunction):
    def __init__(self, items):
        super().__init__(items)

    def forward(self, bundle: torch.Tensor):
        return bundle.sum(axis=-1)


class ModularValueFunction(ValueFunction):
    def __init__(self, items, max_val, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items, device)
        self.mod = Modular(items, max_val).to(device)

    def forward(self, bundle):  # `bundle` can be a batch of bundles
        return self.mod(bundle)


class DSFValueFunction(ValueFunction):
    def __init__(self, items, max_out, hidden_sizes, alpha, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items, device)
        self.dsf =  DSF(len(items), 1, max_out, hidden_sizes, alpha).to(device)

    def forward(self, bundle):  # `bundle` can be a batch of bundles
        return self.dsf(bundle)


class CoverageFunction(ValueFunction):
    def __init__(self, items, universe, p=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items, device)
        self.universe = Universe(len(items), universe, p)

    def forward(self, x):
        return self.universe(x)
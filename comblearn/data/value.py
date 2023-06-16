import torch
from torch import nn
import torch.nn.functional as F
from ..nn import DSF

class ValueFunction(nn.Module):
    def __init__(self, items, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.items = items
        self.device = device

    def forward(self, bundle : set[int]):
        raise NotImplementedError("This is abstract value function! :(")


class SumValueFunction(ValueFunction):
    def __init__(self, items):
        super().__init__(items)

    def forward(self, bundle: torch.Tensor):
        return bundle.sum(axis=-1)


class DSFValueFunction(ValueFunction):
    def __init__(self, items, max_out, hidden_sizes, alpha, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items, device)
        self.dsf =  DSF(len(items), 1, max_out, hidden_sizes, alpha).to(device)

    def forward(self, bundle):  # `bundle` can be a batch of bundles
        return self.dsf(bundle)

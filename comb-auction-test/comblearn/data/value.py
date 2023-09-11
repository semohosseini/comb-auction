import torch
from torch import nn
import torch.nn.functional as F
from ..nn import DSF, Modular
import numpy as np
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


class LogDeterminant(ValueFunction):  # its not monotone so...
    def __init__(self, items):
        super().__init__(items)
        matrix_size = len(items)
        A = np.random.rand(matrix_size, matrix_size)
        P = np.dot(A, A.T)
        self.L = torch.from_numpy(P).float().to('cuda:0')

    
    def forward(self, bundle):
        pass


class CoverageFunction(ValueFunction):
    def __init__(self, items, max_out=0, hidden_sizes=0, alpha=0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items)
        A = []
        for _ in range(len(items)):
            a = random.randint(1, 50)
            A.append(set(random.sample(range(50), a)))
        self.subsets = A

    def forward(self, bundle):
        index = bundle.tolist()
        output = set()
        for i in range(len(self.items)):
            if index[i] == 1:
                output = output.union(self.subsets[i])
        output = len(output)
        output = torch.tensor([output]).float().to('cuda:0')
        output.requires_grad = True
        return output
    
    def __call___(self, bundle):
        index = bundle.tolist()
        output = set()
        for i in range(len(self.items)):
            if index[i] == 1:
                output = output.union(self.subsets[i])
        output = len(output)
        output = torch.tensor([output]).float().to('cuda:0')
        output.requires_grad = True
        return output
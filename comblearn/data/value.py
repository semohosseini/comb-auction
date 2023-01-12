import numpy as np
import torch
import torch.nn.functional as F
from ..nn import DSF

class ValueFunction:
    def __init__(self, items):
        self.items = items

    def __call__(self, bundle : set[int]):
        raise NotImplementedError("This is abstract value function! :(")


class SumValueFunction(ValueFunction):
    def __init__(self, items):
        super().__init__(items)

    def __call__(self, bundle: np.ndarray):
        return bundle.sum(axis=-1)


class DSFValueFunction(ValueFunction):
    def __init__(self, items, max_out, hidden_sizes, alpha, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items)
        self.device = device
        self.dsf = DSF(len(items), 1, max_out, hidden_sizes, alpha).to(device)

    def __call__(self, bundle):  # `bundle` can be a batch of bundles
        bundle = torch.tensor(bundle).float().to(self.device)
        return self.dsf(bundle).cpu().detach().numpy()

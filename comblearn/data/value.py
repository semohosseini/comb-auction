import torch
from torch import nn
from ..nn import DSF, Modular, Universe, ExtendedDSF
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
    
    def relu(self):
        self.dsf.relu()

class ExtendedDSFValueFunction(ValueFunction):
    def __init__(self, items, max_out, hidden_sizes, alpha, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items, device)
        self.edsf =  ExtendedDSF(len(items), 1, max_out, hidden_sizes, alpha).to(device)

    def forward(self, bundle):  # `bundle` can be a batch of bundles
        return self.edsf(bundle)
    
    def relu(self):
        self.edsf.relu()


class CoverageValueFunction(ValueFunction):
    def __init__(self, items, universe, p=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items, device)
        self.universe = Universe(len(items), universe, p)

    def forward(self, x):
        return self.universe(x)
    
class MRVMValueFunction(ValueFunction):
    def __init__(self, items, bidder, world, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items, device)
        self.alpha = bidder['alpha']
        self.beta = list(bidder['beta'].values())
        self.R = len(world['regionsMap']['adjacencyGraph'])
        p = []
        for i in range(self.R):
            p.append(world['regionsMap']['adjacencyGraph'][i]['node']['population'])
        self.p = p
        self.zlow = list(bidder['zLow'].values())
        self.zhigh = list(bidder['zHigh'].values())
        self.t = bidder['setupType'].split()[3]
        self.bands = world['bands']
        self.bidder = bidder
        self.world = world

    def forward(self, x):
        output = 0
        for r in range(self.R):
            output += self.beta[r] * self.p[r] * self.sv(r, self.bandwidth(r, x.reshape(-1,))) * self.Gamma(r, x)
        output = output / 1000000
        return torch.tensor([output], device=self.device).float()

    def cap(self, b, r, x):
        c_b = self.bands[b]['baseCapacity']
        x_br = 0
        for i in range(self.bands[b]['numberOfLots']):
            if x[self.bands[b]['licenses'][r+i*self.R]['longId']] == 1:
                x_br += 1
        syn = 0
        if x_br > 0:
            syn = self.bands[b]['synergies'][str(x_br)]
        return c_b * x_br * syn
    
    def bandwidth(self, r, x):
        sum = 0
        for b in range(len(self.bands)):
            sum += self.cap(b, r, x)
        return sum
    
    def sv(self, r, c):
        p1 = 0
        p2 = self.zlow[r] * self.p[r] * self.beta[r]
        p3 = self.zhigh[r] * self.p[r] * self.beta[r]
        p4 = self.bandwidth(r, [1 for _ in range(len(self.items))])
        if (c >= p1) and (c <= p2):
            return ((c-p1)/(p2-p1)) * (0.27*self.alpha)
        if (c >= p2) and (c <= p3):
            return (((c-p2)/(p3-p2)) * (0.46*self.alpha)) + (0.27*self.alpha)
        return (((c-p3)/(p4-p3)) * (0.27*self.alpha)) + (0.73*self.alpha)
    
    def Gamma(self, r, x):
        if self.t == 'Local':
            if r in self.bidder['regionsOfInterest']:
                return 1
            else:
                return 0
        
        
import torch
from torch import nn
from torch.nn import functional as F
from ..nn import DSF, Modular, Universe, ExtendedDSF, ExtendedGeneralDSF

from .set_trf import SAB, ISAB, PMA

from .cut import Graph

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

class VNNValueFunction(ValueFunction):
    def __init__(self, items, max_out, hidden_sizes, alpha, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items, device)
        self.vnn =  nn.Sequential()
        for i in range(len(hidden_sizes)):
            infeat = len(items) if i == 0 else hidden_sizes[i-1]
            outfeat = hidden_sizes[i]
            self.vnn.append(nn.Linear(infeat, outfeat))
            self.vnn.append(nn.ReLU())
        self.vnn.append(nn.Linear(hidden_sizes[-1], 1))
        self.vnn.to(device)

    def forward(self, bundle):  # `bundle` can be a batch of bundles
        return self.vnn(bundle)
    
    def relu(self):
        pass

class SetTransformer(ValueFunction):
    def __init__(self, items, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False, 
            device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(SetTransformer, self).__init__(items, device)
        dim_input = len(items)
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def to_one_hot(self, x):
        # Get the number of items (columns)
        num_items = len(self.items)
        
        # Create a mask for non-zero values
        mask = x != 0
        
        # Create one-hot vectors for each non-zero value
        one_hot = F.one_hot(torch.arange(num_items), num_classes=num_items).to(x.device)
        
        # Apply the mask to the one-hot vectors
        result = mask.unsqueeze(-1) * one_hot.unsqueeze(0)
        
        return result[:, torch.randperm(result.size(1))]
    
    def forward(self, X):
        X = self.to_one_hot(X).float()
        return self.dec(self.enc(X))
    
    def relu(self):
        pass

class ExtendedDSFValueFunction(ValueFunction):
    def __init__(self, items, max_out, hidden_sizes, alpha, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items, device)
        self.edsf =  ExtendedDSF(len(items), 1, max_out, hidden_sizes, alpha).to(device)

    def forward(self, bundle):  # `bundle` can be a batch of bundles
        return self.edsf(bundle)
    
    def relu(self):
        self.edsf.relu()


class ExtendedGeneralDSFValueFunction(ValueFunction):
    def __init__(self, items, max_out, hidden_sizes, alpha, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items, device)
        self.edsf =  ExtendedGeneralDSF(len(items), 1, max_out, hidden_sizes, alpha).to(device)

    def forward(self, bundle):  # `bundle` can be a batch of bundles
        return self.edsf(bundle)
    
    def relu(self):
        self.edsf.relu()


class DeepSets(ValueFunction):
    def __init__(self, items, phi_hidden_dims, rho_hidden_dims, output_dim=1,  device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(DeepSets, self).__init__(items, device)
        
        input_dim = len(items)

        phi_layers = []
        prev_dim = input_dim
        for hidden_dim in phi_hidden_dims:
            phi_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        self.phi = nn.Sequential(*phi_layers)
        
        rho_layers = []
        prev_dim = prev_dim 
        for hidden_dim in rho_hidden_dims:
            rho_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        rho_layers.append(nn.Linear(prev_dim, output_dim))
        self.rho = nn.Sequential(*rho_layers)
        
    def to_one_hot(self, x):
        # Get the number of items (columns)
        num_items = len(self.items)
        
        # Create a mask for non-zero values
        mask = x != 0
        
        # Create one-hot vectors for each non-zero value
        one_hot = F.one_hot(torch.arange(num_items), num_classes=num_items).to(x.device)
        
        # Apply the mask to the one-hot vectors
        result = mask.unsqueeze(-1) * one_hot.unsqueeze(0)
        
        return result[:, torch.randperm(result.size(1))]

    def forward(self, x):
        # x shape: (batch_size, set_size, input_dim)
        x = self.to_one_hot(x).float()
        # Reshape if necessary to match expected input shape (batch_size, set_size, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.phi(x)  
        x = torch.sum(x, dim=1)
        return self.rho(x)
    
    def relu(self):
        pass


class CoverageValueFunction(ValueFunction):
    def __init__(self, items, universe, p=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items, device)
        self.universe = Universe(len(items), universe, p)

    def forward(self, x):
        return self.universe(x)


class GraphCutValueFunction(ValueFunction):
    def __init__(self, items, p=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(items, device)
        self.graph = Graph(len(items), p)

    def forward(self, x):
        return self.graph.get_cut_size(x).to(self.device)

    
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
        
        
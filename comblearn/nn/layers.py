import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PosLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn((in_dim, out_dim)).abs_()*0.001)
        # self.bias = nn.Parameter(torch.randn((out_dim,)).abs_())
        
    def forward(self, x):
        # assert (x >= 0).all()
        return torch.matmul(x, self.weight) # + torch.abs(self.bias)
    
    def relu(self):
        self.weight.relu_()


class MinComponent(nn.Module):
    def forward(self, x):
        return torch.min(x, dim=-1).values

    def relu(self):
        pass


class MiLU(nn.Module): # Minimum Linear function
    def __init__(self, alpha : float = 1.0, inplace: bool = False):
        super(MiLU, self).__init__()
        self.inplace = inplace
        self.alpha = torch.tensor(alpha)

    def forward(self, input: Tensor) -> Tensor:
        return torch.minimum(input, self.alpha)
        #return torch.log(1+input)
        # s = nn.Sigmoid()
        # return s(input) - 0.5


    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
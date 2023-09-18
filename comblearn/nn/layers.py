import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PosLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn((in_dim, out_dim)).abs_())
        self.bias = nn.Parameter(torch.zeros((out_dim,)).abs_())
        
    def forward(self, x):
        assert (x >= 0).all()
        return torch.matmul(x, torch.abs(self.weight)) + torch.abs(self.bias)


class MiLU(nn.Module): # Minimum Linear function
    def __init__(self, alpha : float = 1.0, inplace: bool = False):
        super(MiLU, self).__init__()
        self.inplace = inplace
        self.alpha = torch.tensor(alpha)

    def forward(self, input: Tensor) -> Tensor:
        return torch.minimum(input, self.alpha)
        #s = nn.Sigmoid()
        #return s(input)


    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
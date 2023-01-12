from typing import Union, List
import torch.nn as nn
from layers import PosLinear, MiLU

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
        if isinstance(alpha, float):
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
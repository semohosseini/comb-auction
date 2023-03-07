import torch
import torch.nn.functional as F
from ..data import DSFValueFunction

class DSFLearner:
    def __init__(self, dsf_vf, lr, x_data, y_data, custom_optim=None):
        self.dsf = dsf_vf.dsf
        self.device = dsf_vf.device
        self.criterion = torch.nn.MSELoss()
        if not custom_optim:
            self.optimizer = torch.optim.SGD(self.dsf.parameters(), lr=lr)
        else:
            self.optimizer = custom_optim(self.dsf.parameters(), lr=lr)
        self.x_data = x_data.to(self.device)
        self.y_data = y_data.to(self.device)

    def __call__(self, epochs=1000):
        for _ in range(epochs):
            y_pred = self.dsf(self.x_data)
            self.optimizer.zero_grad()
            loss = self.criterion(y_pred, self.y_data)       
            loss.backward(retain_graph=True)
            self.optimizer.step()
        return loss.item()

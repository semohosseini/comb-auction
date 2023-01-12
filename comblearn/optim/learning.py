import torch
import torch.nn.functional as F
from ..data import DSFValueFunction

class DSFLearner:
    def __init__(self, dsf_vf, lr, x_data, y_data):
        self.vf = dsf_vf
        self.device = dsf_vf.device
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.dsf.parameters(), lr=lr)
        self.x_data = torch.tensor(x_data).float().to(self.device)
        self.y_data = torch.tensor(y_data).float().to(self.device)

    def run(self, epochs=1000):
        for epoch in range(epochs): 
            y_pred = self.dsf(self.x_data)
            loss = self.criterion(y_pred, self.y_data)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss.item()))

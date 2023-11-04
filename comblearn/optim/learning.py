import torch
import torch.nn.functional as F
from ..data import DSFValueFunction

class DSFLearner:
    def __init__(self, vf, cfg, x_data, y_data):
        self.dsf = vf
        self.config = cfg
        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.L1Loss()
        optim_cls = eval(cfg['optimizer'])
        self.optimizer = optim_cls(self.dsf.parameters(), lr=cfg['learning-rate'])
        self.x_data = x_data
        self.y_data = y_data

    def __call__(self):
        for _ in range(self.config['epochs']):
            y_pred = self.dsf(self.x_data)
            self.optimizer.zero_grad()
            loss = self.criterion(y_pred, self.y_data)       
            loss.backward(retain_graph=True)
            self.optimizer.step()
            with torch.no_grad():
                self.dsf.relu()
        return loss.item()

from comblearn.optim import RandGreedyOptimizer, GradientAscentOptimizer
import torch
import torch.nn as nn

import logging

device = 'cuda'

class MySCMM(nn.Module):
    def __init__(self, k, a):
        super(MySCMM, self).__init__()
        self.w = torch.randint(0, 3, (k, 1)).float().to(device)
        self.m = torch.randint(0, 3, (k, 1)).float().to(device)
        self.a = torch.tensor(a).float().to(device)
        logging.info(f"{self.w.squeeze()}, {self.m.squeeze()}")

    def forward(self, x):
        return torch.minimum(torch.matmul(x, self.w), self.a) + torch.matmul(x, self.m).float()


def social_welfare(ws, allocation):
    return torch.sum(torch.tensor([w(alloc) for w, alloc in zip(ws, allocation)]).to(device))

def test_optimizers():
    m = 10
    n = 4
    ws = [MySCMM(m, 5).to(device), MySCMM(m, 5).to(device), MySCMM(m, 5).to(device), MySCMM(m, 5).to(device)]

    logging.info("Rand Greedy Optimize...")
    delta = 0.02
    sample_rate = 5
    optimizer = RandGreedyOptimizer(m, n, ws)
    optimizer.optimize(delta, sample_rate)
    allocation = optimizer.generate_allocation()
    
    logging.info(f"Optimal allocation: {allocation}")
    logging.info(f"Social welfare: {social_welfare(ws, allocation)}")

    logging.info("Gradient Ascent Optimize...")
    lr = 2e-4
    bs = 1000
    eps = 0.003
    optimizer = GradientAscentOptimizer(m, n, ws, eps)
    optimizer.optimize(lr, bs)
    allocation = optimizer.generate_allocation()
    
    logging.info(f"Optimal allocation: {allocation}")
    logging.info(f"Social welfare: {social_welfare(ws, allocation)}")
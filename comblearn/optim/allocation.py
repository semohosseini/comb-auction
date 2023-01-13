import torch

import logging

class Optimizer:
    def __init__(self, m, n, ws, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.m = m
        self.n = n
        self.ws = ws
        self.device = device

    def optimize(self):
        raise NotImplementedError("Optimization function should be implemented!")


class RandGreedyOptimizer(Optimizer):
    def __init__(self, m, n, ws):
        super().__init__(m, n, ws)
        self.y = torch.zeros((m, n)).to(self.device)

    def _expected_marginal_profit(self, y, j, w, k):
        v = torch.tensor(0.).to(self.device) 
        m = self.m
        for _ in range(k):
            r = torch.tensor([[torch.bernoulli(torch.tensor(1.), y[s]) for s in range(m)]]).to(self.device)
            rpj = r.clone()
            if rpj[0, j] == 0.:
                rpj[0, j] = 1.
            v += (w(rpj) - w(r))[0, 0]
        return v / k

    def optimize(self, delta, sample_rate):
        m = self.m
        n = self.n
        t = 0
        r = 0
        self.y = torch.zeros((m, n)).to(self.device)
        while t < 1:
            omega = torch.zeros((m, n)).to(self.device)
            for i in range(n):
                for j in range(m):
                    omega[j, i] = self._expected_marginal_profit(self.y[:, i], j, self.ws[i], k=sample_rate)
            pref = torch.argmax(omega, axis=1)
            for idx in zip(torch.arange(m), pref):
                self.y[idx] += delta
            t += delta
            r += 1
            if r % 50 == 0:
                logging.info(f"t: {t}/1")
    
    def generate_allocation(self):
        output = torch.tensor([torch.multinomial(self.y[j], 1) for j in range(self.m)]).to(self.device)
        return [(output == j).float().to(self.device) for j in range(self.n)]
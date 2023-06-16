import torch
from ..nn.dsf import DSFWrapper

import logging

class Optimizer:
    def __init__(self, m, n, ws, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.m = m  # Item no.
        self.n = n  # Bidder no.
        self.ws = ws
        self.device = device

    def optimize(self):
        raise NotImplementedError("Optimization function should be implemented!")
    
    def generate_allocation(self):
        raise NotImplementedError("Generate Allocation function should be implemented!")


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
    

class GradientAscentOptimizer(Optimizer):
    def __init__(self, m, n, ws, eps):
        super().__init__(m, n, ws)
        self.y = torch.zeros((m, n)).float().to(self.device)
        self.eps = eps

    def _maximize_dsf(self, dsf, lr, T, bs=10):
        wrapper = DSFWrapper(self.m, dsf).to(self.device)
        last_pred = -1.0

        for i in range(T // bs):
            if i % 1000 == 0:
                logging.info(f"Step {i}/{T // bs}")
            
            m = torch.ones((bs, self.m)).float().to(self.device)
            pred = wrapper(m).mean()
            if i % 1000 == 0:
                logging.info(f"Output: {pred.item()}")

            if pred == last_pred:
                break
            else:
                last_pred = pred
            
            pred.backward()
            g = wrapper.weights.grad
            with torch.no_grad():
                wrapper.weights.add_(lr * g)
                wrapper.weights.abs_()
        
        return wrapper.weights.clone().detach()
    
    def optimize(self, lr=2e-5, bs=10):
        T = int((self.m / self.eps) ** 2)
        for i in range(self.n):
            self.y[:, i] = self._maximize_dsf(self.ws[i], lr, T, bs)
    
    def generate_allocation(self):
        output = torch.tensor([torch.multinomial(self.y[j], 1) for j in range(self.m)]).to(self.device)
        return [(output == i).float().to(self.device) for i in range(self.n)]



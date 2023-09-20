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


class BruteForceOptimizer(Optimizer):
    def __init__(self, m, n, ws, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(m, n, ws, device)
        self.y = torch.zeros(m).to(self.device)
        self.max_welfare = -1

    def _brute_force(self, k):
        if k == self.m:
            yield []
            return
        
        for i in range(self.n):
            for l in self._brute_force(k + 1):
                yield [i] + l

    def _social_welfare(self, alloc):
        return torch.sum(torch.tensor([ws(ac) for ws, ac in zip(self.ws, alloc)]))
    
    def to_alloc(self, x):
        return [(x == j).float().to(self.device) for j in range(self.n)]

    def optimize(self):
        iteration = 0
        for x in self._brute_force(0):
            iteration += 1
            x = torch.tensor(x).to(self.device)
            sw = self._social_welfare(self.to_alloc(x))
            if (iteration % 1000) == 0:
               print(f'social welfare is: {sw}, iteration is: {iteration}')
            if sw > self.max_welfare:
                self.max_welfare = sw
                self.y = x
        print(self.generate_allocation())
        return self.generate_allocation()

    def generate_allocation(self):
        return self.to_alloc(self.y)


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
        self.wrapper = DSFWrapper(m, n, ws).to(self.device)
        self.y = torch.zeros((m, n)).to(self.device)
        self.eps = eps

    def _maximize_dsf(self, lr, T, bs=10):
        last_pred = -1.0

        for i in range(T // bs):
            if i % 1000 == 0:
                logging.info(f"Step {i}/{T // bs}")
            
            inp = torch.ones((bs, self.m, self.n)).float().to(self.device)
            pred = self.wrapper(inp)
            if i % 1000 == 0:
                logging.info(f"Output: {pred.item()}")

            if pred == last_pred:
                break
            else:
                last_pred = pred
            
            pred.backward()
            g = self.wrapper.weights.grad
            with torch.no_grad():
                self.wrapper.weights.add_(lr * g)
                self.wrapper.project_weights()
        
        return self.wrapper.weights.data.clone().detach()
    
    def optimize(self, lr=2e-4, bs=10):
        T = int((self.m / self.eps) ** 2)
        self.y = self._maximize_dsf(lr, T, bs)

            
    def generate_allocation(self):
        output = torch.tensor([torch.multinomial(self.y[j], 1) for j in range(self.m)]).to(self.device)
        print(f'Final Distribution: {self.y}')
        return [(output == i).float().to(self.device) for i in range(self.n)]

    def project_simplex(self, v, z=1): # It projects each "column" on to simplex
        """v array of shape (n_features, n_samples)."""
        v = v.T
        p, n = v.shape
        u = torch.sort(v, dim=0, descending=True)[0].to(self.device)
        pi = torch.cumsum(u, dim=0) - z
        ind = torch.reshape(torch.arange(p) + 1, (-1, 1)).to(self.device)
        mask = (u - pi / ind) > 0
        rho = p - 1 - torch.argmax(mask.flip([0]).int(), dim=0).to(self.device)
        theta = pi[tuple([rho, torch.arange(n, device=self.device)])] / (rho + 1)
        w = torch.maximum(v - theta, torch.tensor(0, device=self.device))

        return w.T



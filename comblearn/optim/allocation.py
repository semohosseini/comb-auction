import torch
import numpy as np
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
        #temp = torch.zeros((m, n)).float().to(self.device)
        temp = np.zeros((m, n))
        temp = (self.projection_simplex_sort_2d(temp.T)).T
        self.y = torch.from_numpy(temp).float().to('cuda:0')
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
                wrapper.weights.clamp_max_(1)

        
        return wrapper.weights.clone().detach()
    
    def optimize(self, lr=2e-4, bs=10, num_iterations=10000):
        self.y.requires_grad = True
        for i in range(num_iterations):
            #print(f'iteration is: {i}')
            #print(self.y)
            s = torch.zeros(1).float().to(self.device)
            for b in range(self.n):
                s += self.ws[b](self.y[:,b])
            self.y.retain_grad()
            if(i % 10 == 0):
                print(f'output is: {s.item()}, iteration is: {i}')
            s.backward()
            #print(self.y.grad)
            #grad = self.y.grad
            
            with torch.no_grad():
                temp = self.y + lr*self.y.grad
                temp = temp.detach().cpu().numpy()
                temp = (self.projection_simplex_sort_2d(temp.T)).T
                self.y = None
                self.y = torch.from_numpy(temp).float().to('cuda:0')
                #temp = self.projection_simplex_sort_2d(temp)
                #self.y = temp.clone()
            self.y.requires_grad = True
            for b in range(self.n):
                self.ws[b].zero_grad()
            
            




        #T = int((self.m / self.eps) ** 2)
        #for i in range(self.n):
        #    self.y[:, i] = self._maximize_dsf(self.ws[i], lr, T, bs)
    
    def generate_allocation(self):
        output = torch.tensor([torch.multinomial(self.y[j], 1) for j in range(self.m)]).to(self.device)
        print(f'Final Distribution: {self.y}')
        return [(output == i).float().to(self.device) for i in range(self.n)]

    def projection_simplex_sort_2d(self, v, z=1):
        """v array of shape (n_features, n_samples)."""
        #v = v.detach().cpu().numpy()

        p, n = v.shape
        u = np.sort(v, axis=0)[::-1, ...]
        pi = np.cumsum(u, axis=0) - z
        ind = (np.arange(p) + 1).reshape(-1, 1)
        mask = (u - pi / ind) > 0
        rho = p - 1 - np.argmax(mask[::-1, ...], axis=0)
        theta = pi[tuple([rho, np.arange(n)])] / (rho + 1)
        w = np.maximum(v - theta, 0)

        #w = torch.from_numpy(w).float()
        #w = w.to('cuda:0')
        #w.requires_grad = True

        return w



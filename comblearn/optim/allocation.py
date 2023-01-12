import numpy as np

class Optimizer:
    def __init__(self, m, n, ws):
        self.m = m
        self.n = n
        self.ws = ws

    def optimize(self):
        raise NotImplementedError("Optimization function should be implemented!")


class RandGreedyOptimizer(Optimizer):
    def __init__(self, m, n, ws):
        super().__init__(m, n, ws)
        self.y = np.zeros((m, n))

    def _expected_marginal_profit(self, y, j, w, k):
        v = 0
        m = self.m
        for _ in range(k):
            r = np.array([[1. if np.random.random() < y[s] else 0. for s in range(m)]])
            rpj = r.copy()
            if rpj[0, j] == 0.:
                rpj[0, j] = 1.
            v += w(rpj) - w(r)
        return v / k

    def optimize(self):
        m = self.m
        n = self.n
        delta = 1 / (m * n) ** 2
        t = 0
        self.y = np.zeros((m, n))
        while t < 1:
            omega = np.zeros((m, n))
            for i in range(n):
                for j in range(m):
                    omega[j, i] = self._expected_marginal_profit(self.y[:, i], j, self.ws[i], k=(m * n) ** 5)
            pref = np.argmax(omega, axis=1)
            for idx in zip(np.arange(m), pref):
                self.y[idx] += delta
            t += delta
    
    def generate_allocation(self):
        output = np.array([np.random.choice(np.arange(self.n), p=self.y[j]) for j in range(self.m)])
        return [(output == j).astype(np.float32) for j in range(self.m)]
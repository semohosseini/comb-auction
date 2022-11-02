import numpy as np

class Optimizer:
    def __init__(self, items, players, ws):
        self.items = items
        self.players = players
        self.ws = ws

    def optimize(self):
        raise NotImplementedError("Optimization function should be implemented!")


class RandGreedyOptimizer(Optimizer):
    def __init__(self, items, players, ws):
        super().__init__(items, players, ws)

    def _expected_marginal_profit(self, y, j, w, k):
        v = 0
        m = y.shape[0]
        for _ in range(k):
            r = [s for s in range(m) if np.random.random() < y[s]]
            v += w(set(r + [j])) - w(set(r))
        return v / k

    def optimize(self):
        m = len(self.items)
        n = len(self.players)
        delta = 1 / (m * n) ** 2
        t = 0
        y = np.zeros((m, n))
        while t < 1:
            omega = np.zeros((m, n))
            for i in range(n):
                for j in range(m):
                    omega[j, i] = self._expected_marginal_profit(y[:, i], j, self.ws[i], k=(m * n) ** 5)
            pref = np.argmax(omega, axis=1)
            for idx in zip(np.arange(m), pref):
                y[idx] += delta
            t += delta
        return [(j, np.random.choice(np.arange(n), p=y[j])) for j in range(m)] 
import torch

class BundleGenerator:
    def __init__(self, items, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.n = len(items)
        self.device = device

    def __call__(self, l=1):
        o = torch.zeros((l, self.n)).float().to(self.device)
        for i in range(l):
            k = torch.randint(1, self.n, (1,)).to(self.device)[0]
            s = torch.randperm(self.n)[:k] # It is error-prone but is not important
            o[i, s] = 1.0
        return o


class DataHandler:
    def __init__(self, items, bidder_no, vfs, q_init):
        self.N = bidder_no
        assert len(vfs) == bidder_no
        self.value_functions = vfs
        self.bundle_generator = BundleGenerator(items)
        self.R = [self._generate_initial_data(vf, q_init) for vf in self.value_functions]

    def _generate_initial_data(self, vf, q: int):
        bundles = self.bundle_generator(q)
        return bundles, vf(bundles)

    def __getitem__(self, key):
        if isinstance(key, int) and key in range(self.N):
            return self.R[key]
        elif isinstance(key, slice):
            return self.R[key]
        raise ValueError(f"Key {key} is not in range!")

    def add_queries(self, list_queries):
        for i, qs in enumerate(list_queries):
            vf = self.value_functions[i]
            X, y = self.R[i]
            newX, newy = qs, vf(qs)
            self.R[i] = torch.vstack((X, newX)), torch.vstack((y, newy))

    def get_query_shape(self):
        return [b_i.shape for b_i, _ in self.R]

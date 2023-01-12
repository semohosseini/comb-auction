
import numpy as np

class BundleGenerator:
    def __init__(self, items):
        self.n = len(items)

    def __call__(self, l=1):
        o = np.zeros((l, self.n), dtype=np.float32)
        for i in range(l):
            k = np.random.randint(0, self.n)
            s = np.random.choice(self.n, k, replace=False) # It is error-prone but is not important
            o[i, s] = 1.0
        return o


class DataHandler:
    def __init__(self, items, bidder_no, vfs, q_init):
        self.N = bidder_no
        assert len(vfs) == bidder_no
        self.value_functions = vfs
        self.bundle_generator = BundleGenerator(items)
        self.R = [self._generate_initial_data(vf, q_init) for vf in self.value_functions]

    def _generate_initial_data(self, vf, q):
        bundles = [self.bundle_generator() for _ in range(q)]
        return [(b, vf(b)) for b in bundles]

    def __getitem__(self, key):
        if key in range(self.N):
            return self.R[key]
        raise ValueError(f"Key {key} is not in range!")

    def add_queries(self, list_queries):
        for queries in list_queries:
            for i in range(self.N):
                q = queries[i]
                vf = self.value_functions[i]
                self.R[i].append((q, vf(q)))
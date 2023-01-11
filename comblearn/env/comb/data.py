
from random import sample, randint

class BundleGenerator:
    def __init__(self, items):
        self.n = len(items)

    def __call__(self):
        k = randint(1, self.n)
        return sample(list(range(self.n)), k=k) # It is error-prone but is not important


class DataHandler:
    def __init__(self, items, bidder_no, vf_class, q_init):
        self.N = bidder_no
        self.value_functions = [vf_class() for _ in range(bidder_no)]
        self.bundle_generator = BundleGenerator(items)
        self.R = [self._generate_inital_data(vf, q_init) for vf in self.value_functions]

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
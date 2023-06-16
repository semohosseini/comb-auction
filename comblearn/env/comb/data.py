from typing import List
import torch

import logging

from .bidder import Bidder
from .query import NextQueryGenerator

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
    def __init__(self, items, bidders: List[Bidder], cfg):
        self.config = cfg
        self.items = items
        self.bidders = bidders
        self.bundle_generator = BundleGenerator(items)
        self.R = {}
        for bidder in self.bidders:
            self.R[bidder.name] = self._generate_initial_data(bidder, self.config['q-init'])

    def _generate_initial_data(self, b, q: int):
        bundles = self.bundle_generator(q)
        return bundles, b(bundles)

    def __getitem__(self, key):
        if isinstance(key, str) and key in self.R:
            return self.R[key]
        raise ValueError(f"Key {key} is not in range!")

    def add_queries(self, list_queries):
        for name, qs in enumerate(list_queries):
            bidder = self.bidders[name]
            X, y = self.R[name]
            newX, newy = qs, bidder(qs)
            self.R[bidder] = torch.vstack((X, newX)), torch.vstack((y, newy))

    def get_query_shape(self):
        return {n: b_i[0].shape for n, b_i in self.R.items()}        

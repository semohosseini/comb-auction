from typing import List
import torch
import ast

import logging

from .bidder import Bidder
from .query import NextQueryGenerator

from ...optim import BruteForceOptimizer

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
    def __init__(self, items, bidders: List[Bidder], cfg, query_address=None):
        self.config = cfg
        self.items = items
        self.bidders = bidders
        self.bundle_generator = BundleGenerator(items)
        self.R = {}
        #for bidder in self.bidders:
            #self.R[bidder.name] = self._generate_initial_data(bidder, self.config['q-init'])
        if 'init' in self.config and self.config['init']:
            q = self.config['q-init']
            bundles = self.bundle_generator(q)
            for bidder in self.bidders:
                self.R[bidder.name] = bundles, bidder(bundles)
        else:
            with open(query_address) as f:
                q = f.read() 
            queries = ast.literal_eval(q)
            for i in range(len(queries)):
                bidder = queries[i]
                for query in bidder:
                    value = torch.tensor(query[-1], device='cuda').float().to("cuda")
                    bundle = torch.tensor(query[:-1], device='cuda').float().to("cuda")
                    dictionary = {str(i): (bundle, value)}
                    self.add_data(dictionary)

            

        self.opt_sw = None
        if 'brute-force' in self.config and self.config['brute-force']:
            self.get_optimal()
            print('Brute Force Search: done.')
            print(f'Optimal Social Welfare: {self.opt_sw}')

    def get_optimal(self):
        m = len(self.items)
        n = len(self.bidders)
        bs = [b.name for b in self.bidders]
        ws = [b.vf for b in self.bidders]

        if not self.opt_sw:
            optim_aux = BruteForceOptimizer(m, n, ws)
            opt_alloc = optim_aux.optimize()
            self.opt_sw = optim_aux._social_welfare(opt_alloc)
        return self.opt_sw
    
    def get_R(self):
        return self.R

    def _generate_initial_data(self, b: Bidder, q: int):
        bundles = self.bundle_generator(q)
        return bundles, b(bundles)

    def __getitem__(self, key):
        if isinstance(key, str) and key in self.R:
            return self.R[key]
        raise ValueError(f"Key {key} is not in range!")
    
    def add_data(self, data):
        for bidder in data:
            if bidder in self.R:
                X, y = self.R[bidder]
                newX, newy = data[bidder]
                self.R[bidder] = torch.vstack((X, newX)), torch.vstack((y, newy))
            else:
                newX, newy = data[bidder]
                self.R[bidder] = newX, newy

    def add_queries(self, list_queries):
        for name, qs in enumerate(list_queries):
            bidder = self.bidders[name]
            X, y = self.R[name]
            newX, newy = qs, bidder(qs)
            self.R[bidder] = torch.vstack((X, newX)), torch.vstack((y, newy))

    def get_query_shape(self):
        return {n: b_i[0].shape for n, b_i in self.R.items()}        

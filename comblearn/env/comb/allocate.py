import torch
import comblearn
import logging
from ...optim.allocation import BruteForceOptimizer

class AllocationHandler:
    def __init__(self, items, models, cfg):
        self.config = cfg
        self.models = models
        self.items = items

    def allocate(self, except_key=None):
        m = len(self.items)
        n = len(self.models)

        deleted = False
        if except_key in self.models:
            except_model = self.models[except_key]
            del self.models[except_key]
            n -= 1
            deleted = True

        bs = self.models.keys()
        ws = [self.models[b] for b in bs]
        
        opt_cls = eval(self.config['optimizer'])
        if self.config['scheme'] == 'RandGreedy':
            optimizer = opt_cls(m, n, ws)
            optimizer.optimize(delta=self.config['delta'], sample_rate=self.config['sample_rate'])
        elif self.config['scheme'] == 'GradientAscent':
            optimizer = opt_cls(m, n, ws, eps=self.config['eps'])
            optimizer.optimize(lr=self.config['learning-rate'], bs=self.config['batch-size'])
        else:
            raise ValueError(f"Scheme {self.config['scheme']} for allocation not supported.")
        allocation = {b: a for b,a in zip(bs, optimizer.generate_allocation())}
        social_welfare = self.social_welfare(allocation)

        if except_key and deleted:
            self.models[except_key] = except_model
        
        return allocation, social_welfare
    
    def social_welfare(self, allocation):
        return torch.sum(torch.tensor([self.models[bidder](allocation[bidder]) for bidder in allocation]))
    
    def calc_payments(self, allocation):
        m = len(self.items)
        n = len(self.models)

        payments = {}
        for bidder in allocation:
            _, social_welfare_i = self.allocate(except_key=bidder)
            
            alloc_copy = dict(allocation)
            del alloc_copy[bidder]
            
            social_welfare_opt_i = self.social_welfare(alloc_copy)
            payments[bidder] = (social_welfare_i - social_welfare_opt_i).relu_()

        return payments
        
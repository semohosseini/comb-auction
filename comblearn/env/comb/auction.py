from .data import DataHandler
from .query import NextQueryGenerator

from ...optim import RandGreedyOptimizer

import numpy as np

class CombinatorialAuction():
    def __init__(self, bidders, items, vf_cls, q_init, q_max) -> None:
        self.q_init = q_init
        self.q_max = q_max
        self.bidders = bidders
        self.items = items
        self.data_handler = DataHandler(items, len(bidders), vf_cls, q_init)
        self.next_queries = NextQueryGenerator(items, self.data_handler)

    def run(self):
        # Generating Data
        T = (self.q_max - self.q_init) // len(self.bidders)
        t = 1

        while t <= T:
            main_queries = self.next_queries(self.bidders, self.data_handler.R)
            marginal_queries = []
            for i in range(len(self.bidders)):
                bidders_i = self.bidders[:i] + self.bidders[i+1:]
                R_i = self.data_handler.R[:i] + self.data_handler.R[i+1:]
                marginal_queries.append(self.next_queries(bidders_i, R_i))
            self.data_handler.add_queries([main_queries] + marginal_queries)
            t += 1

        # # Training value functions on data
        # trainer = Trainer(self.data_handler)
        # vhats = trainer.train()

        # # Running the auction
        # optimizer = RandGreedyOptimizer(self.items, self.bidders, vhats)
        # allocations = optimizer.optimize()

        # TODO: payment calculation
        # payments = np.zeros_like(allocations)
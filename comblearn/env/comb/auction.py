from .data import DataHandler
from .query import NextQueryGenerator

from ...optim import RandGreedyOptimizer, DSFLearner

import logging

import numpy as np

class CombinatorialAuction():
    def __init__(self, bidders, items, vfs, q_init, q_max) -> None:
        self.q_init = q_init
        self.q_max = q_max
        self.bidders = bidders
        self.items = items
        self.data_handler = DataHandler(items, len(bidders), vfs, q_init)

    def social_welfare(self, value_functions, allocation):
        return np.sum([vf(alloc) for vf, alloc in zip(value_functions, allocation)])

    def run(self, epochs=1000, lr=0.001):
        # Parameters
        T = (self.q_max - self.q_init) // len(self.bidders)
        t = 1
        m = len(self.items)
        n = len(self.bidders)

        # Generating next queries
        while t <= T:
            next_queries = NextQueryGenerator(m, n)
            new_queries = next_queries(self.data_handler.value_functions, self.data_handler.R, epochs=epochs, lr=lr)

            next_queries = NextQueryGenerator(m, n-1)
            for i in range(n):
                value_functions_i = self.data_handler.value_functions[:i] + self.data_handler.value_functions[i+1:]
                R_i = self.data_handler.R[:i] + self.data_handler.R[i+1:]
                marginal_query = next_queries(value_functions_i, R_i, epochs=epochs, lr=lr)
                for j in range(n):
                    if j < i:
                        new_queries[j] = np.vstack([new_queries[j], marginal_query[j]])
                    elif j == i:
                        continue
                    else:
                        new_queries[j] = np.vstack([new_queries[j], marginal_query[j-1]])
                    
            self.data_handler.add_queries(new_queries)
            t += 1

        # Final allocation
        for vf, data in zip(self.data_handler.value_functions, self.data_handler.R):
            X, y = data
            learner = DSFLearner(vf, lr, X, y)
            loss = learner(epochs)
            i += 1
            logging.info(f"Bidder {i}, loss: {loss}")

        optimizer = RandGreedyOptimizer(m, n, self.data_handler.value_functions)
        optimizer.optimize()
        allocation = optimizer.generate_allocation()
        social_welfare_opt = self.social_welfare(self.data_handler.value_functions, allocation)

        # Payment calculation
        payments = np.zeros((n, 1))
        for i in range(n):
            value_functions_i = self.data_handler.value_functions[:i] + self.data_handler.value_functions[i+1:]
            optimizer = RandGreedyOptimizer(m, n-1, value_functions_i)
            optimizer.optimize()
            allocation_i = optimizer.generate_allocation()
            social_welfare_i = self.social_welfare(value_functions_i, allocation_i)
            payments[i, 0] = social_welfare_i - social_welfare_opt

        return allocation, payments

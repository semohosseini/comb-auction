from ...optim import DSFLearner, RandGreedyOptimizer
import logging

class NextQueryGenerator:
    def __init__(self, cfg, dh, lh, ah):
        self.config = cfg
        self.data_handler = dh
        self.learning_handler = lh
        self.allocation_handler = ah

    def __call__(self, except_key=None):
        self.learning_handler.learn()

        alloc, _, optimizer = self.allocation_handler.allocate(except_key=except_key, return_optim=True)
        for bn in alloc:
            X, _ = self.data_handler[bn]
            j = 0
            while j < self.config['max-retries']:
                if any((alloc[bn] == X[:]).all(1)):
                    new_alloc = optimizer.generate_allocation()
                    if not any((new_alloc[bn] == X[:]).all(1)):
                        alloc[bn] = new_alloc[bn]
                        break
                else:
                    break
                j += 1
        return alloc
from ...optim import DSFLearner, RandGreedyOptimizer
import logging

class NextQueryGenerator:
    def __init__(self, m, n):
        self.m = m
        self.n = n

    def __call__(self, value_functions, datasets, lr=0.001, epochs=1000, delta=0.001, sample_rate=10, max_retries=10):
        i = 1
        for vf, data in zip(value_functions, datasets):
            X, y = data
            learner = DSFLearner(vf, lr, X, y)
            loss = learner(epochs)
            logging.info(f"Bidder {i}, loss: {loss}")
            i += 1

        optimizer = RandGreedyOptimizer(self.m, self.n, value_functions)
        optimizer.optimize(delta, sample_rate)
        alloc = optimizer.generate_allocation()
        i = 0
        # HINT: This part is not the same as MLCA. 
        # It may be faster but sacrifices the optimality in some situations.
        for query, data in zip(alloc, datasets):
            X, y = data
            j = 0
            while j < max_retries:
                if any((query == X[:]).all(1)):
                    new_alloc = optimizer.generate_allocation()
                    if not any((new_alloc[i] == X[:]).all(1)):
                        alloc[i] = new_alloc[i]
                        break
                else:
                    break
                j += 1
            i += 1
        return alloc
from .data import DataHandler
from .query import NextQueryGenerator

from ...optim import RandGreedyOptimizer, DSFLearner
from ...data import DSFValueFunction

import torch

import logging

class CombinatorialAuction():
    def __init__(self, bidders, items, true_vfs, learn_vfs, q_init, q_max, device='cuda' if torch.cuda.is_available() else 'cpu', custom_optim=None) -> None:
        self.q_init = q_init
        self.q_max = q_max
        self.bidders = bidders
        self.items = items
        self.data_handler = DataHandler(items, len(bidders), true_vfs, q_init)
        self.value_functions = learn_vfs
        self.device = device
        self.custom_optim = custom_optim

    def social_welfare(self, value_functions, allocation):
        return torch.sum(torch.tensor([vf(alloc) for vf, alloc in zip(value_functions, allocation)]).to(self.device))

    def run(self, epochs=1000, lr=0.001, delta=0.001, sample_rate=10, writer=None):
        # Parameters
        T = (self.q_max - self.q_init) // len(self.bidders)
        t = 1
        m = len(self.items)
        n = len(self.bidders)

        # Generating next queries
        logging.info("Query generation...")
        while t <= T:
            logging.info(f"Step: {t}/{T}, Query shapes: {self.data_handler.get_query_shape()}")
            logging.info("Generating main query...")
            next_queries = NextQueryGenerator(m, n, self.custom_optim)
            new_queries = next_queries(self.value_functions, self.data_handler, epochs=epochs, lr=lr, delta=delta, sample_rate=sample_rate, writer=writer, t=t)
            if writer:
                writer.add_scalar("Social Welfare", self.social_welfare(self.value_functions, new_queries), t)
            logging.info("Main query generated.")

            logging.info("Generating marginal queries...")
            next_queries = NextQueryGenerator(m, n-1, self.custom_optim)
            for i in range(n):
                value_functions_i = self.value_functions[:i] + self.value_functions[i+1:]
                R_i = self.data_handler[:i] + self.data_handler[i+1:]
                marginal_query = next_queries(value_functions_i, R_i, epochs=epochs, lr=lr, delta=delta, sample_rate=sample_rate)
                for j in range(n):
                    if j < i:
                        new_queries[j] = torch.vstack((new_queries[j], marginal_query[j]))
                    elif j == i:
                        continue
                    else:
                        new_queries[j] = torch.vstack((new_queries[j], marginal_query[j-1]))
                logging.info(f"Marginal query {i+1} generated")
                    
            self.data_handler.add_queries(new_queries)
            t += 1

        # Final allocation
        i = 0
        logging.info("Final allocation calculation...")
        for vf, data in zip(self.value_functions, self.data_handler):
            X, y = data
            learner = DSFLearner(vf, lr, X, y, self.custom_optim)
            loss = learner(epochs)
            i += 1
            logging.info(f"Bidder {i}, loss: {loss}")
            if writer:
                writer.add_scalar(f"Bidder {i} loss", loss, T+1)

        optimizer = RandGreedyOptimizer(m, n, self.value_functions)
        optimizer.optimize(delta, sample_rate)
        allocation = optimizer.generate_allocation()
        social_welfare_opt = self.social_welfare(self.value_functions, allocation)
        if writer:
            writer.add_scalar("Social Welfare", social_welfare_opt, T+1)
        logging.info(f"Optimal allocation: {allocation}")
        logging.info(f"Social welfare: {social_welfare_opt}")

        # Payment calculation
        logging.info("Payment calculation..")
        payments = torch.zeros((n, 1)).to(self.device)
        for i in range(n):
            value_functions_i = self.value_functions[:i] + self.value_functions[i+1:]
            optimizer = RandGreedyOptimizer(m, n-1, value_functions_i)
            optimizer.optimize(delta, sample_rate)
            allocation_i = optimizer.generate_allocation()
            social_welfare_i = self.social_welfare(value_functions_i, allocation_i)
            allocation_o_i = allocation[:i] + allocation[i+1:]
            social_welfare_opt_i = self.social_welfare(value_functions_i, allocation_o_i)
            payments[i, 0] = social_welfare_i - social_welfare_opt_i
        payments.relu_()
        
        logging.info(f"Payments: {payments.squeeze()}")

        return allocation, payments.squeeze()


class CombinatorialAuctionWithData():
    def __init__(self, bidders, items, dataset, learn_vfs, device='cuda' if torch.cuda.is_available() else 'cpu') -> None:
        self.bidders = bidders
        self.items = items
        self.dataset = dataset
        self.value_functions = learn_vfs
        self.device = device

    def social_welfare(self, value_functions, allocation):
        return torch.sum(torch.tensor([vf(alloc) for vf, alloc in zip(value_functions, allocation)]).to(self.device))

    def run(self, epochs=1000, lr=0.001, delta=0.001, sample_rate=10):
        # Parameters
        m = len(self.items)
        n = len(self.bidders)

        # Final allocation
        i = 0
        logging.info("Allocation calculation...")
        for vf, data in zip(self.value_functions, self.dataset):
            X, y = data
            learner = DSFLearner(vf, lr, X, y)
            loss = learner(epochs)
            i += 1
            logging.info(f"Bidder {i}, loss: {loss}")

        optimizer = RandGreedyOptimizer(m, n, self.value_functions)
        optimizer.optimize(delta, sample_rate)
        allocation = optimizer.generate_allocation()
        social_welfare_opt = self.social_welfare(self.value_functions, allocation)
        logging.info(f"Optimal allocation: {allocation}")
        logging.info(f"Social welfare: {social_welfare_opt}")

        # Payment calculation
        logging.info("Payment calculation..")
        payments = torch.zeros((n, 1)).to(self.device)
        for i in range(n):
            value_functions_i = self.value_functions[:i] + self.value_functions[i+1:]
            optimizer = RandGreedyOptimizer(m, n-1, value_functions_i)
            optimizer.optimize(delta, sample_rate)
            allocation_i = optimizer.generate_allocation()
            social_welfare_i = self.social_welfare(value_functions_i, allocation_i)
            allocation_o_i = allocation[:i] + allocation[i+1:]
            social_welfare_opt_i = self.social_welfare(value_functions_i, allocation_o_i)
            payments[i, 0] = social_welfare_i - social_welfare_opt_i
        payments.relu_()
        
        logging.info(f"Payments: {payments.squeeze()}")

        return allocation, payments.squeeze()

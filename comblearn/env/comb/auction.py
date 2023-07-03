from .allocate import AllocationHandler
from .learn import LearningHandler
from .bidder import Bidder
from .data import DataHandler
from .query import NextQueryGenerator

import torch
import comblearn

import logging

class CombinatorialAuction():
    def __init__(self, cfg):
        self.config = cfg
        self.data_config = cfg['data']
        self.learning_config = cfg['learning']
        self.allocation_config = cfg['allocation']
        self.query_config = cfg['query']
        self.device = cfg['device']
        self.items = cfg['items']
        self.bidders = []
        for bcfg in cfg['bidders']:
            vf_cls = eval(bcfg['cls'])
            vf = vf_cls(self.items, *bcfg['args']).to(self.device)
            self.bidders.append(Bidder(bcfg['name'], vf))

        self.data_handler = DataHandler(self.items, self.bidders, self.data_config)
        
        models = {}
        for mcfg in cfg['learning']['models']:
            vf_cls = eval(mcfg['cls'])
            vf = vf_cls(self.items, *mcfg['args']).to(self.device)
            models[mcfg['name']] = vf
        self.learning_handler = LearningHandler(models, self.data_handler, self.learning_config)

        self.allocation_handler = AllocationHandler(self.items, models, self.allocation_config)
        self.next_queries = NextQueryGenerator(self.query_config, self.data_handler, self.learning_handler, self.allocation_handler)

    def run(self, writer=None):
        # Parameters
        m = len(self.items)
        n = len(self.bidders)

        T = 0
        
        if self.query_config['marginal']:
            T = (self.config['q-max'] - self.config['q-init']) // len(self.bidders)
            t = 1
            # Generating next queries
            logging.info("Query generation...")
            while t <= T:
                logging.info(f"Step: {t}/{T}, Query shapes: {self.data_handler.get_query_shape()}")
                logging.info("Generating main query...")
                new_queries = self.next_queries()
                if writer:
                    writer.add_scalar("Social Welfare", self.allocation_handler.social_welfare(new_queries), t)
                logging.info("Main query generated.")

                logging.info("Generating marginal queries...")
                for bidder in self.bidders:
                    marginal_query = self.next_queries(except_key=bidder.name)
                    for bn in marginal_query:
                        new_queries[bn] = torch.vstack((new_queries[bn], marginal_query[bn]))
                    logging.info(f"Marginal query {bidder.name} generated")
                        
                self.data_handler.add_queries(new_queries)
                t += 1

        # Final allocation
        logging.info("Final allocation calculation...")
        self.learning_handler.learn(writer=writer, step=T+1)
        allocation, social_welfare = self.allocation_handler.allocate()
        if writer:
            writer.add_scalar("Social Welfare", social_welfare, T+1)
        logging.info(f"Optimal allocation:")
        for k in allocation:
            logging.info(f"({k, allocation[k]})")
        logging.info(f"Social welfare: {social_welfare}")

        # Payment calculation
        logging.info("Payment calculation..")
        payments = self.allocation_handler.calc_payments(allocation)
        
        logging.info(f"Payments: {payments}")

        return allocation, payments


from dataset import DataHandler, ConstantValueFunction
from query import NextQueryGenerator
from optimizer import RandGreedyOptimizer
from train import Trainer


# Configuration
N = 10  # Number of bidders
bidders = list(range(N))
q_init = 100  # Number of initial queries
q_max = 250  # Maximum number of queries per bidder
items = list(range(50))

# Generating Data
data_handler = DataHandler(items, N, ConstantValueFunction, q_init)
next_queries = NextQueryGenerator(items)

T = (q_max - q_init) // N
t = 1

while t <= T:
    main_queries = next_queries(bidders, data_handler.R)
    marginal_queries = []
    for i in range(N):
        bidders_i = bidders[:i] + bidders[i+1:]
        R_i = data_handler.R[:i] + data_handler.R[i+1:]
        marginal_queries.append(next_queries(bidders_i, R_i))
    data_handler.add_queries([main_queries] + marginal_queries)
    t += 1

# Training value functions on data
trainer = Trainer(data_handler)
vhats = trainer.train()

# Running the auction
optimizer = RandGreedyOptimizer(items, bidders, vhats)
output = optimizer.optimize()
from comblearn.env import CombinatorialAuction
from comblearn.data import DSFValueFunction

import logging

def test_comb():
    # Configuration
    N = 5  # Number of bidders
    bidders = list(range(N))
    q_init = 500  # Number of initial queries
    q_max = 510  # Maximum number of queries per bidder
    items = list(range(8))

    value_functions = [DSFValueFunction(items, 100, [2, 3, 2], 500), 
                       DSFValueFunction(items, 100, [2, 3, 2], 500), 
                       DSFValueFunction(items, 100, [2, 3, 2], 500), 
                       DSFValueFunction(items, 100, [2, 3, 2], 500), 
                       DSFValueFunction(items, 100, [2, 3, 2], 500)]

    value_functions_l = [DSFValueFunction(items, 110, [2, 4], 300), 
                         DSFValueFunction(items, 110, [2, 4], 300), 
                         DSFValueFunction(items, 110, [2, 4], 300), 
                         DSFValueFunction(items, 110, [2, 4], 300), 
                         DSFValueFunction(items, 110, [2, 4], 300)]

    auction = CombinatorialAuction(bidders, items, value_functions, value_functions_l, q_init, q_max)
    allocations, payments = auction.run(epochs=1000, lr=0.001, delta=0.005, sample_rate=5)
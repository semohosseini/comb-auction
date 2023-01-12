from env.comb import CombinatorialAuction
from data import ConstantValueFunction

# Configuration
N = 10  # Number of bidders
bidders = list(range(N))
q_init = 100  # Number of initial queries
q_max = 250  # Maximum number of queries per bidder
items = list(range(50))

auction = CombinatorialAuction(bidders, items, ConstantValueFunction, q_init, q_max)
allocations, payments = auction.run()
print(allocations, payments)
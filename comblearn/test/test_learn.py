from comblearn.env import DataHandler
from comblearn.data import DSFValueFunction
from comblearn.optim import DSFLearner

import logging

def test_learn_dsf():
    m = 5
    n = 3
    items  = list(range(5))
    lr = 0.001
    epochs = 1000
    vfs = [DSFValueFunction(items, 100, [2, 3, 2], 500), 
           DSFValueFunction(items, 100, [2, 3, 2], 500), 
           DSFValueFunction(items, 100, [2, 3, 2], 500)]
    dh = DataHandler(items, n, vfs, 500)

    vfs_l = [DSFValueFunction(items, 110, [2, 4], 300), 
             DSFValueFunction(items, 110, [2, 4], 300), 
             DSFValueFunction(items, 110, [2, 4], 300)]
    logging.info(f"Query shape: {dh.get_query_shape()}")
    i = 1
    for vf, data in zip(vfs_l, dh):
        X, y = data
        learner = DSFLearner(vf, lr, X, y)
        loss = learner(epochs)
        logging.info(f"Bidder {i}, loss: {loss}")
        i += 1
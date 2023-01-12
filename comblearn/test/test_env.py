from comblearn.env import BundleGenerator, DataHandler
from comblearn.data import DSFValueFunction, SumValueFunction

import logging

def test_bundle_generator():
    bgen = BundleGenerator(list(range(10)))
    assert bgen(7).shape == (7, 10)


def test_data_handler():
    m = 10
    vfs = [DSFValueFunction(list(range(m)), 100, [3, 2], 500), SumValueFunction(list(range(m)))]
    dh = DataHandler(list(range(m)), 2, vfs, 100)
    logging.info(dh.R[0])
    assert len(dh.R[0]) == 100
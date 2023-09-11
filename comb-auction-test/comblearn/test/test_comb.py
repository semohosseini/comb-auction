from comblearn.env import CombinatorialAuction
from comblearn.data import DSFValueFunction
import comblearn

import yaml
import logging

def test_comb_rg():
    # Configuration
    with open("config/config_rg.yaml") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    print("Configuration:", cfg)

    auction = CombinatorialAuction(cfg['auction'])
    allocations, payments = auction.run()

def test_comb_ga():
    # Configuration
    with open("config/config_ga.yaml") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    print("Configuration:", cfg)

    auction = CombinatorialAuction(cfg['auction'])
    allocations, payments = auction.run()
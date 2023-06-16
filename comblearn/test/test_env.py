from comblearn.env import BundleGenerator, DataHandler, Bidder
from comblearn.data import DSFValueFunction, SumValueFunction
import comblearn

import logging
import yaml

def test_bundle_generator():
    bgen = BundleGenerator(list(range(10)))
    assert bgen(7).shape == (7, 10)


def test_data_handler():
    m = 10
    with open("config/config_rg.yaml") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['auction']

    bidders = []
    for bcfg in cfg['bidders']:
        vf_cls = eval(bcfg['cls'])
        vf = vf_cls(cfg['items'], bcfg['max-out'], bcfg['hidden-sizes'], bcfg['alpha'])
        bidders.append(Bidder(bcfg['name'], vf))

    dh = DataHandler(cfg['items'], bidders, cfg['data'])
    assert len(dh.R['ali'][0]) == 500
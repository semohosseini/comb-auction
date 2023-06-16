from comblearn.env import DataHandler, Bidder, LearningHandler, DataHandler
from comblearn.data import DSFValueFunction
from comblearn.optim import DSFLearner

import logging
import comblearn
import yaml

def test_learn_dsf():
    with open("config/config_rg.yaml") as fp:   
        cfg = yaml.load(fp, Loader=yaml.FullLoader)['auction']

    data_config = cfg['data']
    learning_config = cfg['learning']
    device = cfg['device']
    items = cfg['items']
    bidders = []
    for bcfg in cfg['bidders']:
        vf_cls = eval(bcfg['cls'])
        vf = vf_cls(items, bcfg['max-out'], bcfg['hidden-sizes'], bcfg['alpha']).to(device)
        bidders.append(Bidder(bcfg['name'], vf))

    data_handler = DataHandler(items, bidders, data_config)
    
    models = {}
    for mcfg in cfg['learning']['models']:
        vf_cls = eval(bcfg['cls'])
        vf = vf_cls(items, mcfg['max-out'], mcfg['hidden-sizes'], mcfg['alpha']).to(device)
        models[mcfg['name']] = vf
    learning_handler = LearningHandler(models, data_handler, learning_config)

    learning_handler.learn()
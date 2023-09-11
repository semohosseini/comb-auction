import logging
from ...optim import DSFLearner


class LearningHandler:
    def __init__(self, models, dh, cfg):
        self.config = cfg
        self.models = models
        self.data_handler = dh

    def learn(self, writer=None, step=-1):
        for bidder in self.models.keys():
            X, y = self.data_handler[bidder]
            vf = self.models[bidder]
            loss = DSFLearner(vf, self.config, X, y)()
            logging.info(f"Bidder {bidder}, loss: {loss}")
            if writer and step >= 0:
                writer.add_scalar(f"Bidder {bidder} loss", loss, step)

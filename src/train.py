class Trainer:
    def __init__(self, dh):
        self.data_handler = dh

    def train(self): # Returns a list of callables.
        raise NotImplementedError("Train function should be implemented.")
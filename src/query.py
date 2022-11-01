class NextQueryGenerator:
    def __init__(self, items):
        self.items = items

    def __call__(self, bidders, data):
        raise NotImplementedError("Next Query should be implemented.")
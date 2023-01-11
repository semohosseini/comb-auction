class NextQueryGenerator:
    def __init__(self, items, dh):
        self.items = items
        self.data_handler = dh

    def __call__(self, bidders, data):
        raise NotImplementedError("Next Query should be implemented.")
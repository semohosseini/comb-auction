class ValueFunction:
    def __init__(self, items):
        self.items = items

    def __call__(self, bundle : set[int]):
        raise NotImplementedError("This is abstract value function! :(")


class ConstantValueFunction(ValueFunction):
    def __init__(self, items):
        super().__init__(items)

    def __call__(self, bundle: set[int]):
        return 1

from ...data.value import ValueFunction

class Bidder:
    def __init__(self, name, vf: ValueFunction):
        self.name = name
        self.vf = vf

    def __call__(self, bundle):
        return self.vf(bundle)
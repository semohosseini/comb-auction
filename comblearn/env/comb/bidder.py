from ...data.value import ValueFunction

class Bidder:
    def __init__(self, name, vf: ValueFunction = None):
        self.name = name
        self.vf = vf

    def __call__(self, bundle):
        if not self.vf:
            raise ValueError("There is no value function.")
        return self.vf(bundle)
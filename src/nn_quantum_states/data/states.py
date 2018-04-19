import numpy as np

class State(object):
    def __init__(self, display_name, N):
        self.display_name = display_name
        self.N = N

    def sample(self):
        raise NotImplementedError()

class W(State):
    def __init__(self, N):
        super().__init__("W", N)

    def sample(self):
        output = np.zeros(self.N)
        i = np.random.randint(0, self.N)
        output[i] = 1
        return output
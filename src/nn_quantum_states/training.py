import numpy as np

from nn_quantum_states.networks.rbm import RBM


class Trainer(object):
    def __init__(self, rbm, samples):
        self.samples = samples
        self.rbm = rbm

    def train(self):
        pass


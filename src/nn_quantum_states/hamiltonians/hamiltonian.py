import numpy as np

class Hamiltonian(object):
    def __init__(self, display_name, N, bc_periodic=True):
        self.display_name = display_name
        self.N = N

    def find_connection(self, spins):
        raise NotImplementedError()

    def min_flips(self):
        raise NotImplementedError()

    def get_local_energy(self, rbm, spins):
        raise NotImplementedError()
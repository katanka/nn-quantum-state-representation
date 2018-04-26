import numpy as np

class Hamiltonian(object):
    def __init__(self, display_name, num_spins, bc_periodic=True):

        self.display_name = display_name

        self.num_spins = num_spins


    def get_matrix_elems(self, rbm, spins):
        raise NotImplementedError()
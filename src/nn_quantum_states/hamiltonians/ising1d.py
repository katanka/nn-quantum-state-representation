import numpy as np
from nn_quantum_states.hamiltonians.hamiltonian import Hamiltonian


class Ising1D(Hamiltonian):


    def __init__(self, display_name, num_spins, bc_periodic=True, h = 2):
        super().__init__(display_name, num_spins, bc_periodic=True)

        # transverse field strength parameter
        self.h = h

        # periodic boundary conditions
        self.bc_p = bc_periodic

    def get_matrix_elems(self, spins):
        mat_elems = np.zeros((self.num_spins+1, 1)) - self.h

        # the index for the N+1 nearest neighbour spin configurations
        # with hamming distance <= 1.
        non_zero_states = [[i] for i in range(-1, self.num_spins)]
        mat_elems[0] = 0
        # Nearest Neighbor Term s_i * s_(i+1)
        for i in range(self.num_spins - 1):
             mat_elems[0] -= spins[i]*spins[i+1]
        if self.bc_p:
            mat_elems[0] -= spins[0]*spins[self.num_spins - 1]
        return mat_elems, non_zero_states



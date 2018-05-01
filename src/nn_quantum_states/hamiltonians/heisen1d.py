import numpy as np
from nn_quantum_states.hamiltonians.hamiltonian import Hamiltonian


class Heisen1D(Hamiltonian):


    def __init__(self, display_name, num_spins, bc_periodic=True, J = 1):
        super().__init__(display_name, num_spins, bc_periodic=True)

        # transverse field strength parameter
        self.J = J

        # periodic boundary conditions
        self.bc_p = bc_periodic

    def get_matrix_elems(self, spins):

        # the index for the N+1 nearest neighbour spin configurations
        # with hamming distance <= 1.
        non_zero_states = [[-1]]

        mat_elems = [0]
        # Nearest Neighbor Term s_i * s_(i+1)
        for i in range(self.num_spins - 1):
             mat_elems[0] += spins[i]*spins[i+1]
        if self.bc_p:
            mat_elems[0] += spins[0]*spins[self.num_spins - 1]

        for i in range(self.num_spins - 1):
            if spins[i] != spins[i + 1]:
                mat_elems.append(-2)
                non_zero_states.append([i, i + 1])
        if self.bc_p:
            if spins[0] != spins[self.num_spins - 1]:
                mat_elems.append(-2)
                non_zero_states.append([i+1, 0])
        mat_elems *= self.J
        return mat_elems, non_zero_states



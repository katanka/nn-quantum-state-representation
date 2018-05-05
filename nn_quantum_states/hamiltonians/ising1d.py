############################ COPYRIGHT NOTICE #####################################
# Code provided by G. Carleo and M. Troyer, written by G. Carleo, December 2016.  #
# Permission is granted for anyone to copy, use, modify, or distribute the        #
# accompanying programs and documents for any purpose, provided this copyright    #
# notice is retained and prominently displayed, along with a complete citation of #
# the published version of the paper:                                             #
#  ______________________________________________________________________________ #
# | G. Carleo, and M. Troyer                                                     |#
# | Solving the quantum many-body problem with artificial neural-networks        |#
# |______________________________________________________________________________|#
# The programs and documents are distributed without any warranty, express or     #
# implied.                                                                        #
# These programs were written for research purposes only, and are meant to        #
# demonstrate and reproduce the main results obtained in the paper.               #
# All use of these programs is entirely at the user's own risk.                   #
###################################################################################

import numpy as np
from nn_quantum_states.hamiltonians.hamiltonian import Hamiltonian
import itertools


class Ising1D(Hamiltonian):

    def __init__(self, num_spins, bc_periodic=True, h=2):
        super().__init__(num_spins, bc_periodic=True)

        # transverse field strength parameter
        self.h = h

        # periodic boundary conditions
        self.bc_p = bc_periodic

    def get_matrix_elems(self, spins):
        mat_elems = np.zeros((self.num_spins + 1, 1)) - self.h

        # the index for the N+1 nearest neighbour spin configurations
        # with hamming distance <= 1.
        non_zero_states = [[i] for i in range(-1, self.num_spins)]
        mat_elems[0] = 0
        # Nearest Neighbor Term s_i * s_(i+1)
        for i in range(self.num_spins - 1):
            mat_elems[0] -= spins[i] * spins[i + 1]
        if self.bc_p:
            mat_elems[0] -= spins[0] * spins[self.num_spins - 1]
        return mat_elems, non_zero_states

    def get_exact_ground_energy(self):
        basis = list(itertools.product([-1, 1], repeat=self.num_spins))

        H = np.zeros((2 ** self.num_spins, 2 ** self.num_spins))
        for H_i in range(2 ** self.num_spins):
            for H_j in range(2 ** self.num_spins):
                H_sum = 0
                for i in range(self.num_spins):
                    if H_i == H_j:
                        if i == self.num_spins - 1:
                            H_sum -= basis[H_j][i] * basis[H_j][0]
                        else:
                            H_sum -= basis[H_j][i] * basis[H_j][i + 1]

                    sj = list(basis[H_j])
                    sj[i] *= -1
                    if H_i == basis.index(tuple(sj)):
                        H_sum -= self.h

                H[H_i, H_j] = H_sum

        energy = np.min(np.real(np.linalg.eigvals(H))) / self.num_spins

        return energy

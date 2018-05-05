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

from nn_quantum_states.hamiltonians.hamiltonian import Hamiltonian
import numpy as np

class Heisen1D(Hamiltonian):

    def __init__(self, num_spins, bc_periodic=True, J=1):
        super().__init__(num_spins, bc_periodic=True)

        # transverse field strength parameter
        self.J_magnitude = np.abs(J)
        self.J_positive = J > 0

        # periodic boundary conditions
        self.bc_p = bc_periodic

    def get_matrix_elems(self, spins):

        # unpack column vectors
        spins = np.array(spins).ravel()


        # the index for the N+1 nearest neighbour spin configurations
        # with hamming distance <= 1.
        non_zero_states = [[-1]]

        mat_elems = [0]
        # Nearest Neighbor Term s_i * s_(i+1)
        for i in range(self.num_spins - 1):
            mat_elems[0] += spins[i] * spins[i + 1]
            if self.bc_p:
                mat_elems[0] += spins[0] * spins[self.num_spins - 1]

        for i in range(self.num_spins - 1):
            if spins[i] != spins[i + 1]:
                mat_elems.append(-2)
                non_zero_states.append([i, i + 1])
            if self.bc_p:
                if spins[0] != spins[self.num_spins - 1]:
                    mat_elems.append(-2)
                    non_zero_states.append([i + 1, 0])
        mat_elems *= self.J_magnitude
        return mat_elems, non_zero_states

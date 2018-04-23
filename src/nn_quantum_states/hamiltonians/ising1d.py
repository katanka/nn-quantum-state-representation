import numpy as np
from nn_quantum_states.hamiltonians.hamiltonian import Hamiltonian


class Ising1D(Hamiltonian):
    def __init__(self, display_name, N, bc_periodic=True, h = 1):
        super().__init__(display_name, N)
        self.h = h    #transverse field strength

    def get_local_energy(self, rbm, spins):
        El = np.array([0.0]).reshape((1, 1))
        N = len(spins)
        #Nearest Neighbor Term s_i * s_(i+1)
        for i in range(N):
            if i == N - 1:
                El -= spins[i]*spins[0]   #periodic boundary conditions
            else:
                El -= spins[i]*spins[i+1]
        #Tranverse Field Term sigma_x s_i = -s_i
        state_i = rbm.Psi_M(spins)
        for i in range(N):
            spins[i] = -spins[i]   #apply spin flip
            El -= np.real(self.h * spins[i] * (rbm.Psi_M(spins)/state_i))
            spins[i] = -spins[i]
        return El/N



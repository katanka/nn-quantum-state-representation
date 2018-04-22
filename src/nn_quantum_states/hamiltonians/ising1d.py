import numpy as np
from nn_quantum_states.hamiltonians.hamiltonian import Hamiltonian

class Ising1D(Hamiltonian):
    def __init__(self, display_name, N, bc_periodic=True):
        super().__init__(display_name, N)

    def get_local_energy(self, spins):
        raise NotImplementedError()
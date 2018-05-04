class Hamiltonian(object):

    def __init__(self, num_spins: int, bc_periodic=True):
        self.num_spins = num_spins
        self.bc_periodic = bc_periodic

    def get_matrix_elems(self, spins):
        raise NotImplementedError()

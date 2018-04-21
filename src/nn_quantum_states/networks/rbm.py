import numpy as np

from nn_quantum_states.networks.neuralnetwork import NeuralNetwork


def sigmoid(x):
    return np.pow((1 + np.exp(-x)), -1)


class RBM(NeuralNetwork):
    def __init__(self, num_vis, num_hid):
        super().__init__("Restricted Boltzmann Machine")

        self.num_vis = num_vis
        self.num_hid = num_hid

        self.vis_bias = self.random_complex((self.num_vis, 1))
        self.hid_bias = self.random_complex((self.num_hid, 1))
        self.weights = self.random_complex((self.num_vis, self.num_hid))

        self.effective_angles = np.zeros((self.num_hid, 1))

    # Computes the wave function amplitude corresponding to a
    # given n-dimensional state vector

    def random_complex(self, size):
        a = (np.random.random(size) - .5) * 10e-2
        b = (np.random.random(size) - .5) * 10e-2
        return a + b * 1j

    def set_effective_angles(self, spins):
        self.effective_angles = self.vis_bias + self.weights.T @ spins

    def Psi_M(self, spins):
        return np.exp(self.vis_bias @ spins) * np.prod(2 * np.cosh(self.effective_angles))

    def E_Local(self, spins):
        E_loc = 0
        return E_loc

    def flip_accepted(self, spins, flip_idx):
        spins_flipped = np.zeros(spins.shape)
        for i in flip_idx:
            spins_flipped[i] *= -1

        Psi_M_initial = self.Psi_M(spins)
        Psi_M_flipped = self.Psi_M(spins_flipped)

        return np.real(
            Psi_M_flipped * np.conj(Psi_M_flipped) / (Psi_M_initial * np.conj(Psi_M_initial))) >= np.random.normal()

    def step(self, spins, num_spins=1):
        flip_idx = np.random.choice(list(range(self.num_vis)), num_spins)

        if self.flip_accepted(spins, flip_idx):
            for i in flip_idx:
                spins[i] *= -1
            return spins, 0
        else:
            return spins, 1

    def find_state(self):
        spins = np.random.randint(2, size=self.num_vis)
        spins[spins == 0] = -1
        hidden = self.random_complex((self.num_hid, 1))
        for i in range(self.num_vis):
            gamma = sum([self.weights[i, j] * hidden[j] for j in range(self.num_hid)]) + self.vis_bias[i]
            if sigmoid(2 * gamma) > np.random.normal():
                spins[i] = 1
            else:
                spins[i] = -1

        for j in range(self.num_hid):
            theta = sum([self.weights[i, j] * spins[j] for j in range(self.num_vis)]) + self.hid_bias[j]
            if sigmoid(2 * theta) > np.random.normal():
                hidden[i] = 1
            else:
                hidden[i] = -1

        return spins, hidden

# flip random site
# site = np.random.choice(range(self.num_vis), 1)

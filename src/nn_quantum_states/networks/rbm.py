import numpy as np

from nn_quantum_states.networks.neuralnetwork import NeuralNetwork


def sigmoid(x):
    return np.pow((1 + np.exp(-x)), -1)


class RBM(NeuralNetwork):
    def __init__(self, num_vis, num_hid):
        super().__init__("Restricted Boltzmann Machine")

        self.num_vis = num_vis
        self.vis_bias = np.zeros((num_vis, 1))

        self.num_hid = num_hid
        self.hid_bias = np.zeros((num_hid, 1))

        self.eff_angles = np.zeros((num_hid, 1))

        self.weights = np.zeros((num_vis, num_hid))

    # Computes the wave function amplitude corresponding to a
    # given n-dimensional state vector

    def random_complex(self, size):
        a = (np.random.random(size) - .5) * 10e-2
        b = (np.random.random(size) - .5) * 10e-2
        return a + b * 1j

    def effective_angles(self, state):
        return self.vis_bias + self.weight_matrix.T @ state

    def Psi_M(self, state):
        return np.exp(self.vis_bias @ state) * np.prod(2 * np.cosh(self.effective_angles(state)))

    def initialize_weights(self):
        self.vis_bias = self.random_complex((self.num_vis, 1))
        self.hid_bias = self.random_complex((self.num_hid, 1))
        self.weights = self.random_complex((self.num_vis, self.num_hid))

    def E_Local(self, state):
        E_loc = 0
        return E_loc

    def sample(self):
        spins = np.zeros((self.num_vis, 1))
        hidden = np.zeros((self.num_hid, 1))
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

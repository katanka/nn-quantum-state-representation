import tensorflow as tf
import numpy as np

from nn_quantum_states.networks.neuralnetwork import NeuralNetwork

def sigmoid(x):
    return np.pow((1 + np.exp(-x)), -1)

class RBM(object):
    def __init__(self, num_vis, num_hid):

        self.num_vis = num_vis
        self.vis_bias = np.zeros((num_vis, 1))

        self.num_hid = num_hid
        self.hid_bias = np.zeros((num_hid, 1))

        self.eff_angles = np.zeros((num_hid, 1))

        self.weight_matrix = np.zeros((num_vis, num_hid))


    # Computes the wave function amplitude corresponding to a
    # given n-dimensional state vector
    def Psi_M(self, state):
        return np.exp(np.dot(self.vis_bias, state)) * np.prod(2*np.cosh(self.eff_angles))

    def initialize_weights(self):
        pass

    def E_Local(self, state):
        E_loc = 0
        return E_loc

    def Gibbs_step(self):
        #flip random site
        site = np.random.choice(range(self.num_vis), 1)



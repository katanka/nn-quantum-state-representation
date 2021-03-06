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
from nn_quantum_states.sampler import Sampler


def sigmoid(x):
    return np.power((1 + np.exp(-x)), -1)

class RBM(object):
    def __init__(self, num_vis, num_hid, hami, lr = 1):
        # number of visible units (spins in the system)
        self.num_vis = int(num_vis)

        # number of hidden units (hidden correlation parameters)
        self.num_hid = int(num_hid)

        # number of model parameters
        self.num_var = self.num_vis + self.num_hid + self.num_vis*self.num_hid

        # bias on visible units
        self.vis_bias = self.init_weights((self.num_vis, 1))

        # bias on hidden units
        self.hid_bias = self.init_weights((self.num_hid, 1))

        # weight matrix
        self.W = self.init_weights((self.num_vis, self.num_hid))

        # look-up tables for efficient computation of wave function amplitudes
        self.effective_angles = np.complex128(np.zeros((self.num_hid, 1)))

        # hamiltonian governing evolution of system
        self.hamiltonian = hami

        # learning rate
        self.lr = lr

        self.sampler = None

    # Initialize matrix of complex numbers within a range set below.
    def init_weights(self, size):
        if type(size) == tuple:
            size = list(map(int, size))
        else:
            size = int(size)
        a = (np.random.random(size) - .5) * 10e-2
        b = (np.random.random(size) - .5) * 10e-2
        return np.complex128(a + b * 1j)

    # Initialize effective angles with a given spin.
    def init_effective_angles(self, spins):
        self.effective_angles = self.hid_bias + self.W.T @ spins

    # update look-up tables when new spin configuration accepted by Metropolis Algorithm
    def update_eff_angles(self, spins, spins_prime):
        spins_diff = spins - spins_prime
        self.effective_angles -= self.W.T @ (spins_diff).reshape((self.num_vis, 1))

    # returns wave function amplitude of SPINS configuration
    def Psi_M(self, spins):
        return np.complex128(np.exp(np.dot(self.vis_bias.T, spins)) * np.prod(2 * np.cosh(self.effective_angles)))

    # ratio of wave function amplitudes for spin configurations spins_prime/spins
    def amp_ratio(self, spins, spins_prime):
        return np.exp(self.log_amp_ratio(spins, spins_prime))

    # logarithm of amplitude ratio (easier for numerical stability reasons)
    def log_amp_ratio(self, spins, spins_prime):
        spins_diff = spins - spins_prime
        piece1 = -1*np.dot(self.vis_bias.T, spins_diff)
        piece2 = np.sum(np.log(np.cosh(self.effective_angles - self.W.T @ spins_diff))
                         - np.log(np.cosh(self.effective_angles)))
        log = piece1 + piece2
        return log

    # run NUM_EPOCHS iterations of the SR algorithm, using MCMC sampling with NUM_SAMPLES iterations,
    # returns the energy per spin after each parameter update.
    def optimize(self, num_epochs, num_samples, therm_factor, sweep_factor, verbose=False):
        energies = []
        for iter_num in range(num_epochs):
            try:
                self.sampler = Sampler(self.hamiltonian, self)
                self.sampler.generate_samples(num_samples, therm_factor, sweep_factor)
                param_step, E_locs = self.get_SR_gradient(self.sampler, iter_num, num_samples)
                self.update_params(self.lr*param_step)
                energy = np.mean(np.real(E_locs)) / self.num_vis
                if verbose:
                    print("%d: %f" % (iter_num+1, energy))
                energies.append(energy)
            except RuntimeWarning:
                # overflow detected, no point in continuing
                break
        return np.array(energies)

    # returns the SR parameter update using the samples generated by SAMPLER
    #  as well as local energies for each spin configuration created
    def get_SR_gradient(self, sampler, iter_num, num_samples):
        spin_set = np.array(sampler.spin_set).reshape((num_samples, self.num_vis))
        E_locs = np.array(sampler.local_energies).reshape((len(spin_set), 1))
        param_step, grads = self.compute_grads(iter_num, spin_set, E_locs, num_samples)
        return param_step, E_locs

    # Computes variational parameters and returns the parameter update for each weight in
    # the RBM as well as the gradients calculated
    def compute_grads(self, iter_num, spin_set, E_locs, num_samples):
        eff_angles = self.W.T @ spin_set.T + self.hid_bias
        var_a = np.reshape(spin_set.T, (self.num_vis, num_samples))
        var_b = np.tanh(eff_angles)
        var_b = np.reshape(var_b, (self.num_hid, num_samples))
        var_W = (spin_set.T).reshape((self.num_vis, 1, num_samples)) * np.tanh(eff_angles.reshape((1, self.num_hid, num_samples)))

        grads = np.concatenate([var_a, var_b, var_W.reshape(self.num_vis*self.num_hid, num_samples)])

        exp_grads = np.real(np.sum(grads, axis=1, keepdims=True)) / num_samples

        # Cross-correlation matrix S
        exp_grads_matrix = np.conj(exp_grads.reshape((grads.shape[0], 1))) * exp_grads.reshape((1, grads.shape[0]))
        cross_term = (np.conjugate(grads) @ grads.T)/num_samples
        S = cross_term - exp_grads_matrix

        F = np.sum(E_locs.T * grads.conj(), axis=1)/num_samples
        F -= np.sum(E_locs.T, axis=1) * np.sum(grads.conj(), axis=1) /np.square(num_samples)
        lamb = 100*(.85**iter_num)
        if lamb < 10e-4:
            lamb = 10e-4
        S_ridge = S + lamb*np.eye(S.shape[0])
        lr = self.lr*(1/np.sqrt(iter_num + 1))
        param_step = (np.linalg.solve(S_ridge, -1*lr*F)).reshape(grads.shape[0], 1)

        return param_step, grads

    # updates each weight parameter based on the output of COMPUTE_GRADS
    def update_params(self, steps):
        self.vis_bias += steps[:self.num_vis].copy()
        self.hid_bias += steps[self.num_vis:(self.num_vis+self.num_hid)].copy()
        self.W += steps[(self.num_vis+self.num_hid):].reshape((self.num_vis, self.num_hid)).copy()

#   # Not used
    def gibbs_sample(self, spins, hidden):
        for i in range(self.num_vis):
            gamma = sum([self.W[i, j] * hidden[j] for j in range(self.num_hid)]) + self.vis_bias[i]
            if sigmoid(2 * gamma) > np.random.random():
                spins[i] = 1
            else:
                spins[i] = -1
        for j in range(self.num_hid):
            theta = self.effective_angles[j]
            if sigmoid(2 * theta) > np.random.random():
                hidden[i] = 1
            else:
                hidden[i] = -1
        self.update_eff_angles(spins)
        return spins, hidden

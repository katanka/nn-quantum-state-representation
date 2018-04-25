import numpy as np
from nn_quantum_states.networks.neuralnetwork import NeuralNetwork


def sigmoid(x):
    return np.power((1 + np.exp(-x)), -1)


class RBM(NeuralNetwork):
    def __init__(self, num_vis, num_hid, hami):
        super().__init__("Restricted Boltzmann Machine")

        # number of visible units (spins in the system)
        self.num_vis = num_vis

        # number of hidden units (hidden correlation parameters)
        self.num_hid = num_hid

        # number of model parameters
        self.num_var = self.num_vis + self.num_hid + self.num_vis*self.num_hid

        # bias on visible units
        self.vis_bias = self.init_weights((self.num_vis, 1))

        # bias on hidden units
        self.hid_bias = self.init_weights((self.num_hid, 1))

        # weight matrix
        self.W = self.init_weights((self.num_vis, self.num_hid))

        # flattened vector of variational parameters for optimization
        self.param_flat = np.vstack((self.vis_bias, self.hid_bias, np.ravel(self.W).reshape((self.num_hid*self.num_vis, 1))))

        # look-up tables for efficient computation of wave function amplitudes
        self.effective_angles = np.complex128(np.zeros((self.num_hid, 1)))

        # hamiltonian governing evolution of system
        self.hamiltonian = hami

    def init_weights(self, size):
        a = (np.random.random(size) - .5) * 10e-3
        b = (np.random.random(size) - .5) * 10e-3
        return np.complex128(a + b * 1j)

    def init_effective_angles(self, spins):
        self.effective_angles = self.hid_bias + self.W.T @ spins

    # update look-up tables when new spin configuration accepted by Metropolis Algorithm
    def update_eff_angles(self, spins):
        self.effective_angles -= 2*self.W.T @ spins.reshape((self.num_vis, 1))

    # returns wave function amplitude of SPINS configuration
    def Psi_M(self, spins):
        return np.complex128(np.exp(-np.dot(self.vis_bias.T, spins)) * np.prod(2 * np.cosh(self.effective_angles)))

    # call local energy from hamiltonian object
    def E_Local(self, spins, hamiltonian):
        return hamiltonian.get_local_energy(self, spins)

    # flip NUM_SPINS spins in the input configuration SPINS and accept the new
    # configuration if the flip is accepted.
    def step(self, spins, num_spins=1):
        flip_idx = np.random.choice(list(range(self.num_vis)), num_spins)

        if self.flip_accepted(spins, flip_idx):
            for i in flip_idx:
                spins[i] *= -1
            self.update_eff_angles(spins)
            return spins, 0
        else:
            return spins, 1

    # returns True if the flipped state is accepted and False otherwise
    def flip_accepted(self, spins, flip_idx):
        spins_flipped = np.ones(spins.shape)
        for i in flip_idx:
            spins_flipped[i] *= -1

        Psi_M_initial = self.Psi_M(spins)
        Psi_M_flipped = self.Psi_M(spins_flipped)
        transition_prob = np.square(np.absolute(Psi_M_flipped / (Psi_M_initial)))
        return transition_prob >= np.random.random()

    def init_random_state(self):
        spins = np.random.randint(2, size=self.vis_bias.shape)
        spins[spins == 0] = -1
        return spins

    def SR_step(self, iterations, therm_factor, reg):
        lr = .001 # learning rate
        N_var = self.num_var

        #thermalization
        spins = self.init_random_state()
        for k in range(int(iterations*therm_factor)):
            #spins, hidden = self.gibbs_sample(spins, hidden)
            spins, _ = self.step(spins)

        E_locs = []
        exp_var = np.zeros((N_var, 1), dtype=np.complex)
        exp_conj_var = np.zeros((N_var, 1), dtype=np.complex128)
        probs = []
        S = np.zeros((N_var, N_var), dtype=np.complex128)
        F = np.zeros((N_var, 1), dtype=np.complex128)
        
        for k in range(iterations):
            #spins, hidden = self.gibbs_sample(spins, hidden)
            spins, _ = self.step(spins)
            if k % 100 == 0:

                var_a = spins.reshape((len(spins), 1))
                var_b = np.tanh(self.effective_angles)
                var_W = var_a @ var_b.T
                var_flat = np.vstack((var_a, var_b, np.ravel(var_W).reshape((self.num_hid * self.num_vis, 1))))

                E_loc = self.E_Local(spins, self.hamiltonian)
                E_locs.append(E_loc)

                prob = np.absolute(self.Psi_M(spins)*self.Psi_M(spins).conj())
                probs.append(prob)

                exp_var += prob*var_flat
                exp_conj_var += prob*np.conj(var_flat)

                S += prob * (np.conj(var_flat) @ var_flat.T)
                F += prob*E_loc*np.conj(var_flat)

        sum_probs = np.sum(probs)
        exp_var = exp_var/sum_probs
        exp_conj_var = exp_conj_var/sum_probs
        S = S/sum_probs
        S -= np.conj(exp_var) @ exp_var.T
        exp_E_loc = np.sum(E_locs)/sum_probs
        F = F/sum_probs - exp_E_loc*exp_conj_var
        if reg <= .001:
            reg = .001
        steps = np.linalg.solve(S + reg*np.eye(N_var), -lr*F)
        #steps = np.matmul(np.linalg.inv(S + reg*np.eye(N_var)), -lr*F)
        self.vis_bias += steps[:self.num_vis].reshape((self.num_vis, 1))
        self.hid_bias += steps[self.num_vis:(self.num_vis + self.num_hid)]
        self.W += steps[(self.num_vis + self.num_hid):].reshape((self.num_vis, self.num_hid))
        print('Local Energy: ', exp_E_loc)
        return exp_E_loc



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

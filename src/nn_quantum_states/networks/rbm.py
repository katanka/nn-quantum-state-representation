import numpy as np

from nn_quantum_states.networks.neuralnetwork import NeuralNetwork


def sigmoid(x):
    return np.pow((1 + np.exp(-x)), -1)


class RBM(NeuralNetwork):
    def __init__(self, num_vis, num_hid, hami):
        super().__init__("Restricted Boltzmann Machine")

        self.num_vis = num_vis
        self.num_hid = num_hid
        self.num_var = self.num_vis + self.num_hid + self.num_vis*self.num_hid

        self.vis_bias = self.random_complex((self.num_vis, 1))
        self.hid_bias = self.random_complex((self.num_hid, 1))
        self.W = self.random_complex((self.num_vis, self.num_hid))
        self.param_flat = np.vstack((self.vis_bias, self.hid_bias, np.ravel(self.W).reshape((self.num_hid*self.num_vis, 1))))

        self.effective_angles = np.zeros((self.num_hid, 1))

        self.hamiltonian = hami

    # Computes the wave function amplitude corresponding to a
    # given n-dimensional state vector

    def random_complex(self, size):
        a = (np.random.random(size) - .5) * 10e-2
        b = (np.random.random(size) - .5) * 10e-2
        return a + b * 1j

    def set_effective_angles(self, spins):
        self.effective_angles = self.hid_bias + self.W.T @ spins

    def update_eff_angles(self, spins):
        self.effective_angles -= 2*self.W.T @ spins

    def Psi_M(self, spins):
        return np.exp(self.vis_bias @ spins) * np.prod(2 * np.cosh(self.effective_angles))

    def E_Local(self, spins, hamiltonian):
        return hamiltonian.get_local_energy(self, spins)

    def flip_accepted(self, spins, flip_idx):
        spins_flipped = np.zeros(spins.shape)
        for i in flip_idx:
            spins_flipped[i] *= -1

        Psi_M_initial = self.Psi_M(spins)
        Psi_M_flipped = self.Psi_M(spins_flipped)

        return np.absolute(
            Psi_M_flipped * np.conj(Psi_M_flipped) / (Psi_M_initial * np.conj(Psi_M_initial))) >= np.random.normal()

    def step(self, spins, num_spins=1):
        flip_idx = np.random.choice(list(range(self.num_vis)), num_spins)

        if self.flip_accepted(spins, flip_idx):
            for i in flip_idx:
                spins[i] *= -1
            self.update_eff_angles(spins)
            return spins, 0
        else:
            return spins, 1

    def SR_step(self, iterations, therm_factor):
        lr = .1 #learning rate
        spins = np.random.randint(2, size=self.num_vis)
        spins[spins == 0] = -1
        rejected = 0
        N_var = self.num_var
        #thermalization
        for k in range(iterations*therm_factor):
            spins, count = self.step(spins)
            rejected += count
        E_locs = []
        exp_var = np.zeros((N_var, 1))
        exp_conj_var = np.zeros((N_var, 1))
        probs = []
        F = []
        exp_cross = np.zeros((N_var, N_var))
        for k in range(iterations):
            spins, count = self.step(spins)
            rejected += count
            if k % 100 == 0:
                var_a = spins.reshape((len(spins), 1))
                var_b = np.tanh(self.effective_angles)
                var_W = var_a @ var_b.T
                var_flat = np.vstack(var_a, var_b, np.ravel(var_W).reshape((self.num_hid * self.num_vis, 1)))
                E_loc = self.E_Local(spins, self.hamiltonian)
                prob = np.absolute(self.Psi_M(spins)*self.Psi_M(spins).conj())
                E_locs.append(E_loc)
                probs.append(prob)
                exp_var += prob*var_flat
                exp_conj_var += prob*np.conj(var_flat)
                exp_cross += prob * (np.conj(var_flat) @ var_flat.T)
                F += prob*E_loc*np.conj(var_flat)
        sum_probs = np.sum(probs)
        exp_var = exp_var/sum_probs
        exp_conj_var = exp_conj_var/sum_probs
        exp_cross = exp_cross/sum_probs
        S = exp_cross - np.conj(exp_var) @ exp_var.T
        exp_E_loc = np.sum(E_locs)/sum_probs
        F = F/sum_probs - exp_E_loc*exp_conj_var
        steps = np.scipy.sparse.linalg.spsolve(S, -lr*F)
        self.vis_bias += steps[:len(self.num_vis)]
        self.hid_bias += steps[self.num_vis:(self.num_vis + self.num_hid)]
        self.W += steps[(self.num_vis + self.num_hid):].reshape((self.num_vis, self.num_hid))
        print('Local Energy: ' + exp_E_loc)
        return exp_E_loc



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

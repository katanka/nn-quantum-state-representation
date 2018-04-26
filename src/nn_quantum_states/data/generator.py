import numpy as np

class Generator(object):
    """
    Performs Markov Chain Monte Carlo Sampling to generate a set of
    spin configurations sampled from the wave function we are trying
    to model.
    """

    def __init__(self, hamiltonian, rbm, init_state = None):

        # hamiltonian of system to compute local energies
        self.hami = hamiltonian

        # neural network spin model
        self.model = rbm

        # number of spins
        self.num_spins = self.model.num_vis

        # list of spin configurations for computing energy gradients
        self.spin_set = []

        # list of local energies for each spin configuration in spin_set
        self.local_energies = []

        # number of rejected moves for each Metropolis sampling step
        self.num_rej = 0

        # current state during sampling process
        self.curr_state = init_state
        if self.curr_state is None:
            self.curr_state = self.init_random_state()


    def init_random_state(self):
        spins = np.random.randint(2, size=self.model.vis_bias.shape)
        spins[spins == 0] = -1
        return spins


    def get_local_energy(self):
        elems, state_idx = self.hami.get_matrix_elems(self.curr_state)
        energy_terms = []
        for i in range(len(state_idx)):
            id = state_idx[i]
            flipped_state = self.flip_spins(self.curr_state, id)
            energy_terms.append(self.model.amp_ratio(self.curr_state, flipped_state) * elems[i])
        return sum(energy_terms)

    # flip NUM_SPINS spins in the input configuration SPINS and accept the new
    # configuration if the flip is accepted.
    def step(self, num_spins=1):
        flip_idx = np.random.choice(list(range(self.model.num_vis)), num_spins)
        spins = self.curr_state

        if self.flip_accepted(spins, flip_idx):
            spins_prime = self.flip_spins(self.curr_state, flip_idx)
            self.model.update_eff_angles(spins, spins_prime)
            self.curr_state = spins_prime
            self.num_spins = self.num_spins
        else:
            self.num_rej += 1

    # returns True if the flipped state is accepted and False otherwise
    def flip_accepted(self, spins, flip_idx):
        flipped_spins = self.flip_spins(spins, flip_idx)
        transition_prob = np.square(np.absolute(self.model.amp_ratio(spins, flipped_spins)))
        return transition_prob >= np.random.random()

    def flip_spins(self, spins, flip_idx):
        flipped_spins = spins.copy()
        for i in flip_idx:
            if i == -1:
                return spins
            else:
                flipped_spins[i][0] = spins[i][0]*-1
        return flipped_spins

    def generate_samples(self, iterations, therm_factor=0.1, sweep_factor=1):

        self.model.init_effective_angles(self.curr_state)

        # thermalization
        for k in range(int(iterations * therm_factor)*int(sweep_factor*self.num_spins)):
            # spins, hidden = self.gibbs_sample(spins, hidden)
            self.step()

        # Monte Carlo Sampling
        for i in range(int(iterations)):
            for k in range(int(sweep_factor*self.num_spins)):
                self.step()
            self.spin_set.append(np.array(self.curr_state).reshape((1, self.model.num_vis)))
            self.local_energies.append(self.get_local_energy())


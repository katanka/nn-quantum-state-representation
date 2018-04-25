from nn_quantum_states.networks.rbm import RBM
from nn_quantum_states.hamiltonians.ising1d import Ising1D
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, './')


num_spins = 4
alpha = 2
iter = 20000

hamiltonian = Ising1D("ising1D", num_spins, h=1)
model = RBM(num_spins, alpha*num_spins, hamiltonian)

spins = np.random.randint(2, size=model.num_vis)
spins[spins == 0] = -1

model.init_effective_angles(spins.reshape((num_spins, 1)))

E = []
for i in range(30):
    E.append(model.SR_step(iter, therm_factor=.01, reg=100*.9**i))

plt.figure()
plt.plot(range(30), E)
plt.xlabel('Iteration')
plt.ylabel('$E_{loc}$')
plt.title('Energy vs. Iteration')
plt.show()
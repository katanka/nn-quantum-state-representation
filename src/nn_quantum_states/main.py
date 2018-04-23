from nn_quantum_states.networks.rbm import RBM
from nn_quantum_states.hamiltonians.ising1d import Ising1D
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './')

num_spins = 5
alpha = 2

num_SR_steps = 50
hami = Ising1D('Ising Model', num_spins, h=2)
rbm = RBM(num_spins, alpha * num_spins, hami)
Energies = []
for _ in range(50):
    E_loc = rbm.SR_step(iterations=10000, therm_factor=.1)
    Energies.append(E_loc)

plt.figure()
plt.plot(range(50), Energies)
plt.xlabel('Iteration')
plt.ylabel('$E_{loc}$')
plt.title('Energy vs. Iteration')
plt.show()
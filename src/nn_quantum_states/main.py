from nn_quantum_states.networks.rbm import RBM
from nn_quantum_states.hamiltonians.ising1d import Ising1D
import matplotlib.pyplot as plt
<<<<<<< HEAD
from nn_quantum_states.data.generator import Generator
=======
import numpy as np
>>>>>>> 35da1c9ebdc13cc4e89d42c9adab6f41b1e8ad97

import sys
sys.path.insert(0, './')

<<<<<<< HEAD
=======

>>>>>>> 35da1c9ebdc13cc4e89d42c9adab6f41b1e8ad97
num_spins = 4
alpha = 2
iter = 20000

hamiltonian = Ising1D("ising1D", num_spins, h=1)
model = RBM(num_spins, alpha*num_spins, hamiltonian)

spins = np.random.randint(2, size=model.num_vis)
spins[spins == 0] = -1

model.init_effective_angles(spins.reshape((num_spins, 1)))

<<<<<<< HEAD
iterations = 20000
epochs = 50
hami = Ising1D('Ising Model', num_spins, h=2)
model = RBM(num_spins, alpha * num_spins, hami)
Energies = []
model.optimize(epochs, iterations)

# plt.figure()
# plt.plot(range(50), Energies)
# plt.xlabel('Iteration')
# plt.ylabel('$E_{loc}$')
# plt.title('Energy vs. Iteration')
# plt.show()
=======
E = []
for i in range(30):
    E.append(model.SR_step(iter, therm_factor=.01, reg=100*.9**i))

plt.figure()
plt.plot(range(30), E)
plt.xlabel('Iteration')
plt.ylabel('$E_{loc}$')
plt.title('Energy vs. Iteration')
plt.show()
>>>>>>> 35da1c9ebdc13cc4e89d42c9adab6f41b1e8ad97

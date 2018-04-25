import numpy as np
from nn_quantum_states import *
from nn_quantum_states.networks.rbm import RBM
from nn_quantum_states.hamiltonians.ising1d import Ising1D
import matplotlib.pyplot as plt

num_spins = 4
alpha = 2
iter = 20000

hami = Ising1D("ising1D", num_spins, h=1)
model = RBM(num_spins, alpha*num_spins, hami)
spins = np.random.randint(2, size=model.num_vis)
spins[spins == 0] = -1
model.set_effective_angles(spins.reshape((num_spins, 1)))
E = []
for i in range(30):
    E.append(model.SR_step(iter, therm_factor=.01, reg=100*.9**i))
plt.plot(range(30), E)

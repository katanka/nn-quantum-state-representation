from nn_quantum_states.hamiltonians.ising1d import Ising1D
from nn_quantum_states.hyperparameters.optimizer import Optimizer

import sys

# Architecture
# num_spins = 4


# Hyperparameters
# alpha = 2
# iterations = 1000 # per spin
# epochs = 40 # number of times to apply gradient descent step to spin
# learning_rate = 1
# therm_factor = 0.01 # proportion of iterations to use on talization
# sweep_factor = 1 # ???
#

free_params_initial = {
    'learning_rate': 1
}

if len(sys.argv) == 2:
    free_params_initial['learning_rate'] = float(sys.argv[1])
    print('Set learning rate to %f' % float(sys.argv[1]))

fixed_params = {
    'num_spins': 4,
    'alpha': 2,
    'iterations': 1000,
    'epochs': 1000,
    'therm_factor': 0.01,
    'sweep_factor': 1
}

params = {**free_params_initial, **fixed_params}


hamiltonian = Ising1D('Ising Model', params['num_spins'])
# hamiltonian = Heisen1D('Hamiltonian Model', fixed_params['num_spins'])


print(Optimizer.evaluate(hamiltonian, **params, err=True, plot=('ground_state_search_err_%.4f.pdf' % params['learning_rate']).replace('.', '_', 1)))

# best_values = Optimizer.optimize_hyperparams(hamiltonian, free_params_initial, fixed_params, plot=True)

# print(best_values)



# # alphas = np.arange(2, 16, 1/2)
# errs = []
#
# for _ in range(10):
#     model = RBM(num_spins, alpha * num_spins, hami, lr=learning_rate)
#     energies = np.array(model.optimize(epochs, iterations, therm_factor, sweep_factor))
#     errs.append(evaluate(energies))
#
#
#
# plt.figure()
# for err in errs:
#     plt.plot(range(epochs), err)
# plt.xlabel('Iteration')
# plt.ylabel('$|E^* - \hat{E}|$')
# plt.title('Error vs. Iteration')
# plt.savefig('alpha_2_iters.pdf')
# plt.show()

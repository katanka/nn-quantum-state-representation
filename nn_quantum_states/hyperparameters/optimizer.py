import random

from rbm import RBM
import numpy as np

from nn_quantum_states.visualization.plots import Plots


class Optimizer(object):

    @staticmethod
    def get_energies(hamiltonian, num_spins, alpha, iterations, epochs, learning_rate, therm_factor, sweep_factor,
                     plot=False, err=False):
        model = RBM(num_spins, alpha * num_spins, hamiltonian, lr=learning_rate)
        energies = model.optimize(epochs, iterations, therm_factor, sweep_factor)

        if plot:
            Plots.plot_ground_state_search(energies, learning_rate, filename=plot, err=err)

        return energies

    @staticmethod
    def evaluate(hamiltonian, num_spins, alpha, iterations, epochs, learning_rate, therm_factor, sweep_factor,
                 plot=False, err=False):
        energies = Optimizer.get_energies(hamiltonian, num_spins, alpha, iterations, epochs, learning_rate,
                                          therm_factor, sweep_factor, plot=plot, err=err)
        true_ground_energy = -2.1357792050698565  # ising
        return np.abs(energies - true_ground_energy)

    @staticmethod
    def optimize_hyperparams(hamiltonian, free_params_initial, fixed_params, min_value=0.001, max_value=2.0,
                             num_sweeps=1, values_per_sweep=50, samples_per_value=20, plot=False):

        params = {**free_params_initial, **fixed_params}
        min_err = 1

        for i in range(num_sweeps):
            # pick random parameter to vary
            variable = random.choice(list(free_params_initial))

            # find range to sweep
            param_range = np.linspace(min_value, max_value, num=values_per_sweep)

            # sweep and find min
            measured_errs = []

            for value in param_range:
                params[variable] = value
                errs = []

                for i in range(samples_per_value):
                    error = Optimizer.evaluate(hamiltonian, **params)[-1]
                    errs.append(error)

                mean_err = np.mean(errs)
                measured_errs.append(mean_err)

                print('%f: %f' % (value, mean_err))

            best_value = param_range[np.argmin(measured_errs)]
            params[variable] = best_value

            min_err = min(measured_errs)
            print('Set %s to %f: E = %f' % (variable, best_value, min_err))

        if plot:
            Plots.plot_learning_rate_sweep(param_range, measured_errs)

        return (params, min_err)

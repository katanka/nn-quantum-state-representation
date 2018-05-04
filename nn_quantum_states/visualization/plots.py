import matplotlib.pyplot as plt
import numpy as np

class Plots(object):

    @staticmethod
    def plot_ground_state_search(energies, learning_rate, filename='ground_state_search.pdf', err=False):
        plt.figure()
        if err:
            true_ground_energy = -2.1357792050698565  # ising
            errs = np.abs(energies - true_ground_energy)
            filter_len = 10
            errs_filtered = np.convolve(errs, np.ones((filter_len,))/filter_len, mode='valid')
            plt.semilogy(range(len(errs_filtered)), errs_filtered)
            plt.ylabel('Error')
        else:
            plt.plot(range(len(energies)), energies)
            plt.ylabel('Energy')
        plt.xlabel('Iteration')
        plt.title('%s vs. Iteration (lr: %.4f)' % ('Error' if err else 'Energy', learning_rate))
        plt.savefig('../img/%s' % filename)
        # plt.show()

    @staticmethod
    def plot_learning_rate_sweep(values, errors):
        plt.figure()
        plt.semilogy(values, errors)
        plt.xlabel('Learning Rate')
        plt.ylabel('Error')
        plt.title('Error vs. Learning Rate (Ising Model)')
        plt.savefig('learning_rate_sweep.pdf')
        # plt.show()
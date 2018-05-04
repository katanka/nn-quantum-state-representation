import matplotlib.pyplot as plt
import numpy as np


def plot_search(energies, true_val=None, save_name=None):
    plt.figure()
    plt.ylabel('Energy')
    plt.xlabel('Iteration')
    plt.title('Ground State Energy vs. Iteration')


    if true_val is not None:
        plt.plot(range(len(energies)), energies, label='Predicted Value')
        plt.plot(range(len(energies)), [true_val for _ in range(len(energies))], dashes=[4, 2], label='Exact Value')
        plt.legend()
    else:
        plt.plot(range(len(energies)), energies)

    if save_name is not None:
        plt.savefig(save_name)

    plt.show()


def plot_search_error(energies, true_val, save_name=None):
    errors = np.abs(energies - true_val)

    plt.figure()
    plt.semilogy(range(len(errors)), errors)
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.title('Ground State Energy Error vs. Iteration')
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
